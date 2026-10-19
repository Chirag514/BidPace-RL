"""NumPy Q-networks, n-step replay memory and greedy DQN policies."""

import numpy as np

ACTIONS = np.array([0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0])
OBS_DIM = 8


class MLP:
    """Two-hidden-layer Q-network with manual backpropagation and Adam."""

    def __init__(self, seed=0, hidden=(48, 32), input_dim=OBS_DIM, output_dim=len(ACTIONS)):
        rng = np.random.default_rng(seed)
        dims = (input_dim, *hidden, output_dim)
        self.p = {}
        for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
            self.p[f"w{i}"] = (rng.normal(size=(din, dout)) * np.sqrt(2 / din)).astype(np.float32)
            self.p[f"b{i}"] = np.zeros(dout, dtype=np.float32)
        # Start with small Q estimates to avoid arbitrary initial optimism.
        self.p["w2"] *= 0.05
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def __call__(self, x):
        z1 = np.maximum(0, np.asarray(x) @ self.p["w0"] + self.p["b0"])
        z2 = np.maximum(0, z1 @ self.p["w1"] + self.p["b1"])
        return z2 @ self.p["w2"] + self.p["b2"]

    def gradients(self, x, actions, targets):
        x = np.asarray(x, dtype=np.float32)
        z1 = x @ self.p["w0"] + self.p["b0"]
        a1 = np.maximum(z1, 0)
        z2 = a1 @ self.p["w1"] + self.p["b1"]
        a2 = np.maximum(z2, 0)
        q = a2 @ self.p["w2"] + self.p["b2"]
        ids = np.arange(len(x))
        error = q[ids, actions] - targets
        loss = np.where(np.abs(error) <= 1, 0.5 * error**2, np.abs(error) - 0.5).mean()
        # The Huber derivative saturates at +/-1 for large target errors.
        dq = np.zeros_like(q)
        dq[ids, actions] = np.clip(error, -1, 1) / len(x)
        g = {"w2": a2.T @ dq, "b2": dq.sum(axis=0)}
        dz2 = dq @ self.p["w2"].T * (z2 > 0)
        g.update(w1=a1.T @ dz2, b1=dz2.sum(axis=0))
        dz1 = dz2 @ self.p["w1"].T * (z1 > 0)
        g.update(w0=x.T @ dz1, b0=dz1.sum(axis=0))
        return (float(loss), g)

    def update(self, x, actions, targets, lr=0.0005):
        loss, g = self.gradients(x, actions, targets)
        self.apply_gradients(g, lr)
        return loss

    def apply_gradients(self, g, lr):
        """Apply a globally clipped Adam step to externally computed gradients."""
        norm = np.sqrt(sum((float((a * a).sum()) for a in g.values())))
        # Clip the global gradient norm before the Adam update.
        scale = min(1.0, 5.0 / max(norm, 1e-08))
        self.t += 1
        for k in self.p:
            grad = g[k] * scale
            self.m[k] = 0.9 * self.m[k] + 0.1 * grad
            self.v[k] = 0.999 * self.v[k] + 0.001 * grad**2
            self.p[k] -= (
                lr
                * (self.m[k] / (1 - 0.9**self.t))
                / (np.sqrt(self.v[k] / (1 - 0.999**self.t)) + 1e-08)
            )

    def copy_from(self, other):
        self.p = {k: v.copy() for k, v in other.p.items()}


class Replay:
    """Fixed-capacity ring buffer for n-step transitions."""

    def __init__(self, capacity=80000, input_dim=OBS_DIM):
        self.capacity, self.size, self.pos = (capacity, 0, 0)
        self.s = np.zeros((capacity, input_dim), dtype=np.float32)
        self.ns = np.zeros_like(self.s)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.discount = np.zeros(capacity, dtype=np.float32)

    def add(self, s, a, r, ns, discount):
        i = self.pos
        self.s[i], self.a[i], self.r[i], self.ns[i], self.discount[i] = (s, a, r, ns, discount)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, rng, n):
        i = rng.integers(0, self.size, n)
        return (self.s[i], self.a[i], self.r[i], self.ns[i], self.discount[i])


def bellman(online, target, rewards, next_states, discount, double=True):
    """Compute conventional or Double DQN bootstrap targets."""
    values = target(next_states)
    if double:
        a = online(next_states).argmax(axis=1)
        bootstrap = values[np.arange(len(a)), a]
    else:
        bootstrap = values.max(axis=1)
    return rewards + discount * bootstrap


def emit_nstep(queue, replay, n=3, gamma=1.0, flush=False):
    """Emit exactly one transition per original step, including terminal tails."""
    while len(queue) >= n or (flush and queue):
        seq = list(queue)[:n]
        reward = sum((gamma**i * x[2] for i, x in enumerate(seq)))
        terminal = any((x[4] for x in seq))
        replay.add(seq[0][0], seq[0][1], reward, seq[-1][3], 0.0 if terminal else gamma ** len(seq))
        queue.popleft()
        if not flush and len(queue) < n:
            break


class DQNPolicy:
    """Choose an absolute bid multiplier by maximizing learned action values."""

    def __init__(self, model):
        self.model = model
        self.actions = ACTIONS
        self.input_dim = OBS_DIM

    def reset(self, day):
        # The feed-forward controller keeps no private episode state.
        pass

    def encode(self, obs):
        return np.asarray(obs, dtype=np.float32)

    def apply(self, action):
        return float(self.actions[action])

    def act(self, obs):
        return self.apply(int(self.model(self.encode(obs)).argmax()))
