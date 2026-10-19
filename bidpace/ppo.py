"""Original NumPy PPO: causal continuous bidding with a TD value critic."""

import numpy as np
from .agents import MLP


def bid_from_latent(u):
    return 3.0 / (1.0 + np.exp(-np.clip(u, -60, 60)))


def log_probability(u, mean, std):
    return -0.5 * ((u - mean) / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi)


def gae(rewards, values, dones, gamma=1.0, lam=1.0):
    """Terminal masks prevent bootstrapping or advantages crossing episodes."""
    if len(values) != len(rewards) + 1 or len(dones) != len(rewards):
        raise ValueError("GAE requires T rewards, T dones and T+1 values")
    advantage = np.zeros(len(rewards), dtype=np.float64)
    carry = 0.0
    for i in reversed(range(len(rewards))):
        alive = 1.0 - float(dones[i])
        delta = rewards[i] + gamma * alive * values[i + 1] - values[i]
        carry = delta + gamma * lam * alive * carry
        advantage[i] = carry
    return advantage, advantage + np.asarray(values[:-1])


class Actor(MLP):
    def policy_gradients(self, x, u, old_logp, advantage, std, clip=0.2):
        """Clipped surrogate; fixed action-transform Jacobians cancel in ratios."""
        x = np.asarray(x)
        z1 = x @ self.p["w0"] + self.p["b0"]
        a1 = np.maximum(z1, 0)
        z2 = a1 @ self.p["w1"] + self.p["b1"]
        a2 = np.maximum(z2, 0)
        mean = (a2 @ self.p["w2"] + self.p["b2"])[:, 0]
        logratio = log_probability(u, mean, std) - old_logp
        ratio = np.exp(np.clip(logratio, -60, 60))
        loss = -np.minimum(ratio * advantage, np.clip(ratio, 1 - clip, 1 + clip) * advantage).mean()
        active = ((advantage >= 0) & (ratio <= 1 + clip)) | ((advantage < 0) & (ratio >= 1 - clip))
        dq = (-(active * ratio * advantage * (u - mean)) / (std * std * len(x)))[:, None]
        gradients = {"w2": a2.T @ dq, "b2": dq.sum(0)}
        dz2 = dq @ self.p["w2"].T * (z2 > 0)
        gradients.update(w1=a1.T @ dz2, b1=dz2.sum(0))
        dz1 = dz2 @ self.p["w1"].T * (z1 > 0)
        gradients.update(w0=x.T @ dz1, b0=dz1.sum(0))
        return float(loss), gradients, float(np.mean(ratio - 1 - logratio))


class PPOPolicy:
    """Actor sees training profiles and previous feedback, never future prices."""

    def __init__(self, actor, history=False, relative=False):
        self.model = actor
        self.history, self.relative = history, relative
        self.input_dim = 8 + (30 if history else 0) + int(relative)

    def reset(self, day):
        self.profile = day.profile
        self.last_hour = 0
        self.price_history = self.win_history = 0.0
        self.alpha = 1.0

    def encode(self, obs):
        extra = []
        if self.history:
            hour = min(24, int(round(float(obs[1]) * 24)))
            if hour > self.last_hour:
                if hour != self.last_hour + 1:
                    raise ValueError("history requires consecutive observations")
                self.price_history = 0.8 * self.price_history + 0.2 * float(obs[7])
                self.win_history += float(obs[3])
                self.last_hour = hour
            onehot = np.zeros(24)
            if hour < 24:
                onehot[hour] = 1
            remaining = self.profile.volume[hour:].sum()
            value = (
                np.dot(self.profile.volume[hour:], self.profile.ctr[hour:]) / self.profile.mean_ctr
            )
            descriptors = [
                remaining,
                value,
                remaining - float(obs[0]),
                self.price_history,
                self.win_history / max(hour, 1),
                min(float(obs[0] * obs[6]) / max(remaining, 0.02), 4) / 4,
            ]
            extra.extend([onehot, np.asarray(descriptors)])
        if self.relative:
            extra.append(np.asarray([self.alpha / 3]))
        return np.concatenate([obs, *extra]).astype(np.float32)

    def apply(self, u):
        if self.relative:
            logit = np.log(self.alpha / max(3 - self.alpha, 1e-10))
            self.alpha = float(bid_from_latent(logit + 0.5 * np.tanh(u)))
        else:
            self.alpha = float(bid_from_latent(u))
        return self.alpha

    def act(self, obs):
        return self.apply(float(self.model(self.encode(obs))[0]))
