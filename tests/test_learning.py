"""Analytical DQN gradients, Double-Q targets and terminal replay behavior."""

from collections import deque
import unittest
import numpy as np
from bidpace.agents import MLP, Replay, bellman, emit_nstep


class LearningTests(unittest.TestCase):
    def test_dqn_learns_delayed_budget_value(self):
        # Two decision times share one budget. Buying immediately earns one
        # click; waiting allows three clicks at the second decision time.
        first = np.zeros(8, dtype=np.float32)
        first[0] = 1.0
        later = first.copy()
        later[1] = 1.0 / 24
        terminal = np.zeros(8, dtype=np.float32)
        replay = Replay(4)
        for transition in [
            (first, 0, 0.0, later, 1.0),
            (first, 1, 1.0, terminal, 0.0),
            (later, 0, 0.0, terminal, 0.0),
            (later, 1, 3.0, terminal, 0.0),
        ]:
            replay.add(*transition)
        online, target = MLP(5, output_dim=2), MLP(6, output_dim=2)
        target.copy_from(online)
        for update in range(800):
            targets = bellman(online, target, replay.r, replay.ns, replay.discount)
            online.update(replay.s, replay.a, targets, lr=0.002)
            if update % 20 == 0:
                target.copy_from(online)
        self.assertEqual(int(online(first).argmax()), 0)
        self.assertEqual(int(online(later).argmax()), 1)

    def test_variable_dimension_gradients_and_replay(self):
        model = MLP(17, input_dim=14, output_dim=7)
        model.p = {k: v.astype(np.float64) for k, v in model.p.items()}
        x = np.random.default_rng(71).uniform(0.1, 0.8, (5, 14))
        acts, targets = np.arange(5), np.array([0.2, -2, 3, 0.4, 0.1])
        _, gradients = model.gradients(x, acts, targets)
        for key, index in [("w0", (13, 1)), ("w2", (2, 6))]:
            old, delta = model.p[key][index], 0.0001
            model.p[key][index] = old + delta
            plus = model.gradients(x, acts, targets)[0]
            model.p[key][index] = old - delta
            minus = model.gradients(x, acts, targets)[0]
            model.p[key][index] = old
            self.assertAlmostEqual((plus - minus) / (2 * delta), gradients[key][index], delta=2e-5)
        replay = Replay(2, input_dim=14)
        replay.add(x[0], 6, 1.0, x[1], 0.0)
        self.assertEqual(replay.sample(np.random.default_rng(1), 1)[0].shape, (1, 14))

    def test_numerical_gradients(self):
        m = MLP(17)
        m.p = {k: v.astype(np.float64) for k, v in m.p.items()}
        x = np.random.default_rng(71).uniform(0.1, 0.8, (5, 8))
        a = np.array([1, 3, 2, 4, 5])
        y = np.array([0.2, -2, 3, 0.4, 0.1])
        _, g = m.gradients(x, a, y)
        for k in m.p:
            for idx in (tuple((0 for _ in m.p[k].shape)), tuple((s - 1 for s in m.p[k].shape))):
                old = m.p[k][idx]
                eps = 0.0001
                m.p[k][idx] = old + eps
                plus = m.gradients(x, a, y)[0]
                m.p[k][idx] = old - eps
                minus = m.gradients(x, a, y)[0]
                m.p[k][idx] = old
                self.assertAlmostEqual((plus - minus) / (2 * eps), g[k][idx], delta=2e-05)

    def test_optimizer_decreases_loss(self):
        m = MLP(2)
        x = np.ones((16, 8), dtype=np.float32) * 0.2
        a = np.ones(16, dtype=int)
        y = np.ones(16)
        before, _ = m.gradients(x, a, y)
        for _ in range(250):
            m.update(x, a, y)
        self.assertLess(m.gradients(x, a, y)[0], before * 0.1)

    def test_double_and_terminal_targets(self):

        class Net:

            def __init__(self, v):
                self.v = np.array(v)

            def __call__(self, s):
                return np.tile(self.v, (len(s), 1))

        a, b = (Net([9, 1]), Net([2, 8]))
        x = np.zeros((2, 8))
        r = np.array([1.0, 3.0])
        discount = np.array([1.0, 0.0])
        np.testing.assert_array_equal(bellman(a, b, r, x, discount, True), [3.0, 3.0])
        np.testing.assert_array_equal(bellman(a, b, r, x, discount, False), [9.0, 3.0])

    def test_nstep_tails(self):
        r = Replay(10)
        q = deque()
        s = np.zeros(8)
        for i in range(4):
            q.append((s, i, float(i + 1), s, i == 3))
            emit_nstep(q, r, n=3, gamma=0.5, flush=i == 3)
        self.assertEqual(r.size, 4)
        self.assertEqual(len(q), 0)
        np.testing.assert_allclose(r.r[:4], [2.75, 4.5, 5.0, 4.0])
        np.testing.assert_allclose(r.discount[:4], [0.125, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
