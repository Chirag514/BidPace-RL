"""Terminal advantage estimates, action bounds and clipped actor gradients."""

import unittest
import numpy as np
from bidpace.ppo import Actor, bid_from_latent, gae, log_probability


class PPOTests(unittest.TestCase):
    def test_terminal_advantages(self):
        a, ret = gae([1, 2, 3], [0.5, 0.7, 0.9, 100], [False, True, True], gamma=1, lam=1)
        np.testing.assert_allclose(ret, [3, 2, 3])
        np.testing.assert_allclose(a, [2.5, 1.3, 2.1])

    def test_continuous_bid_bounds(self):
        b = bid_from_latent(np.array([-100, 0, 100]))
        self.assertTrue(np.all((b > 0) & (b <= 3)))
        self.assertAlmostEqual(float(bid_from_latent(np.log(0.5))), 1)

    def test_actor_gradients_finite_differences(self):
        rng = np.random.default_rng(6)
        actor = Actor(4, hidden=(3, 2), input_dim=2, output_dim=1)
        actor.p = {k: v.astype(float) for k, v in actor.p.items()}
        actor.p["b0"][:] = 0.4
        actor.p["b1"][:] = 0.3
        x = rng.normal(size=(5, 2)) * 0.1
        mean = actor(x)[:, 0]
        u = mean + np.array([0.1, -0.2, 0.3, -0.1, 0.2])
        old = log_probability(u, mean, 0.4)
        advantage = np.array([1, -0.5, 0.6, -0.2, 0.3])
        _, g, _ = actor.policy_gradients(x, u, old, advantage, 0.4)
        for key, parameter in actor.p.items():
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                eps = 1e-5
                parameter[index] = original + eps
                plus = actor.policy_gradients(x, u, old, advantage, 0.4)[0]
                parameter[index] = original - eps
                minus = actor.policy_gradients(x, u, old, advantage, 0.4)[0]
                parameter[index] = original
                self.assertAlmostEqual(g[key][index], (plus - minus) / (2 * eps), places=6)

    def test_clipped_advantage_has_zero_gradient(self):
        actor = Actor(3, input_dim=2, output_dim=1)
        x = np.zeros((1, 2))
        u = np.array([0.0])
        old = log_probability(u, actor(x)[:, 0], 0.3) - np.log(1.5)
        _, g, _ = actor.policy_gradients(x, u, old, np.array([1.0]), 0.3)
        for value in g.values():
            np.testing.assert_array_equal(value, 0)


if __name__ == "__main__":
    unittest.main()
