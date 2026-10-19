"""Bellman learning, causal calibration and the complete empirical RL stage."""

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import subprocess
import sys
import numpy as np
from bidpace import model_based as mb
from bidpace.replay import ReplayData, ReplayEnv, make_replay_day, rollout_replay
import test_control as fixtures


class EmpiricalTests(unittest.TestCase):
    def test_unified_cli_runs_all_stages_and_both_exports(self):
        with tempfile.TemporaryDirectory(prefix="bidpace cli ") as td:
            root = Path(td)
            cfg = fixtures.ControlTests().config(fixtures.fixture(root))
            cfg["seeds"] = [11]
            cfg["workers"] = 1
            cfg["dqn"].update(episodes=4, checkpoints=[0, 4])
            cfg["ppo"].update(episodes=4, checkpoints=[0, 4])
            config = root / "configuration.json"
            config.write_text(json.dumps(cfg))
            output = root / "experiment"
            common = [sys.executable, str(Path("run.py").resolve())]
            common_options = ["--config", str(config), "--out", str(output)]
            commands = [
                ["all"],
                ["export", "--agent", str(root / "primary.npz")],
                ["export", "--variant", "empirical_q", "--agent", str(root / "empirical.npz")],
                ["demo", "--agent", str(root / "empirical.npz")],
            ]
            for command in commands:
                result = subprocess.run(
                    common + command + common_options, capture_output=True, text=True, timeout=120
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((output / "final_evaluation.csv").exists())
            self.assertTrue((output / "candidate_comparisons.json").exists())
            self.assertTrue((output / "comparison.png").exists())

    def test_bellman_saves_budget_for_more_valuable_future(self):
        cost = np.zeros((24, 2))
        reward = np.zeros_like(cost)
        cost[0, 1] = cost[1, 1] = 1.0
        reward[0, 1] = 1.0
        reward[1, 1] = 3.0
        q = mb.bellman_table(cost, reward, np.array([0.0, 1.0, 2.0]))
        np.testing.assert_allclose(q[:, 0], 0.0)
        np.testing.assert_allclose(q[0, 1], [3.0, 1.0])
        self.assertEqual(q[0, 2, 1], 4.0)
        self.assertTrue(np.all(np.diff(q, axis=1) >= -1e-6))

    def test_partial_fulfillment_and_terminal_reward(self):
        cost = np.zeros((24, 2))
        reward = np.zeros_like(cost)
        cost[23, 1] = 2.0
        reward[23, 1] = 4.0
        q = mb.bellman_table(cost, reward, np.array([0.0, 0.5, 1.0, 2.0]))
        np.testing.assert_allclose(q[23, :, 1], [0.0, 1.0, 2.0, 4.0])

    def test_unobserved_campaign_hours_receive_training_only_prior(self):
        with tempfile.TemporaryDirectory() as td:
            path = fixtures.fixture(Path(td))
            with np.load(path) as z:
                arrays = {k: z[k] for k in z.files}
            missing = (arrays["keys"][:, 0] == 20200101) & (arrays["keys"][:, 2] < 12)
            arrays["ends"][missing] = arrays["starts"][missing]
            np.savez_compressed(path, **arrays)
            data = ReplayData(path, [20200101], [20200102])
            result = mb.fit_tables(data, 11, (0.0,))
            spec = mb.control_spec(data)
            self.assertEqual(spec["model_inventory"][0][0], 18.0)
            q = result["tables"][0.0][0]
            self.assertGreater(q[0, -1, -1], q[0, -1, 0])

    def test_inventory_estimator_uses_completed_public_feedback(self):
        with tempfile.TemporaryDirectory() as td:
            data = ReplayData(fixtures.fixture(Path(td)), [20200101], [20200102])
            fitted = mb.fit_tables(data, 11, (0.0,))
            spec = mb.control_spec(data)
            policy = mb.EmpiricalPolicy(
                dict(q=fitted["tables"][0.0], grid=fitted["grid"], actions=fitted["actions"]), spec
            )
            day = make_replay_day(data.templates["validation"][0])
            env = ReplayEnv(day, True)
            policy.reset(
                SimpleNamespace(profile=day.profile, budget=day.budget, advertiser=day.advertiser)
            )
            alpha = policy.act(env.obs())
            self.assertGreater(alpha, 0.0)
            env.step(alpha)
            policy.act(env.obs())
            self.assertAlmostEqual(policy.seen, len(day.hours[0].prices), places=4)
            self.assertAlmostEqual(policy.traffic, 1.0, places=5)

    def test_fit_ignores_heldout_prices_and_clicks(self):
        with tempfile.TemporaryDirectory() as td:
            path = fixtures.fixture(Path(td))
            data = ReplayData(path, [20200101, 20200102], [20200103])
            expected = mb.fit_tables(data, 11, (0.0, 0.5, 1.0))
            arrays = {k: v.copy() for k, v in data.arrays.items()}
            for key, start, end in zip(arrays["keys"], arrays["starts"], arrays["ends"]):
                if key[0] > 20200102:
                    arrays["prices"][start:end] = 999999
                    arrays["label_dates"][start:end] = key[0]
            np.savez_compressed(path, **arrays)
            actual = mb.fit_tables(ReplayData(path, [20200101, 20200102], [20200103]), 11)
            for weight in mb.WEIGHTS:
                np.testing.assert_array_equal(expected["tables"][weight], actual["tables"][weight])

    def test_bootstrap_reproducible_and_causal_policy_has_hard_budget(self):
        with tempfile.TemporaryDirectory() as td:
            data = ReplayData(fixtures.fixture(Path(td)), [20200101, 20200102], [20200103])
            a = mb.fit_tables(data, 23, (0.5,))
            b = mb.fit_tables(data, 23, (0.5,))
            np.testing.assert_array_equal(a["tables"][0.5], b["tables"][0.5])
            tensors = dict(q=a["tables"][0.5], grid=a["grid"], actions=a["actions"])
            policy = mb.EmpiricalPolicy(tensors, mb.control_spec(data))
            day = make_replay_day(
                data.templates["validation"][0], budget_fraction=0.15, competition=1.5
            )
            policy.reset(
                SimpleNamespace(profile=day.profile, budget=day.budget, advertiser=day.advertiser)
            )
            self.assertTrue(np.isfinite(policy.act(ReplayEnv(day, True).obs())))
            result = rollout_replay(day, policy)
            self.assertLessEqual(result["spent"], day.budget + 1e-7)
            self.assertEqual(result["violations"], 0)

    def test_complete_final_stage_selection_export_and_integrity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = fixtures.ControlTests().config(fixtures.fixture(root))
            out = root / "results"
            # Keep this integration test small while exercising real learning,
            # model selection, serialization, exact replay and verification.
            cfg["variants"] = ["traffic_dqn"]
            cfg["seeds"] = [11]
            cfg["workers"] = 1
            cfg["dqn"].update(episodes=4, checkpoints=[0, 4])
            with contextlib.redirect_stdout(io.StringIO()):
                mb.base.train(cfg, out)
                saved = mb.select(cfg, out)
                self.assertEqual(saved["phase"], "selected")
                self.assertFalse((out / "evaluation.csv").exists())
                mb.evaluate(cfg, out)
                mb.verify(cfg, out)
                mb.export(cfg, out, root / "agent.npz")
                mb.demo(cfg, out, root / "agent.npz")
            self.assertTrue((out / "final_summary.csv").is_file())
            changed = dict(cfg, selection_penalty=0.7)
            with self.assertRaises(ValueError):
                mb.load_final(changed, out)
            with np.load(out / "empirical_models.npz") as z:
                arrays = {k: z[k] for k in z.files}
            arrays["ensemble__q"] = arrays["ensemble__q"] + 1.0
            np.savez_compressed(out / "empirical_models.npz", **arrays)
            with self.assertRaisesRegex(ValueError, "empirical models"):
                mb.load_final(cfg, out)


if __name__ == "__main__":
    unittest.main()
