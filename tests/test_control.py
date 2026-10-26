"""Causal observations, sampling integrity and anchored reinforcement learning."""

from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
from bidpace.control import fit_control
from bidpace.protocol import (
    create_policy,
    fit_rl,
    Baseline,
    select_iterations,
)
from bidpace.replay import ReplayData, ReplayEnv, make_replay_day, rollout_replay, combine


def fixture(root):
    prices = []
    labels = []
    keys = []
    starts = []
    ends = []
    for date in range(20200101, 20200107):
        for hour in range(24):
            starts.append(len(prices))
            keys.append((date, 1, hour))
            prices.extend([1, 5, 10, 20, 40, 80] * 3)
            labels.extend([date if (hour + i) % 11 == 0 else 0 for i in range(18)])
            ends.append(len(prices))
    path = root / "replay.npz"
    np.savez_compressed(
        path,
        prices=np.asarray(prices, dtype=np.uint32),
        label_dates=np.asarray(labels, dtype=np.int32),
        attached=np.zeros(len(prices), dtype=np.uint8),
        keys=np.asarray(keys, dtype=np.int32),
        starts=np.asarray(starts),
        ends=np.asarray(ends),
    )
    return path


class ControlTests(unittest.TestCase):
    def test_reward_scale_context_distinguishes_otherwise_equal_states(self):
        from dataclasses import replace

        with tempfile.TemporaryDirectory() as td:
            data = ReplayData(fixture(Path(td)), [20200101], [20200102])
            day = make_replay_day(data.templates["validation"][0])
            p = day.profile
            other = replace(day, profile=replace(p, ctr=p.ctr * 2, mean_ctr=p.mean_ctr * 2))
            a, b = ReplayEnv(day, True).obs(), ReplayEnv(other, True).obs()
            np.testing.assert_allclose(a[:-1], b[:-1])
            self.assertGreater(b[-1], a[-1])

    def test_budget_reference_uses_active_campaign_days(self):
        with tempfile.TemporaryDirectory() as td:
            path = fixture(Path(td))
            with np.load(path, allow_pickle=False) as z:
                a = {k: z[k] for k in z.files}
            a["keys"][a["keys"][:, 0] != 20200101, 1] = 2
            np.savez_compressed(path, **a)
            data = ReplayData(path, [20200101, 20200102, 20200103])
            expected = sum(float(h.prices.sum()) for h in data.templates["train"][0].hours)
            self.assertEqual(data.profiles[1].reference_cost, expected)
            self.assertEqual(data.profiles[1].mean_daily_impressions, 432.0)
            fitted = fit_control(data)
            self.assertTrue(np.all(np.diff(fitted["cost_curve"], axis=2) >= 0))

    def test_combine_rejects_overlap_and_preserves_label_dates(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = fixture(root)
            with self.assertRaises(ValueError):
                combine([path, path], root / "bad")
            with np.load(path, allow_pickle=False) as z:
                a = {k: z[k] for k in z.files}
            a["keys"][:, 0] += 100
            a["label_dates"][a["label_dates"] > 0] += 100
            second = root / "second.npz"
            np.savez_compressed(second, **a)
            audit = combine([path, second], root / "combined", [20200106])
            data = ReplayData(root / "combined/replay.npz", [20200101], [20200102])
            self.assertEqual(audit["impressions"], 432 * 11)
            self.assertNotIn(20200106, data.arrays["keys"][:, 0])
            self.assertIn(20200201, data.arrays["label_dates"])

    def test_global_budget_anchor_for_campaign_without_cv_history(self):
        with tempfile.TemporaryDirectory() as td:
            data = ReplayData(fixture(Path(td)), [20200101], [20200102])
            day = make_replay_day(data.templates["validation"][0])
            policy = Baseline(
                dict(
                    kind="budget_linear",
                    fractions=[0.2, 0.3, 0.45],
                    alphas={"999": [0.1, 0.2, 0.3]},
                    fallback=[0.5, 1.0, 2.0],
                )
            )
            policy.reset(day)
            self.assertAlmostEqual(policy.act(ReplayEnv(day).obs()), 1.0)

    def test_robust_selection_uses_paired_date_gains(self):
        import pandas as pd

        baseline = dict(kind="mpc", rate=0.5, reserve=1.0)
        search = [
            dict(policy="mpc", fold=f, score=10.0, spec=json.dumps(baseline, sort_keys=True))
            for f in range(3)
        ]
        rows = []
        for name, values in [
            ("traffic_dqn", [16.0, 10.0, 10.0]),
            ("campaign_dqn", [11.0, 11.0, 11.0]),
        ]:
            rows.extend(
                dict(variant=name, episode=4, seed=-1, fold=f, score=v)
                for f, v in enumerate(values)
            )
        result = select_iterations(
            pd.DataFrame(rows),
            dict(variants=["traffic_dqn", "campaign_dqn"], selection_penalty=0.5),
            dict(baselines={"mpc": baseline}, baseline_search=search),
        )
        self.assertGreater(result["traffic_dqn"]["score"], result["campaign_dqn"]["score"])
        self.assertLess(result["traffic_dqn"]["objective"], result["campaign_dqn"]["objective"])

    def config(self, path):
        cfg = json.loads(Path("config.json").read_text())
        cfg.update(
            data_file=str(path),
            development_dates=[20200101, 20200102, 20200103],
            cv_validation_dates=[20200102, 20200103],
            test_dates=[20200104, 20200105, 20200106],
            seeds=[11, 23],
            workers=2,
            variants=["traffic_dqn", "campaign_dqn"],
            traffic_baselines=True,
        )
        cfg["domain_randomization"]["coverage"] = [0.35, 1.0]
        cfg["dqn"].update(episodes=8, checkpoints=[0, 4, 8], warmup=4, batch_size=4)
        return cfg

    def test_coverage_keeps_prices_and_labels_paired(self):
        with tempfile.TemporaryDirectory() as td:
            data = ReplayData(fixture(Path(td)), [20200101], [20200102])
            template = data.templates["validation"][0]
            a = make_replay_day(template, coverage=0.5, seed=19)
            b = make_replay_day(template, coverage=0.5, seed=19)
            count = 0
            for original, left, right in zip(template.hours, a.hours, b.hours):
                np.testing.assert_array_equal(left.prices, right.prices)
                np.testing.assert_array_equal(left.clicks, right.clicks)
                # The exact original subsequence can be independently reconstructed.
                count += len(left.prices)
                self.assertTrue(all(p in original.prices for p in left.prices))
            rng = np.random.default_rng(19)
            for original, kept in zip(template.hours, a.hours):
                mask = rng.random(len(original.prices)) < 0.5
                np.testing.assert_array_equal(kept.prices, original.prices[mask])
                np.testing.assert_array_equal(kept.clicks, original.clicks[mask])
            self.assertTrue(0 < count < sum(len(h.prices) for h in template.hours))

    def test_observations_use_completed_inventory_only(self):
        from dataclasses import replace
        from bidpace.replay import ReplayHour

        with tempfile.TemporaryDirectory() as td:
            data = ReplayData(fixture(Path(td)), [20200101], [20200102])
            day = make_replay_day(data.templates["validation"][0])
            future = list(day.hours)
            future[23] = ReplayHour(
                np.zeros(10000, dtype=np.uint32), np.ones(10000, dtype=np.uint8), 1.0
            )
            other = replace(day, hours=tuple(future), price_multiplier=999.0)
            a, b = ReplayEnv(day, True), ReplayEnv(other, True)
            np.testing.assert_array_equal(a.obs(), b.obs())
            a.step(0.5)
            b.step(0.5)
            np.testing.assert_array_equal(a.obs(), b.obs())
            self.assertEqual(a.seen, len(day.hours[0].prices))
            self.assertEqual(a.obs().shape, (13,))

    def test_controller_fit_ignores_future_prices_and_labels(self):
        with tempfile.TemporaryDirectory() as td:
            path = fixture(Path(td))
            before = ReplayData(path, [20200101], [20200102])
            expected = fit_control(before)
            arrays = {k: v.copy() for k, v in before.arrays.items()}
            for key, start, end in zip(arrays["keys"], arrays["starts"], arrays["ends"]):
                if key[0] > 20200101:
                    arrays["prices"][start:end] = 99999
                    arrays["label_dates"][start:end] = 20200102
            np.savez_compressed(path, **arrays)
            actual = fit_control(ReplayData(path, [20200101], [20200102]))
            for key in expected:
                np.testing.assert_array_equal(expected[key], actual[key])


    def test_traffic_learners_fixed_checkpoint_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            path = fixture(Path(td))
            data = ReplayData(path, [20200101], [20200102])
            cfg = self.config(path)
            for variant in cfg["variants"]:
                full = fit_rl(data, cfg, variant, 11)
                prefix = fit_rl(data, cfg, variant, 11, stop=4)
                for key in full["snapshots"][4]:
                    np.testing.assert_array_equal(
                        full["snapshots"][4][key], prefix["snapshots"][4][key]
                    )
                policy = create_policy(full["snapshots"][8], full["spec"])
                day = make_replay_day(data.templates["validation"][0])
                policy.reset(day)
                self.assertEqual(
                    policy.encode(ReplayEnv(day, True).obs()).shape, (policy.input_dim,)
                )



if __name__ == "__main__":
    unittest.main()
