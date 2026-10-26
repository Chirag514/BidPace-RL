"""Temporal selection, uncertainty controls, ensembles and end-to-end replay."""

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from bidpace.protocol import (
    Ensemble,
    folds_for,
    training_controls,
    fit_rl,
    train,
    source_hashes,
    Baseline,
    market_tensors,
)
from bidpace.replay import prepare, ReplayData, make_replay_day, ReplayEnv


def row(bid, date, hour, kind=1, price=20, attached=None):
    fields = ["null"] * 24
    fields[0] = bid
    fields[1] = f"{date}{hour:02d}0000000"
    fields[2] = str(kind)
    fields[20] = str(price)
    fields[22] = "1458"
    if attached is not None:
        fields.extend([str(attached), "0"])
    return "\t".join(fields) + "\n"


class ProtocolTests(unittest.TestCase):
    def config(self):
        return json.loads(Path("config.json").read_text())

    def fixture(self, root):
        (root / "imp.txt").write_text(
            "".join(
                row(f"{d}-{i}", d, i, price=10 + i)
                for d in [20130606, 20130607, 20130608]
                for i in range(5)
            )
        )
        (root / "clk.txt").write_text(
            "".join(row(f"{d}-1", d, 2, kind=2) for d in [20130606, 20130607, 20130608])
        )
        (root / "leaderboard.test.data.txt").write_text(
            "".join(
                row(f"t-{i}", 20130609, i, price=10 + i, attached=int(i == 1)) for i in range(5)
            )
        )
        prepare(root, root / "data", workers=1)
        cfg = self.config()
        cfg.update(
            data_file=str(root / "data/replay.npz"),
            development_dates=[20130606, 20130607, 20130608],
            cv_validation_dates=[20130607, 20130608],
            test_dates=[20130609],
            seeds=[11, 23],
            workers=2,
            variants=["traffic_dqn", "campaign_dqn"],
        )
        cfg["dqn"].update(episodes=8, checkpoints=[0, 4, 8], warmup=4, batch_size=4)
        return cfg

    def test_forward_only_folds_and_reserved_dates(self):
        cfg = self.config()
        for train_dates, validation_dates in folds_for(cfg):
            self.assertLess(max(train_dates), min(validation_dates))
        cfg["cv_validation_dates"] = [cfg["test_dates"][0]]
        with self.assertRaises(ValueError):
            folds_for(cfg)

    def test_randomization_declared_range_and_unseen_stress(self):
        cfg = self.config()
        rng = np.random.default_rng(7)
        values = np.asarray([training_controls(cfg, "traffic_dqn", rng) for _ in range(1000)])
        self.assertGreater(values[:, 0].min(), 0.15)
        self.assertLess(values[:, 0].max(), 0.6)
        self.assertLess(values[:, 1].max(), 1.5)
        a = np.random.default_rng(8)
        b = np.random.default_rng(8)
        training_controls(cfg, "traffic_dqn", a)
        training_controls(cfg, "campaign_dqn", b)
        self.assertEqual(a.random(), b.random())

    def test_q_ensemble_averages_values_before_argmax(self):
        class Policy:
            def __init__(self, q):
                self.q = np.asarray(q)

            def model(self, x):
                return self.q

            def encode(self, x):
                return x

            def apply(self, a):
                return float(a)

        # Majority voting would choose action zero; mean Q chooses action one.
        ensemble = Ensemble([Policy([2, 0]), Policy([2, 0]), Policy([0, 9])], "dqn")
        self.assertEqual(ensemble.act(np.zeros(8)), 1.0)

    def test_fixed_refit_reproduces_checkpoint_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = self.fixture(Path(td))
            data = ReplayData(cfg["data_file"], [20130606, 20130607])
            for variant in ["traffic_dqn", "campaign_dqn"]:
                full = fit_rl(data, cfg, variant, 11, 8)
                short = fit_rl(data, cfg, variant, 11, 4)
                for key in full["snapshots"][4]:
                    np.testing.assert_array_equal(
                        full["snapshots"][4][key], short["snapshots"][4][key]
                    )

    def test_baseline_action_cap_and_market_feedback_use_executed_bid(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = self.fixture(Path(td))
            data = ReplayData(cfg["data_file"], [20130606, 20130607])
            day = make_replay_day(data.templates["train"][0])
            obs = ReplayEnv(day).obs()
            baseline = Baseline(
                dict(kind="pid", alpha=3.0, factor=10.0, parameters=dict(kp=6.0, ki=0.2, kd=1.0))
            )
            baseline.reset(day)
            self.assertLessEqual(baseline.act(obs), 3.0)
            market = Baseline(
                dict(kind="market_linear", alpha=3.0, factor=2.0, rate=0.25), market_tensors(data)
            )
            market.reset(day)
            market.filter.price = 4.0
            action = market.act(obs)
            self.assertEqual(action, 3.0)
            expected = action * day.profile.ctr[0] / day.profile.mean_ctr
            self.assertAlmostEqual(market.filter.previous[1], expected)

    def test_missing_replay_input_gives_actionable_error_without_output(self):
        with tempfile.TemporaryDirectory(prefix="bidpace replay setup ") as td:
            root = Path(td)
            cfg = self.config()
            cfg["data_file"] = str(root / "missing replay.npz")
            out = root / "results"
            with self.assertRaisesRegex(FileNotFoundError, "Missing replay input"):
                train(cfg, out)
            self.assertFalse(out.exists())

    def test_training_freezes_sources_and_resumes_without_replacing_models(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = self.fixture(root)
            out = root / "results"
            train(cfg, out)
            frozen = json.loads((out / "protocol.json").read_text())
            self.assertEqual(frozen["frozen"]["source_sha256"], source_hashes())
            self.assertEqual(frozen["phase"], "trained")
            with self.assertRaises(FileExistsError):
                train(cfg, out)
            # Explicit resume must not refit or replace a completed model archive.
            before = (out / "models.npz").read_bytes()
            train(cfg, out, resume=True)
            self.assertEqual(before, (out / "models.npz").read_bytes())


if __name__ == "__main__":
    unittest.main()
