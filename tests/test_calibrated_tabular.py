"""Model-consistent traffic feedback and cached-table experiment integrity."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from bidpace import tabular, tabular_study as base
from bidpace.replay import ReplayData, ReplayEnv, make_replay_day
from experiments import calibrated_tabular as calibrated
import test_control as fixtures


class CalibrationTests(unittest.TestCase):
    def test_public_ablation_matches_upstream_and_model_feedback_is_consistent(self):
        with tempfile.TemporaryDirectory() as td:
            data = ReplayData(fixtures.fixture(Path(td)), [20200101], [20200102])
            tensors, original = tabular.fit(data, np.linspace(0, 2, 41), np.array([0, .3, 1, 3]))
            spec = calibrated.inventory_spec(data, original)
            day = make_replay_day(data.templates["validation"][0])
            a = tabular.TabularPolicy(tensors, original)
            b = calibrated.CalibratedPolicy(tensors, spec, traffic="public")
            a.reset(day)
            b.reset(day)
            env = ReplayEnv(day, features=True)
            while not env.done:
                obs = env.obs()
                alpha = a.act(obs)
                self.assertEqual(alpha, b.act(obs))
                env.step(alpha)
            c = calibrated.CalibratedPolicy(tensors, spec)
            c.reset(day)
            env = ReplayEnv(day, features=True)
            env.step(c.act(env.obs()))
            c.act(env.obs())
            self.assertAlmostEqual(c.seen, len(day.hours[0].prices), places=4)
            self.assertEqual(c.expected_seen, spec["model_inventory"][0][0])

    def test_recalibration_selects_and_exports_before_test(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = fixtures.fixture(root)
            cfg = fixtures.ControlTests().config(path)
            original = dict(configs={"1": cfg}, data_sha256={"1": base.sha(path)}, source_sha256={})
            fit = tabular.fit
            source, out = root / "original", root / "calibrated"
            with contextlib.redirect_stdout(io.StringIO()), patch.object(
                tabular, "fit", side_effect=lambda data: fit(
                    data, np.linspace(0, 2, 41), np.array([0, .3, 1, 3]))
            ):
                base.select(original, 1, source)
                study = dict(configs={"1": cfg}, source=str(source))
                calibrated.select(study, 1, out)
                self.assertFalse((out / "season1/evaluation.csv").exists())
                saved = calibrated.check_selection(out / "season1")
                self.assertEqual(saved["settings"], calibrated.CANDIDATES[saved["candidate"]])
                with (out / "season1/agent.npz").open("ab") as output:
                    output.write(b"changed")
                with self.assertRaisesRegex(ValueError, "artifact"):
                    calibrated.check_selection(out / "season1")


if __name__ == "__main__":
    unittest.main()
