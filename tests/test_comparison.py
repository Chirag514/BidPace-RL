"""Imported-model chronology, selection, export and full benchmark replay."""

import contextlib
import io
import json
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import build_release as release
from bidpace import protocol, tabular, tabular_study as base
from experiments import calibrated_tabular as calibrated, compare as multi
import test_control as fixtures


class ComparisonTests(unittest.TestCase):
    def test_imported_candidates_select_before_test_and_export_every_method(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = fixtures.fixture(root)
            cfg = fixtures.ControlTests().config(data)
            cfg.update(variants=list(multi.RL[:2]), seeds=[11], workers=1)
            cfg["dqn"].update(episodes=2, checkpoints=[0, 2])
            neural, tables, cal, out = [root / name for name in ("neural", "tables", "calibrated", "comparison")]
            study = dict(configs={"1": cfg}, source_sha256=protocol.source_hashes(),
                         data_sha256={"1": base.sha(data)})
            tables.mkdir()
            protocol.dump(tables / "study.json", study)
            fit = tabular.fit
            with contextlib.redirect_stdout(io.StringIO()), patch.object(
                tabular, "fit", side_effect=lambda d: fit(d, np.linspace(0, 2, 41), np.array([0, .3, 1, 3]))
            ):
                protocol.train(cfg, neural / "season1")
                base.select(study, 1, tables)
                cal_study = calibrated.freeze(tables, cal)
                calibrated.select(cal_study, 1, cal)
                frozen = multi.freeze(neural, tables, cal, out)
                multi.select(frozen, "1", out)
                self.assertFalse((out / "season1/evaluation.csv").exists())
                for variant in multi.RL:
                    self.assertTrue((out / "season1" / f"{variant}.npz").exists())
                    self.assertTrue(multi.load_agent(out / "season1" / f"{variant}.npz").requires_traffic)
                multi.shared_selection(frozen, out)
                multi.evaluate(frozen, "1", out)
                multi.evaluate(frozen, "1", out, verify=True)
                multi.summarize(frozen, out)
                # A release must reproduce its test outcomes without any source
                # training directory, including after equivalent NPZ repackaging.
                published = root / "release"
                agents = published / "agents/comparison"
                evidence = published / "evidence"
                agents.mkdir(parents=True)
                evidence.mkdir()
                paths = [Path("agents/comparison") / f"season1_{name}.npz" for name in multi.RL]
                for name, path in zip(multi.RL, paths):
                    shutil.copy2(out / "season1" / f"{name}.npz", published / path)
                for path in release.EVIDENCE:
                    if path.name in ("validation.csv", "selection.json"):
                        continue
                    shutil.copy2(out / path.name, published / path)
                pd.read_csv(out / "season1/validation.csv").assign(season=1).to_csv(evidence / "validation.csv", index=False)
                protocol.dump(evidence / "selection.json", {"1": base.read(out / "season1/selection.json")})
                protocol.dump(agents / "manifest.json", {
                    path.as_posix(): base.sha(published / path) for path in paths + release.EVIDENCE
                })
                prepared = published / "data/season1/replay.npz"
                prepared.parent.mkdir(parents=True)
                arrays = base.load_arrays(data)
                np.savez(prepared, **arrays)  # different archive bytes, identical arrays
                self.assertNotEqual(base.sha(prepared), base.sha(data))
                for source in (neural, tables, cal):
                    source.rename(source.with_name(source.name + "-unavailable"))
                with patch.object(release, "AGENTS", paths):
                    replayed = multi.replay_published(published / "data", evidence, agents)
                    self.assertEqual(len(replayed), len(pd.read_csv(out / "evaluation.csv")))
                    arrays["prices"][0] += 1
                    np.savez(prepared, **arrays)
                    with self.assertRaisesRegex(ValueError, "replay data differs"):
                        multi.replay_published(published / "data", evidence, agents)
                # Restore only for the separate import-contract rejection below.
                for source in (neural, tables, cal):
                    source.with_name(source.name + "-unavailable").rename(source)
            result = base.read(out / "result.json")
            self.assertIn("every_rl_beats_every_baseline_all_seasons", result["claims"])
            # A selected model cannot be replaced after recording validation.
            with (out / "season1/campaign_dqn.npz").open("ab") as output:
                output.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "Changed selected"):
                multi.check_selected(out / "season1")
            # A source with a different evaluation timeline cannot be imported.
            changed = json.loads(json.dumps(study))
            changed["configs"]["1"]["test_dates"] = [20200105, 20200106]
            protocol.dump(tables / "study.json", changed)
            with self.assertRaises(ValueError):
                multi.freeze(neural, tables, cal, root / "invalid")

    def test_common_algorithm_selection_normalizes_season_scale(self):
        records = {
            "1": dict(primary="traffic_dqn", reference="linear", validation_scores={
                "traffic_dqn": 1000, "campaign_dqn": 900, "sampled_q": 800, "linear": 1000}),
            "2": dict(primary="campaign_dqn", reference="linear", validation_scores={
                "traffic_dqn": 5, "campaign_dqn": 10, "sampled_q": 8, "linear": 10}),
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for season, saved in records.items():
                (root / f"season{season}").mkdir()
                protocol.dump(root / f"season{season}/selection.json", saved)
            with patch.object(multi, "check_selected", side_effect=lambda p: records[p.name[-1]]):
                saved = multi.shared_selection(dict(configs={"1": {}, "2": {}}), root)
            self.assertEqual(saved["primary"], "campaign_dqn")
            self.assertAlmostEqual(saved["normalized_validation_scores"]["campaign_dqn"], .95)


if __name__ == "__main__":
    unittest.main()
