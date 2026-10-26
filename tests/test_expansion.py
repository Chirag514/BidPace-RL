"""Three-season parsing, explicit campaign context and replication contracts."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
from bidpace.benchmark import configuration, SEASONS
from bidpace.control import CampaignDQN
from bidpace.fetch import selected, PREFIX
from bidpace.replay import prepare, ReplayData, SEASON1_CAMPAIGNS
from test_replay import record
import test_control as fixtures


def first_season(bid, date, kind=1, clicked=None):
    fields = record(bid, date, 1, kind=kind).rstrip().split("\t")[:22]
    fields[21] = next(iter(SEASON1_CAMPAIGNS))
    if clicked is not None:
        fields.extend([str(clicked), "0"])
    return "\t".join(fields) + "\n"


class ExpansionTests(unittest.TestCase):
    def test_all_priced_season_roles_and_test_opt_in(self):
        for season, ordinal, date, test in [
            (1, "1st", "20130311", "20130318_20"),
            (2, "2nd", "20130606", "20130613_15"),
            (3, "3rd", "20131019", "20131021_28"),
        ]:
            self.assertTrue(selected(PREFIX + f"training{ordinal}/imp.{date}.txt.bz2", season, "priced"))
            self.assertFalse(selected(PREFIX + f"training{ordinal}/bid.{date}.txt.bz2", season, "priced"))
            path = PREFIX + f"testing{ordinal}/leaderboard.test.data.{test}.txt.bz2"
            self.assertFalse(selected(path, season, "priced"))
            self.assertTrue(selected(path, season, "priced", True))

    def test_actual_season1_22_and_24_column_layouts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "imp.txt").write_text(first_season("a", 20130311))
            (root / "clk.txt").write_text(first_season("a", 20130311, kind=2))
            (root / "leaderboard.test.data.txt").write_text(first_season("b", 20130318, clicked=1))
            prepare(root, root / "out", workers=1)
            data = ReplayData(root / "out/replay.npz", [20130311], (), [20130318])
            self.assertEqual(list(data.profiles), [1001])
            for split in ("train", "test"):
                self.assertEqual(sum(h.clicks.sum() for t in data.templates[split] for h in t.hours), 1)
            with self.assertRaises(ValueError):
                ReplayData(root / "out/replay.npz", [20130311, 20130318])

    def test_overlapping_season3_leaderboard_cannot_add_training_labels(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "imp.txt").write_text(record("past", 20131022, 1))
            (root / "leaderboard.test.data.txt").write_text(
                record("past", 20131022, 1, attached=1)
                + record("future", 20131025, 1, attached=1))
            audit = prepare(root, root / "out", workers=1, leaderboard_after=20131024)
            data = ReplayData(root / "out/replay.npz", [20131022], (), [20131025])
            self.assertEqual(sum(h.clicks.sum() for h in data.templates["train"][0].hours), 0)
            self.assertEqual(sum(h.clicks.sum() for h in data.templates["test"][0].hours), 1)
            self.assertEqual(sum(s.get("excluded_overlapping_leaderboard_rows", 0)
                                 for s in audit["sources"]), 1)

    def test_missing_date_fails_and_unseen_campaign_is_audited(self):
        with tempfile.TemporaryDirectory() as td:
            path = fixtures.fixture(Path(td))
            with self.assertRaisesRegex(ValueError, "missing"):
                ReplayData(path, [20200101], (), [20200109])
            with np.load(path) as z:
                a = {k: z[k] for k in z.files}
            a["keys"][a["keys"][:, 0] == 20200103, 1] = 2
            np.savez_compressed(path, **a)
            audit = ReplayData(path, [20200101], (), [20200103]).audit()
            self.assertEqual(audit["coverage"]["test"]["unseen_campaign_days"], [[20200103, 2]])
            self.assertEqual(audit["coverage"]["test"]["evaluated_advertiser_days"], 0)

    def test_public_campaign_identity_changes_context_only(self):
        policy = CampaignDQN(None, [1, 2])
        obs = np.arange(13, dtype=np.float32)
        policy.reset(SimpleNamespace(advertiser=1))
        a = policy.encode(obs)
        policy.reset(SimpleNamespace(advertiser=2))
        b = policy.encode(obs)
        np.testing.assert_array_equal(a[:13], b[:13])
        np.testing.assert_array_equal(a[-2:], [1, 0])
        np.testing.assert_array_equal(b[-2:], [0, 1])
        self.assertEqual(len(a), policy.input_dim)
        with self.assertRaises(ValueError):
            policy.reset(SimpleNamespace(advertiser=3))

    def test_season_configuration_preserves_template_and_chronology(self):
        template = json.loads(Path("config.json").read_text())
        for season in SEASONS:
            cfg = configuration(template, season, "data")
            self.assertLess(max(cfg["development_dates"]), min(cfg["test_dates"]))
            self.assertEqual(cfg["variants"], template["variants"])
            cfg["seeds"].append(999)
            self.assertNotIn(999, template["seeds"])




if __name__ == "__main__":
    unittest.main()
