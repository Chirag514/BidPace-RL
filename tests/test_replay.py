"""Raw replay: joins, dated labels, no look-ahead and budget accounting."""

from pathlib import Path
from dataclasses import replace
import tempfile
import unittest
import numpy as np
from bidpace.replay import (
    prepare,
    ReplayData,
    ReplayEnv,
    ReplayHour,
    make_replay_day,
    affordable_indices,
)


def record(bid, date, hour, kind=1, price=20, attached=None, ms=0):
    fields = ["null"] * 24
    fields[0] = bid
    fields[1] = f"{date}{hour:02d}0000{ms:03d}"
    fields[2] = str(kind)
    fields[20] = str(price)
    fields[22] = "1458"
    if attached is not None:
        fields.extend([str(attached), "0"])
    return "\t".join(fields) + "\n"


class ReplayTests(unittest.TestCase):
    def fixture(self, root):
        (root / "imp.txt").write_text(
            record("a", 20130606, 1)
            + record("a", 20130606, 1, ms=30)
            + record("delayed", 20130607, 23)
            + record("b", 20130608, 2)
            + record("bad", 20130606, 3, price=10)
            + record("bad", 20130606, 3, price=30)
        )
        (root / "clk.txt").write_text(
            record("delayed", 20130608, 0, kind=2) + record("b", 20130608, 3, kind=2)
        )
        (root / "leaderboard.test.data.txt").write_text(record("c", 20130609, 4, attached=1))
        return prepare(root, root / "out", workers=1)

    def test_consistent_timestamp_repeats_and_conflicting_prices(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audit = self.fixture(root)
            self.assertEqual(audit["exported_impressions"], 4)
            self.assertEqual(audit["repeated_impression_records"], 2)
            self.assertEqual(audit["conflicting_price_ids"], 1)
            data = ReplayData(root / "out/replay.npz", [20130606, 20130607], [20130608], [20130609])
            self.assertEqual(
                sum(h.clicks.sum() for t in data.templates["train"] for h in t.hours), 0
            )
            self.assertEqual(
                sum(h.clicks.sum() for t in data.templates["validation"] for h in t.hours), 1
            )
            self.assertEqual(
                sum(h.clicks.sum() for t in data.templates["test"] for h in t.hours), 1
            )

    def test_future_click_is_not_a_training_label(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            path = root / "out/replay.npz"
            before = ReplayData(path, [20130606, 20130607], [20130608], [20130609])
            after = ReplayData(path, [20130606, 20130607, 20130608], (), [20130609])
            self.assertGreater(after.profiles[1458].mean_ctr, before.profiles[1458].mean_ctr)

    def test_future_price_and_label_changes_do_not_change_training_profiles(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            path = root / "out/replay.npz"
            before = ReplayData(path, [20130606, 20130607], [20130608], [20130609])
            arrays = {k: v.copy() for k, v in before.arrays.items()}
            for key, start, end in zip(arrays["keys"], arrays["starts"], arrays["ends"]):
                if key[0] >= 20130608:
                    arrays["prices"][start:end] = 999
                    arrays["label_dates"][start:end] = 20130609
            np.savez_compressed(path, **arrays)
            after = ReplayData(path, [20130606, 20130607], [20130608], [20130609])
            for name in ["ctr", "volume", "price", "reference_cost"]:
                np.testing.assert_array_equal(
                    getattr(before.profiles[1458], name), getattr(after.profiles[1458], name)
                )

    def test_attached_test_labels_cannot_become_training_data(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            with self.assertRaises(ValueError):
                ReplayData(root / "out/replay.npz", [20130606, 20130609])

    def test_affordability_matches_literal_arrival_rule(self):
        rng = np.random.default_rng(18)
        for _ in range(200):
            prices = rng.integers(0, 301, size=500).astype(float) * rng.uniform(0.8, 1.6)
            ids = np.flatnonzero(prices < rng.uniform(0, 400))
            budget = float(rng.uniform(0, prices.sum()))
            expected = []
            left = budget
            for i in ids:
                if prices[i] <= left:
                    expected.append(i)
                    left -= prices[i]
            np.testing.assert_array_equal(affordable_indices(prices, ids, budget), expected)
        prices = np.array([6.0, 4.0, 0.0, 3.0, 1.0, 0.0])
        np.testing.assert_array_equal(affordable_indices(prices, np.arange(6), 10.0), [0, 1, 2, 5])

    def test_replay_reward_and_hard_budget(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            data = ReplayData(
                root / "out/replay.npz", [20130606, 20130607, 20130608], (), [20130609]
            )
            day = make_replay_day(data.templates["test"][0], budget_fraction=0.3)
            env = ReplayEnv(day)
            while not env.done:
                env.step(3.0)
            self.assertLessEqual(env.spent, day.budget)
            self.assertEqual(env.clicks, 0)  # Its single impression costs more than budget.

    def test_future_reused_id_keeps_past_price_and_assigns_click_causally(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "imp.txt").write_text(
                record("reused", 20130606, 1, price=20) + record("reused", 20130608, 1, price=50)
            )
            (root / "clk.txt").write_text(
                record("reused", 20130607, 1, kind=2) + record("reused", 20130608, 2, kind=2)
            )
            audit = prepare(root, root / "out", workers=1)
            self.assertEqual(audit["conflicting_price_ids"], 0)
            early = ReplayData(root / "out/replay.npz", [20130606], [20130608])
            np.testing.assert_array_equal(early.templates["train"][0].hours[1].prices, [20])
            self.assertEqual(early.templates["train"][0].hours[1].clicks.sum(), 0)
            self.assertEqual(early.templates["validation"][0].hours[1].clicks.sum(), 1)
            later = ReplayData(root / "out/replay.npz", [20130606, 20130608])
            self.assertEqual(
                sum(h.clicks.sum() for t in later.templates["train"] for h in t.hours), 2
            )

    def test_stress_preserves_pairing_and_thinning_is_reproducible(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            data = ReplayData(
                root / "out/replay.npz", [20130606, 20130607, 20130608], (), [20130609]
            )
            template = data.templates["test"][0]
            hours = list(template.hours)
            prices = np.arange(1, 101, dtype=np.uint32)
            clicks = (prices % 3 == 0).astype(np.uint8)
            hours[18] = ReplayHour(prices, clicks, hours[18].bid_quality)
            template = replace(template, hours=tuple(hours))
            traffic = make_replay_day(template, traffic=3.0)
            np.testing.assert_array_equal(traffic.hours[18].prices, np.tile(prices, 3))
            np.testing.assert_array_equal(traffic.hours[18].clicks, np.tile(clicks, 3))
            shifted = make_replay_day(template, competition=1.5)
            np.testing.assert_array_equal(shifted.hours[18].clicks, clicks)
            np.testing.assert_array_equal(shifted.hours[18].prices, np.rint(prices * 1.5))
            a = make_replay_day(template, quality=0.5, seed=19)
            b = make_replay_day(template, quality=0.5, seed=19)
            np.testing.assert_array_equal(a.hours[18].clicks, b.hours[18].clicks)
            self.assertTrue(np.all(a.hours[18].clicks <= clicks))
            self.assertGreater(a.hours[18].clicks.sum(), 0)
            self.assertLess(a.hours[18].clicks.sum(), clicks.sum())

    def test_currency_units_do_not_change_actions_wins_or_rewards(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            data = ReplayData(
                root / "out/replay.npz", [20130606, 20130607, 20130608], (), [20130609]
            )
            template = data.templates["test"][0]
            hours = tuple(
                ReplayHour(
                    np.array([1, 5, 20, 40], dtype=np.uint32),
                    np.array([0, 1, 0, 1], dtype=np.uint8),
                    h.bid_quality,
                )
                for h in template.hours
            )
            template = replace(template, hours=hours)
            day = make_replay_day(template, budget_fraction=0.6)
            scale = 7
            profile = replace(
                day.profile,
                price=day.profile.price * scale,
                reference_cost=day.profile.reference_cost * scale,
            )
            scaled = replace(
                day,
                profile=profile,
                budget=day.budget * scale,
                hours=tuple(
                    ReplayHour(h.prices * scale, h.clicks, h.bid_quality * scale) for h in day.hours
                ),
            )
            a, b = ReplayEnv(day), ReplayEnv(scaled)
            while not a.done:
                np.testing.assert_allclose(a.obs(), b.obs(), atol=1e-7)
                alpha = [0.2, 0.5, 1.0, 3.0][a.h % 4]
                _, ra, da = a.step(alpha)
                _, rb, db = b.step(alpha)
                self.assertEqual((ra, da, a.wins), (rb, db, b.wins))
                self.assertAlmostEqual(a.spent * scale, b.spent)


if __name__ == "__main__":
    unittest.main()
