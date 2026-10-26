"""Exact sampled transitions, chronology and delayed-reward regression checks."""

from pathlib import Path
from types import SimpleNamespace
import contextlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from bidpace import tabular
from bidpace import tabular_study as study
from bidpace.replay import ReplayData, ReplayEnv, make_replay_day, affordable_indices
import test_control as fixtures


class TabularTests(unittest.TestCase):
    def test_study_selects_before_test_and_protects_export(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = fixtures.fixture(root)
            cfg = fixtures.ControlTests().config(path)
            frozen = dict(configs={"1": cfg}, data_sha256={"1": study.sha(path)},
                          source_sha256={}, disclosure="synthetic integration fixture")
            fit = tabular.fit
            with contextlib.redirect_stdout(io.StringIO()), patch.object(
                tabular, "fit", side_effect=lambda data: fit(
                    data, np.linspace(0, 2, 41), np.array([0, .3, .8, 1.5, 3.0]))
            ):
                selected = study.select(frozen, 1, root)
                self.assertFalse((root / "season1/evaluation.csv").exists())
                study.select(frozen, 1, root)  # unchanged completed selection is reused
            metadata = json.loads(str(np.load(root / "season1/agent.npz")["metadata"]))
            self.assertEqual(metadata["settings"], selected["settings"])
            self.assertEqual(metadata["development_dates"], cfg["development_dates"])
            changed = dict(selected, reference="privileged_linear")
            with self.assertRaisesRegex(ValueError, "Reference"):
                study.check_selection(root / "season1", changed)
            with (root / "season1/agent.npz").open("ab") as output:
                output.write(b"changed")
            with self.assertRaisesRegex(ValueError, "artifact"):
                study.check_selection(root / "season1", selected)

    def test_batched_transitions_equal_literal_ordered_replay(self):
        rng = np.random.default_rng(72)
        for _ in range(15):
            prices = rng.integers(0, 20, 80, dtype=np.uint32)
            reward = rng.random((80, 3))
            budgets = np.r_[0, np.arange(0.5, 120, 2)]
            thresholds = [-1, 0, 2, 7, 20]
            cost, earned = tabular.ordered_outcomes(prices, reward, thresholds, budgets)
            for b, budget in enumerate(budgets):
                for a, threshold in enumerate(thresholds):
                    ids = np.flatnonzero(prices <= threshold)
                    picked = affordable_indices(prices, ids, budget) if budget > 0 else []
                    self.assertEqual(cost[b, a], prices[picked].sum())
                    np.testing.assert_allclose(earned[b, a], reward[picked].sum(axis=0), atol=1e-12)

    def test_fitted_policy_is_causal_and_ignores_heldout_labels(self):
        with tempfile.TemporaryDirectory() as td:
            path = fixtures.fixture(Path(td))
            data = ReplayData(path, [20200101, 20200102], [20200103])
            grid = np.linspace(0, 2, 41)
            actions = np.array([0, 0.3, 0.8, 1.5, 3.0])
            tensors, spec = tabular.fit(data, grid, actions)
            arrays = {k: v.copy() for k, v in data.arrays.items()}
            for key, start, end in zip(arrays["keys"], arrays["starts"], arrays["ends"]):
                if key[0] > 20200102:
                    arrays["prices"][start:end] = 99999
                    arrays["label_dates"][start:end] = key[0]
            np.savez_compressed(path, **arrays)
            other, _ = tabular.fit(ReplayData(path, [20200101, 20200102], [20200103]), grid, actions)
            np.testing.assert_array_equal(tensors["q"], other["q"])
            np.testing.assert_array_equal(tensors["q"][:, :, :, 0], 0)
            policy = tabular.TabularPolicy(tensors, spec)
            day = make_replay_day(data.templates["validation"][0])
            policy.reset(SimpleNamespace(profile=day.profile, budget=day.budget,
                                         advertiser=day.advertiser))
            env = ReplayEnv(day, features=True)
            while not env.done:
                env.step(policy.act(env.obs()))
            self.assertLessEqual(env.spent, day.budget)

    def test_backup_reserves_budget_for_later_click(self):
        profile = SimpleNamespace(price=1.0, ctr=np.ones(24), mean_ctr=1.0,
                                  reference_cost=2.0, mean_daily_impressions=2.0)
        empty = SimpleNamespace(prices=np.array([], dtype=np.uint32), clicks=np.array([]))
        hours = [empty] * 24
        hours[0] = SimpleNamespace(prices=np.array([1], dtype=np.uint32), clicks=np.array([0]))
        hours[1] = SimpleNamespace(prices=np.array([1], dtype=np.uint32), clicks=np.array([1]))
        # Observed zero-price, zero-click arrivals keep subsequent hours observed.
        for h in range(2, 24):
            hours[h] = SimpleNamespace(prices=np.array([0], dtype=np.uint32), clicks=np.array([0]))
        data = SimpleNamespace(profiles={1: profile}, templates={"train": [
            SimpleNamespace(advertiser=1, hours=hours)]})
        tensors, _ = tabular.fit(data, np.array([0, 0.5, 1.0]), np.array([0, 1.0]))
        np.testing.assert_allclose(tensors["q"][2, 0, 0, 1], [1, 0])

    def test_sampled_backup_does_not_spend_the_mean_hour_cost(self):
        profile = SimpleNamespace(price=1.0, ctr=np.ones(24), mean_ctr=1.0,
                                  reference_cost=1.0, mean_daily_impressions=24.0)
        unbuyable = SimpleNamespace(prices=np.array([100], dtype=np.uint32), clicks=np.array([0]))
        a, b = [unbuyable] * 24, [unbuyable] * 24
        a[0] = SimpleNamespace(prices=np.array([2], dtype=np.uint32), clicks=np.array([1]))
        b[0] = SimpleNamespace(prices=np.array([], dtype=np.uint32), clicks=np.array([]))
        data = SimpleNamespace(profiles={1: profile}, templates={"train": [
            SimpleNamespace(advertiser=1, hours=a), SimpleNamespace(advertiser=1, hours=b)]})
        tensors, _ = tabular.fit(data, np.array([0, 1.0, 2.0]), np.array([0, 3.0]))
        # Mean cost is 1, but neither sampled hour buys a click at budget 1.
        self.assertEqual(tensors["q"][2, 0, 0, 1, 1], 0)
        self.assertEqual(tensors["q"][2, 0, 0, 2, 1], 0.5)


if __name__ == "__main__":
    unittest.main()
