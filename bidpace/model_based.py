"""Empirical model-based RL and final selection across the RL candidates.

The source, input and upstream-model fingerprints are frozen before test replay.
The empirical MDP uses training-only mean rewards/costs; execution always uses the
exact ordered auction simulator. Its within-hour proportional approximation is
not an exact auction model.
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from bidpace import protocol as base
from bidpace.adaptive import MarketFilter
from bidpace.control import fit_control
from bidpace.agents import ACTIONS
from bidpace.replay import ReplayData, make_replay_day, rollout_replay

WEIGHTS = (0.0, 0.5, 1.0)


def bellman_table(cost, reward, grid):
    """Backward value iteration for the empirical finite-horizon budget MDP."""
    q = np.zeros((24, len(grid), cost.shape[1]), dtype=np.float32)
    future = np.zeros(len(grid))
    for hour in reversed(range(24)):
        spend = np.minimum(grid[:, None], cost[hour][None, :])
        fraction = np.ones_like(spend)
        np.divide(spend, cost[hour][None, :], out=fraction, where=cost[hour][None, :] > 0)
        immediate = reward[hour][None, :] * fraction
        continuation = np.interp((grid[:, None] - spend).ravel(), grid, future).reshape(spend.shape)
        q[hour] = immediate + continuation
        q[hour, 0] = 0.0  # The actual environment terminates at an exhausted budget.
        future = q[hour].max(axis=1)
    return q


def fit_tables(data, seed, weights=WEIGHTS):
    """Bootstrap active training days; learn empirical action transition models."""
    rng = np.random.default_rng(seed)
    actions = np.unique(np.r_[ACTIONS, np.geomspace(0.04, 3.0, 25)])
    grid = np.r_[0.0, np.geomspace(1e-5, 2.0, 400)]
    campaigns = sorted(data.profiles)
    tables = {
        w: np.empty((len(campaigns), 24, len(grid), len(actions)), dtype=np.float32)
        for w in weights
    }
    for j, adv in enumerate(campaigns):
        templates = [t for t in data.templates["train"] if t.advertiser == adv]
        chosen = [templates[i] for i in rng.integers(len(templates), size=len(templates))]
        profile = data.profiles[adv]
        # An hour never recorded for a newly launched campaign is unobserved,
        # rather than evidence of zero future opportunity. Its prior is the
        # campaign's pooled observed training hours, including price/label pairs.
        pooled_prices = np.concatenate([h.prices for t in chosen for h in t.hours])
        pooled_labels = np.concatenate([h.clicks for t in chosen for h in t.hours])
        observed_hours = sum(any(len(t.hours[h].prices) for t in chosen) for h in range(24))
        cost = np.empty((24, len(actions)))
        count = np.empty_like(cost)
        clicked = np.empty_like(cost)
        for hour in range(24):
            prices = np.concatenate([t.hours[hour].prices for t in chosen])
            labels = np.concatenate([t.hours[hour].clicks for t in chosen])
            denominator = len(chosen)
            if not len(prices):
                prices = pooled_prices
                labels = pooled_labels
                denominator *= observed_hours
            order = np.argsort(prices, kind="stable")
            prices = prices[order]
            labels = labels[order]
            thresholds = actions * profile.price * profile.ctr[hour] / profile.mean_ctr
            ids = np.searchsorted(prices, thresholds, side="right")
            count[hour] = ids / denominator
            cost[hour] = (
                np.r_[0.0, np.cumsum(prices, dtype=np.float64)][ids]
                / denominator
                / profile.reference_cost
            )
            clicked[hour] = np.r_[0.0, np.cumsum(labels, dtype=np.float64)][ids] / denominator
        cost[:, 0] = 0.0
        count[:, 0] = 0.0
        clicked[:, 0] = 0.0
        for weight in weights:
            reward = weight * clicked + (1 - weight) * profile.ctr[:, None] * count
            tables[weight][j] = bellman_table(cost, reward, grid)
    return dict(tables=tables, grid=grid, actions=actions)


def control_spec(data):
    control = fit_control(data)
    spec = {
        k: v.tolist()
        for k, v in control.items()
        if k in ("campaigns", "market_quantiles", "market_prices")
    }
    inventories = []
    for adv in spec["campaigns"]:
        templates = [t for t in data.templates["train"] if t.advertiser == adv]
        inventory = np.array(
            [sum(len(t.hours[h].prices) for t in templates) / len(templates) for h in range(24)]
        )
        inventory[inventory == 0] = inventory[inventory > 0].mean()
        inventories.append(inventory.tolist())
    spec["model_inventory"] = inventories
    return spec


class EmpiricalPolicy:
    requires_traffic = True

    def __init__(self, tensors, spec):
        self.tensors = tensors
        self.spec = spec
        self.filter = MarketFilter(spec, rate=0.5)

    def reset(self, day):
        self.profile = day.profile
        self.budget = day.budget
        self.j = self.spec["campaigns"].index(day.advertiser)
        self.hour = -1
        self.filter.reset(day)
        self.filter.budget = day.budget / day.profile.price
        self.filter.auctions = day.profile.mean_daily_impressions
        self.inventory = np.asarray(self.spec["model_inventory"][self.j])
        self.seen = self.expected_seen = 0.0
        self.last_left = self.budget
        self.traffic = 1.0

    def act(self, obs):
        hour = min(int(round(float(obs[1]) * 24)), 23)
        if hour != self.hour:
            self.filter.observe(obs)
            if hour > 0:
                # Infer completed inventory from public paid-price/win feedback.
                # Updating both sums only on uncensored feedback avoids treating
                # a no-bid hour as an observed zero-traffic hour.
                win = float(obs[3])
                paid = float(obs[7]) * 4.0 * self.profile.price
                spend = max(0.0, self.last_left - float(obs[0]) * self.budget)
                if win > 1e-8 and paid > 1e-8 and float(obs[7]) < 0.999:
                    self.seen += spend / paid / win
                    self.expected_seen += self.inventory[hour - 1]
            prior = 0.05 * self.inventory.sum()
            self.traffic = float(
                np.clip((self.seen + prior) / (self.expected_seen + prior), 0.05, 4.0)
            )
            self.hour = hour
        budget = (
            float(obs[0])
            * self.budget
            / self.profile.reference_cost
            / (self.filter.price * self.traffic)
        )
        curve = self.tensors["q"][self.j, hour]
        values = np.asarray(
            [np.interp(budget, self.tensors["grid"], curve[:, a]) for a in range(curve.shape[1])]
        )
        alpha = float(
            np.clip(self.tensors["actions"][values.argmax()] * self.filter.price, 0.0, 3.0)
        )
        self.filter.record(obs, alpha)
        self.last_left = float(obs[0]) * self.budget
        return alpha


def worker(task):
    cfg, train, seed, weights = task
    data = ReplayData(cfg["data_file"], train)
    return seed, fit_tables(data, seed, weights)


def average(results, weight):
    return dict(
        q=np.mean([r["tables"][weight] for r in results.values()], axis=0, dtype=np.float64),
        grid=next(iter(results.values()))["grid"],
        actions=next(iter(results.values()))["actions"],
    )


def fingerprint(cfg, out, state):
    return dict(
        config=cfg,
        base_frozen=state["frozen"],
        base_models=hashlib.sha256((out / "models.npz").read_bytes()).hexdigest(),
        base_cv_models=hashlib.sha256((out / "cv_models.npz").read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        candidate_reward_weights=list(WEIGHTS),
    )


def select(cfg, out):
    out = Path(out)
    state = base.load_state(cfg, out)
    frozen = fingerprint(cfg, out, state)
    path = out / "final_protocol.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["frozen"] != frozen:
            raise ValueError("Changed final-stage protocol")
        return saved
    # No reserved data object is constructed in selection.
    records = []
    for fold, (train, validation) in enumerate(base.folds_for(cfg)):
        data = ReplayData(cfg["data_file"], train, validation)
        spec = control_spec(data)
        tasks = [(cfg, train, seed, WEIGHTS) for seed in cfg["seeds"]]
        with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
            results = dict(pool.map(worker, tasks))
        days = base.validation_days(data, cfg)
        for weight in WEIGHTS:
            policy = EmpiricalPolicy(average(results, weight), spec)
            score = base.value(policy, days)
            records.append(
                dict(
                    fold=fold,
                    variant="empirical_q",
                    seed=-1,
                    episode=int(weight * 100),
                    score=score,
                )
            )
            print("Empirical CV", fold, "reward weight", weight, "clicks", score, flush=True)
    cv = pd.DataFrame(records)
    cv.to_csv(out / "empirical_cv.csv", index=False)
    local = dict(cfg, variants=["empirical_q"])
    picked = base.select_iterations(cv, local, state)["empirical_q"]
    weight = picked["episode"] / 100.0
    primary = (
        "empirical_q"
        if picked["objective"]
        > state["selection"]["iterations"][state["selection"]["primary"]]["objective"]
        else state["selection"]["primary"]
    )
    data = ReplayData(cfg["data_file"], cfg["development_dates"])
    spec = control_spec(data)
    tasks = [(cfg, tuple(cfg["development_dates"]), seed, (weight,)) for seed in cfg["seeds"]]
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        results = dict(pool.map(worker, tasks))
    arrays = {}
    for seed, result in results.items():
        for key, value in dict(
            q=result["tables"][weight], grid=result["grid"], actions=result["actions"]
        ).items():
            arrays[f"s{seed}__{key}"] = value
    arrays.update({f"ensemble__{k}": v for k, v in average(results, weight).items()})
    base.save_arrays(out / "empirical_models.npz", arrays)
    saved = dict(
        frozen=frozen,
        primary=primary,
        neural_selection=state["selection"],
        empirical_selection=dict(reward_weight=weight, **picked),
        spec=spec,
        model_sha256=hashlib.sha256((out / "empirical_models.npz").read_bytes()).hexdigest(),
        phase="selected",
    )
    base.dump(path, saved)
    print(
        "FINAL VALIDATION SELECTION:",
        primary,
        "empirical objective:",
        picked["objective"],
        flush=True,
    )
    return saved


def load_final(cfg, out):
    out = Path(out)
    state = base.load_state(cfg, out)
    saved = json.loads((out / "final_protocol.json").read_text())
    if saved["frozen"] != fingerprint(cfg, out, state):
        raise ValueError("Changed final-stage code or inputs")
    if (
        saved["model_sha256"]
        != hashlib.sha256((out / "empirical_models.npz").read_bytes()).hexdigest()
    ):
        raise ValueError("Changed empirical models")
    return saved


def empirical_policies(cfg, out, saved):
    with np.load(Path(out) / "empirical_models.npz", allow_pickle=False) as z:
        return [
            (
                seed,
                EmpiricalPolicy(
                    {k: z[f"s{seed}__{k}"] for k in ("q", "grid", "actions")}, saved["spec"]
                ),
            )
            for seed in cfg["seeds"]
        ] + [
            (
                -1,
                EmpiricalPolicy(
                    {k: z[f"ensemble__{k}"] for k in ("q", "grid", "actions")}, saved["spec"]
                ),
            )
        ]


def evaluate(cfg, out):
    out = Path(out)
    saved = load_final(cfg, out)
    if saved["phase"] == "evaluated":
        return
    base.evaluate(cfg, out)
    base.analyze(cfg, out)
    data = ReplayData(cfg["data_file"], cfg["development_dates"], (), cfg["test_dates"])
    rows = []
    traces = []
    policies = empirical_policies(cfg, out, saved)
    for scenario, changes in base.SCENARIOS.items():
        for i, template in enumerate(data.templates["test"]):
            settings = dict(budget_fraction=cfg["evaluation_budget_fraction"])
            settings.update(changes)
            day = make_replay_day(template, seed=3000000 + i * 10000, **settings)
            for seed, policy in policies:
                result = rollout_replay(day, policy)
                if scenario == "standard":
                    traces.append(
                        dict(
                            scenario=scenario,
                            date=day.date,
                            advertiser=day.advertiser,
                            policy="empirical_q",
                            seed=seed,
                            trace=result["trace"],
                        )
                    )
                result.pop("trace")
                rows.append(
                    dict(
                        scenario=scenario,
                        date=day.date,
                        advertiser=day.advertiser,
                        policy="empirical_q",
                        seed=seed,
                        privileged=False,
                        upper_bound=base.upper_bound(day),
                        **result,
                    )
                )
    raw = pd.concat([pd.read_csv(out / "evaluation.csv"), pd.DataFrame(rows)], ignore_index=True)
    raw.to_csv(out / "final_evaluation.csv", index=False)
    base.dump(out / "final_traces.json", json.loads((out / "traces.json").read_text()) + traces)
    saved.update(
        phase="evaluated",
        evaluation_sha256=hashlib.sha256((out / "final_evaluation.csv").read_bytes()).hexdigest(),
    )
    base.dump(out / "final_protocol.json", saved)
    analyze(cfg, out)


def analyze(cfg, out):
    out = Path(out)
    saved = load_final(cfg, out)
    raw = pd.read_csv(out / "final_evaluation.csv")
    ensemble = raw[raw.seed == -1]
    summary = (
        ensemble.groupby(["scenario", "policy", "privileged"])
        .agg(
            clicks=("clicks", "mean"),
            utilization=("utilization", "mean"),
            violations=("violations", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(out / "final_summary.csv", index=False)
    primary = saved["primary"]
    rng = np.random.default_rng(81913)
    comparisons = []
    for scenario, g in ensemble.groupby("scenario"):
        p = g[g.policy == primary].set_index(["date", "advertiser"]).clicks
        for name in g.policy.unique():
            if name == primary:
                continue
            delta = p - g[g.policy == name].set_index(["date", "advertiser"]).clicks.reindex(
                p.index
            )
            clusters = delta.groupby(level="date").agg(["sum", "count"])
            ids = rng.integers(len(clusters), size=(5000, len(clusters)))
            draws = clusters["sum"].to_numpy()[ids].sum(1) / clusters["count"].to_numpy()[ids].sum(
                1
            )
            low, high = np.quantile(draws, [0.025, 0.975])
            baseline = g[g.policy == name].clicks.mean()
            comparisons.append(
                dict(
                    scenario=scenario,
                    primary=primary,
                    comparator=name,
                    difference=float(delta.mean()),
                    relative_percent=100 * float(delta.mean()) / max(baseline, 1e-8),
                    ci_low=float(low),
                    ci_high=float(high),
                    date_clusters=len(clusters),
                )
            )
    base.dump(
        out / "final_analysis.json",
        dict(
            primary=primary,
            comparisons=comparisons,
            limitations=f"{len(raw[['date','advertiser']].drop_duplicates())} observed advertiser-days; {raw.date.nunique()} date clusters; conditional logged-winning-inventory replay. Empirical within-hour MDP is approximate; test execution uses exact replay.",
        ),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 9, "savefig.dpi": 180, "axes.spines.top": False, "axes.spines.right": False}
    )
    nominal = summary[summary.scenario == "standard"].sort_values("clicks")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        nominal.policy,
        nominal.clicks,
        color=[
            "#236f8e" if n == primary else "#b08c4e" if n.startswith("privileged") else "#929aa2"
            for n in nominal.policy
        ],
    )
    ax.set_xlabel("Recorded clicks per advertiser-day")
    fig.tight_layout()
    fig.savefig(out / "comparison.png")
    plt.close(fig)
    state = base.load_state(cfg, out)
    names = list(
        dict.fromkeys(
            [
                primary,
                json.loads((out / "analysis.json").read_text())["reference"],
                "pid",
                "quality_pid",
            ]
        )
    )
    scenarios = list(base.SCENARIOS)
    x = np.arange(len(scenarios))
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for j, name in enumerate(names):
        ax.bar(
            x + (j - (len(names) - 1) / 2) * w,
            summary[summary.policy == name].set_index("scenario").loc[scenarios, "clicks"],
            width=w,
            label=name,
        )
    ax.set_xticks(
        x,
        [
            "Nominal",
            "Prices x1.5",
            "Evening x3",
            "Budget .15",
            "Budget .60",
            "Clicks x.5",
            "Coverage .5",
        ],
    )
    ax.set_ylabel("Recorded clicks / advertiser-day")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "stress.png")
    plt.close(fig)
    traces = json.loads((out / "final_traces.json").read_text())
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for name in list(dict.fromkeys([primary, "traffic_linear", "linear", "pid"])):
        curves = [r["trace"] for r in traces if r["policy"] == name and r["seed"] == -1]
        ax.plot(range(25), np.mean(curves, axis=0), label=name)
    ax.set(xlabel="Hour", ylabel="Mean fraction of budget spent", ylim=(0, 1.04))
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "spending.png")
    plt.close(fig)
    print("FINAL NOMINAL RESULTS")
    print(summary[summary.scenario == "standard"].to_string(index=False))


def verify(cfg, out):
    out = Path(out)
    base.verify(cfg, out)
    saved = load_final(cfg, out)
    state = base.load_state(cfg, out)
    cv = pd.read_csv(out / "empirical_cv.csv")
    expected = {
        (fold, int(weight * 100)) for fold in range(len(base.folds_for(cfg))) for weight in WEIGHTS
    }
    assert set(cv[["fold", "episode"]].itertuples(index=False, name=None)) == expected
    assert len(cv) == len(expected) and set(cv.variant) == {"empirical_q"} and set(cv.seed) == {-1}
    picked = base.select_iterations(cv, dict(cfg, variants=["empirical_q"]), state)["empirical_q"]
    assert all(
        np.isclose(saved["empirical_selection"][key], value) for key, value in picked.items()
    )
    weight = picked["episode"] / 100.0
    assert saved["empirical_selection"]["reward_weight"] == weight
    neural = state["selection"]["primary"]
    assert saved["primary"] == (
        "empirical_q"
        if picked["objective"] > state["selection"]["iterations"][neural]["objective"]
        else neural
    )
    # Refit the chosen empirical candidate from each dated prefix. Neither this
    # check nor selection constructs a reserved-day template.
    for fold, (train, validation) in enumerate(base.folds_for(cfg)):
        tasks = [(cfg, train, seed, (weight,)) for seed in cfg["seeds"]]
        with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
            results = dict(pool.map(worker, tasks))
        data = ReplayData(cfg["data_file"], train, validation)
        policy = EmpiricalPolicy(average(results, weight), control_spec(data))
        recorded = cv[(cv.fold == fold) & (cv.episode == picked["episode"])].score.iloc[0]
        assert np.isclose(base.value(policy, base.validation_days(data, cfg)), recorded)
    raw = pd.read_csv(out / "final_evaluation.csv")
    assert (
        hashlib.sha256((out / "final_evaluation.csv").read_bytes()).hexdigest()
        == saved["evaluation_sha256"]
    )
    assert not raw.duplicated(["scenario", "date", "advertiser", "policy", "seed"]).any()
    assert np.isfinite(raw.select_dtypes("number")).all().all()
    assert (raw.spent <= raw.budget + 1e-7).all() and raw.violations.sum() == 0
    assert (raw.clicks <= raw.upper_bound + 1e-7).all()
    data = ReplayData(cfg["data_file"], cfg["development_dates"], (), cfg["test_dates"])
    assert len(raw) == len(pd.read_csv(out / "evaluation.csv")) + len(base.SCENARIOS) * len(
        data.templates["test"]
    ) * (len(cfg["seeds"]) + 1)
    expected = {
        (case, t.date, t.advertiser, "empirical_q", seed)
        for case in base.SCENARIOS
        for t in data.templates["test"]
        for seed in cfg["seeds"] + [-1]
    }
    assert (
        set(
            raw[raw.policy == "empirical_q"][
                ["scenario", "date", "advertiser", "policy", "seed"]
            ].itertuples(index=False, name=None)
        )
        == expected
    )
    for template in data.templates["test"]:
        day = make_replay_day(template, budget_fraction=cfg["evaluation_budget_fraction"])
        for seed, policy in empirical_policies(cfg, out, saved):
            actual = rollout_replay(day, policy)
            row = raw[
                (raw.policy == "empirical_q")
                & (raw.seed == seed)
                & (raw.scenario == "standard")
                & (raw.date == day.date)
                & (raw.advertiser == day.advertiser)
            ].iloc[0]
            assert actual["clicks"] == row.clicks and np.isclose(actual["spent"], row.spent)
    for row in pd.read_csv(out / "final_summary.csv").itertuples():
        values = raw[(raw.scenario == row.scenario) & (raw.policy == row.policy) & (raw.seed == -1)]
        assert np.isclose(values.clicks.mean(), row.clicks)
    print(
        "PASS final stage:",
        len(raw),
        "rows; reconstructed empirical CV; frozen selection; empirical nominal rollouts; source/data/model hashes; zero budget violations.",
    )


def export(cfg, out, path):
    out = Path(out)
    path = Path(path)
    saved = load_final(cfg, out)
    if saved["primary"] != "empirical_q":
        return base.export(cfg, out, path)
    if path.exists():
        raise FileExistsError("Agent exists")
    with np.load(out / "empirical_models.npz", allow_pickle=False) as z:
        arrays = {k: z["ensemble__" + k] for k in ("q", "grid", "actions")}
    arrays["metadata"] = np.asarray(
        json.dumps(
            dict(
                variant="empirical_q",
                spec=saved["spec"],
                data_sha256=saved["frozen"]["base_frozen"]["data_sha256"],
                development_dates=cfg["development_dates"],
                test_dates=cfg["test_dates"],
                seeds=cfg["seeds"],
            )
        )
    )
    base.save_arrays(path, arrays)
    print("Exported validation-selected empirical Q ensemble")


def demo(cfg, out, agent):
    with np.load(agent, allow_pickle=False) as z:
        metadata = json.loads(str(z["metadata"]))
        if metadata["variant"] != "empirical_q":
            return base.demo(cfg, out, agent)
        arrays = {k: z[k] for k in ("q", "grid", "actions")}
    assert (
        hashlib.sha256(Path(cfg["data_file"]).read_bytes()).hexdigest() == metadata["data_sha256"]
    )
    data = ReplayData(cfg["data_file"], metadata["development_dates"], (), metadata["test_dates"])
    policy = EmpiricalPolicy(arrays, metadata["spec"])
    print(
        json.dumps(
            rollout_replay(
                make_replay_day(
                    data.templates["test"][0], budget_fraction=cfg["evaluation_budget_fraction"]
                ),
                policy,
            ),
            indent=2,
        )
    )
