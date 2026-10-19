"""Explain a completed run without fitting or selecting another policy.

Nominal validation uses the already selected checkpoints and baseline parameters.
Inventory shifts are descriptive associations, not evidence of a causal mechanism.
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from itertools import product
import json
from pathlib import Path
import numpy as np
import pandas as pd
from bidpace.protocol import (
    load_state,
    folds_for,
    policy_spec,
    create_policy,
    tensors_at,
    prefix,
    Ensemble,
    Baseline,
    market_tensors,
)
from bidpace.replay import ReplayData, make_replay_day, rollout_replay


def diagnose(cfg, out):
    out = Path(out)
    state = load_state(cfg, out)
    if state["phase"] != "evaluated":
        raise ValueError("Complete training and evaluation before diagnostics")
    with np.load(out / "cv_models.npz", allow_pickle=False) as archive:
        weights = {key: archive[key] for key in archive.files}
    rows = []
    for fold, (train, validation) in enumerate(folds_for(cfg)):
        data = ReplayData(cfg["data_file"], train, validation)
        market = market_tensors(data)
        policies = {name: Baseline(spec, market) for name, spec in state["baselines"].items()}
        for variant in cfg["variants"]:
            episode = state["selection"]["iterations"][variant]["episode"]
            spec = policy_spec(data, variant)
            members = [
                create_policy(tensors_at(weights, prefix(fold, variant, seed, episode)), spec)
                for seed in cfg["seeds"]
            ]
            policies[variant] = Ensemble(members, spec["family"])
        for template in data.templates["validation"]:
            day = make_replay_day(template, budget_fraction=cfg["evaluation_budget_fraction"])
            for name, policy in policies.items():
                rows.append(
                    dict(
                        fold=fold,
                        date=day.date,
                        advertiser=day.advertiser,
                        policy=name,
                        clicks=rollout_replay(day, policy)["clicks"],
                    )
                )
    nominal = pd.DataFrame(rows)
    nominal.to_csv(out / "nominal_cv.csv", index=False)

    data = ReplayData(cfg["data_file"], cfg["development_dates"], (), cfg["test_dates"])
    arrays = data.arrays
    rows = []
    for key, start, end in zip(arrays["keys"], arrays["starts"], arrays["ends"]):
        date, advertiser, _ = map(int, key)
        prices = arrays["prices"][start:end]
        rows.append(
            dict(
                date=date,
                advertiser=advertiser,
                impressions=int(end - start),
                cost=int(prices.sum(dtype=np.uint64)),
            )
        )
    daily = (
        pd.DataFrame(rows)
        .groupby(["date", "advertiser"])[["impressions", "cost"]]
        .sum()
        .reset_index()
    )
    rows = []
    for advertiser, group in daily.groupby("advertiser"):
        train = group[group.date.isin(cfg["development_dates"])]
        test = group[group.date.isin(cfg["test_dates"])]
        if test.empty:
            continue
        train_volume, test_volume = train.impressions.mean(), test.impressions.mean()
        train_price = train.cost.sum() / train.impressions.sum()
        test_price = test.cost.sum() / test.impressions.sum()
        rows.append(
            dict(
                advertiser=int(advertiser),
                development_daily_impressions=train_volume,
                test_daily_impressions=test_volume,
                traffic_ratio=test_volume / train_volume,
                training_min_volume_ratio=train.impressions.min() / train_volume,
                training_max_volume_ratio=train.impressions.max() / train_volume,
                development_mean_price=train_price,
                test_mean_price=test_price,
                price_ratio=test_price / train_price,
                nominal_budget_divided_by_test_daily_cost=cfg["evaluation_budget_fraction"]
                * train.cost.mean()
                / test.cost.mean(),
            )
        )
    distribution = pd.DataFrame(rows)
    distribution.to_csv(out / "distribution.csv", index=False)
    evaluation = pd.read_csv(out / "evaluation.csv")
    if (out / "final_evaluation.csv").exists():
        evaluation = pd.read_csv(out / "final_evaluation.csv")
        candidate_diagnostics(cfg, out, evaluation)
    selected = evaluation[(evaluation.scenario == "standard") & (evaluation.seed == -1)]
    selected.pivot_table(index="advertiser", columns="policy", values="clicks").to_csv(
        out / "per_campaign.csv"
    )
    print("Selected policies on nominal forward-validation days:")
    print(nominal.groupby("policy").clicks.mean().round(4).to_string())
    print("\nObserved inventory shift (explanatory; no new model selection):")
    print(distribution.round(4).to_string(index=False))


def candidate_diagnostics(cfg, out, evaluation):
    """Describe every pre-specified RL candidate; never change model selection.

    Intervals are unadjusted descriptive date-cluster bootstrap intervals. With
    at most six dates, enumerate the bootstrap distribution exactly. Optimization
    seeds are not substituted for independent historical dates.
    """
    from bidpace.protocol import SCENARIOS, dump

    saved = json.loads((out / "final_protocol.json").read_text())
    candidates = cfg["variants"] + ["empirical_q"]
    ensemble = evaluation[evaluation.seed == -1]
    comparisons = []
    for scenario, group in ensemble.groupby("scenario"):
        baseline_names = sorted(set(group[~group.privileged].policy) - set(candidates))
        for name in candidates:
            policy = group[group.policy == name].set_index(["date", "advertiser"]).clicks
            for baseline in baseline_names:
                reference = (
                    group[group.policy == baseline]
                    .set_index(["date", "advertiser"])
                    .clicks.reindex(policy.index)
                )
                delta = policy - reference
                clusters = delta.groupby(level="date").agg(["sum", "count"])
                n = len(clusters)
                if n <= 6:
                    ids = np.asarray(list(product(range(n), repeat=n)))
                    method = "exact empirical date-cluster bootstrap; inverted CDF quantiles"
                else:
                    ids = np.random.default_rng(81913).integers(n, size=(5000, n))
                    method = "5000 date-cluster bootstrap draws"
                draws = clusters["sum"].to_numpy()[ids].sum(1) / clusters["count"].to_numpy()[
                    ids
                ].sum(1)
                low, high = np.quantile(draws, [0.025, 0.975], method="inverted_cdf")
                comparisons.append(
                    dict(
                        scenario=scenario,
                        candidate=name,
                        baseline=baseline,
                        difference=float(delta.mean()),
                        relative_percent=100 * float(delta.mean()) / max(reference.mean(), 1e-8),
                        ci_low=float(low),
                        ci_high=float(high),
                        date_clusters=n,
                        method=method,
                    )
                )
    dump(
        out / "candidate_comparisons.json",
        dict(
            primary=saved["primary"],
            comparisons=comparisons,
            interpretation="Secondary candidate analyses are exploratory. No primary selection is changed. Intervals are unadjusted for multiple comparisons.",
        ),
    )
    evaluation[evaluation.seed != -1].groupby(["scenario", "policy", "seed"]).agg(
        clicks=("clicks", "mean"), utilization=("utilization", "mean")
    ).reset_index().to_csv(out / "final_seed_summary.csv", index=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = pd.read_csv(out / "final_summary.csv")
    reference = json.loads((out / "analysis.json").read_text())["reference"]
    names = list(dict.fromkeys([saved["primary"], "empirical_q", reference, "pid", "unscaled"]))
    scenarios = list(SCENARIOS)
    x = np.arange(len(scenarios))
    width = 0.82 / len(names)
    fig, ax = plt.subplots(figsize=(11, 4))
    for j, name in enumerate(names):
        values = summary[summary.policy == name].set_index("scenario").loc[scenarios, "clicks"]
        label = name + (
            " (secondary)" if name == "empirical_q" and saved["primary"] != name else ""
        )
        ax.bar(x + (j - (len(names) - 1) / 2) * width, values, width=width, label=label)
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
    ax.set_ylabel("Recorded clicks per advertiser-day")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "stress.png", dpi=180)
    plt.close(fig)
    traces = json.loads((out / "final_traces.json").read_text())
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for name in list(dict.fromkeys([saved["primary"], "empirical_q", "linear", "pid"])):
        curves = [r["trace"] for r in traces if r["policy"] == name and r["seed"] == -1]
        ax.plot(range(25), np.mean(curves, axis=0), label=name)
    ax.set(xlabel="Hour", ylabel="Mean fraction of budget spent", ylim=(0, 1.04))
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "spending.png", dpi=180)
    plt.close(fig)
    nominal = summary[summary.scenario == "standard"].sort_values("clicks")
    labels = [
        name
        + (
            " (primary)"
            if name == saved["primary"]
            else " (secondary)" if name == "empirical_q" else ""
        )
        for name in nominal.policy
    ]
    colors = [
        (
            "#236f8e"
            if name == saved["primary"]
            else (
                "#288b76"
                if name == "empirical_q"
                else "#b08c4e" if name.startswith("privileged") else "#929aa2"
            )
        )
        for name in nominal.policy
    ]
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    ax.barh(labels, nominal.clicks, color=colors)
    ax.set_xlabel("Recorded clicks per advertiser-day")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=180)
    plt.close(fig)


def export_empirical(cfg, out, path):
    """Export the empirical candidate while preserving validation selection."""
    from bidpace.model_based import load_final
    from bidpace.protocol import save_arrays

    out, path = Path(out), Path(path)
    saved = load_final(cfg, out)
    if path.exists():
        raise FileExistsError("Choose a new candidate agent path")
    with np.load(out / "empirical_models.npz", allow_pickle=False) as archive:
        arrays = {key: archive["ensemble__" + key] for key in ("q", "grid", "actions")}
    arrays["metadata"] = np.asarray(
        json.dumps(
            dict(
                variant="empirical_q",
                spec=saved["spec"],
                data_sha256=saved["frozen"]["base_frozen"]["data_sha256"],
                development_dates=cfg["development_dates"],
                test_dates=cfg["test_dates"],
                seeds=cfg["seeds"],
                budget_fraction=cfg["evaluation_budget_fraction"],
                status="Candidate export; validation-selected primary is " + saved["primary"],
            )
        )
    )
    save_arrays(path, arrays)
    print("Exported empirical candidate; primary selection unchanged.")
