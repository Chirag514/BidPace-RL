"""Compare three RL candidates against tuned baselines across iPinYou seasons.

Pipeline: optional source training -> freeze -> nominal CV selection -> exported
agent evaluation -> replay verification -> combined reports. Source/model hashes
make each run immutable. The public load_agent() helper also supports inference
from the nine standalone release agents without a training-cache directory.

Imported neural checkpoints retain their original forward-validation stopping
rule; this runner selects among the resulting methods on nominal CV only.
"""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from bidpace import protocol, tabular_study as base
from bidpace.benchmark import configuration, SEASONS
from bidpace.replay import ReplayData, make_replay_day, rollout_replay
from bidpace.control import fit_control
from experiments import calibrated_tabular as calibrated

# This order also breaks equal-score RL ties; it is shared by every season.
RL = ("traffic_dqn", "campaign_dqn", "sampled_q")
# Plot highlights only. Reporting and superiority checks retain all baselines.
CORE_BASELINES = ("constant", "linear", "pid", "mpc")


def replay_input_hash(path):
    """Hash array content, independent of NPZ compression and ZIP timestamps."""
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as archive:
        for name in sorted(archive.files):
            array = np.ascontiguousarray(archive[name])
            header = json.dumps([name, array.dtype.str, list(array.shape)])
            digest.update(header.encode("utf8") + b"\0")
            if array.nbytes:
                digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def train_missing(args):
    """Fresh CPU training is opt-in; completed sources are reused unchanged."""
    template = base.read(args.template)
    template["variants"] = list(RL[:2])
    for season in SEASONS:
        target = Path(args.neural_source) / f"season{season}"
        if not (target / "models.npz").exists():
            cfg = configuration(template, season, args.data_root)
            protocol.train(cfg, target, resume=(target / "protocol.json").exists())
    study = base.freeze(args.table_source, args.template, args.data_root)
    for season in SEASONS:
        base.select(study, season, args.table_source)
    study = calibrated.freeze(args.table_source, args.calibrated_source)
    for season in SEASONS:
        calibrated.select(study, season, args.calibrated_source)


def freeze(neural_source, table_source, calibrated_source, out):
    """Validate compatible source artifacts and persist this comparison's contract.

    Hashing a historical evaluation file records its identity; selection never
    reads its outcomes. Those outcomes are consumed only by replay verification.
    """
    original = base.read(Path(table_source) / "study.json")
    calibration = base.read(Path(calibrated_source) / "study.json")
    # Import completed artifacts under their original provenance. Resuming training
    # remains source-strict; importing additionally reconstructs archived CV below.
    if calibration["source_sha256"] != original["source_sha256"]:
        raise ValueError("Table/calibration source provenance differs")
    if calibration["configs"] != original["configs"]:
        raise ValueError("Table/calibration configurations differ")
    inputs = {str(Path(folder) / "study.json"): base.sha(Path(folder) / "study.json")
              for folder in (table_source, calibrated_source)}
    for name, expected in calibration["input_hashes"].items():
        relative = Path(name.replace("\\", "/")).relative_to(
            Path(calibration["source"].replace("\\", "/")))
        path = Path(table_source) / relative
        if base.sha(path) != expected:
            raise ValueError(f"Changed calibration input: {relative}")
        inputs[path.as_posix()] = expected
    neural_protocols = {}
    for season, cfg in original["configs"].items():
        neural = Path(neural_source) / f"season{season}"
        table = Path(table_source) / f"season{season}"
        cal = Path(calibrated_source) / f"season{season}"
        state = base.read(neural / "protocol.json")
        previous = state["frozen"]["config"]
        for key in ("development_dates", "cv_validation_dates", "test_dates", "evaluation_budget_fraction"):
            if previous[key] != cfg[key]:
                raise ValueError(f"Neural/table {key} differs")
        digest = base.sha(cfg["data_file"])
        if digest != original["data_sha256"][season] or digest != state["frozen"]["data_sha256"]:
            raise ValueError("Neural/table replay inputs differ")
        if not set(RL[:2]).issubset(state["policy_specs"]):
            raise ValueError("Both DQN variants must be trained for every season")
        for name, key in (("models.npz", "model_sha256"), ("cv_models.npz", "cv_model_sha256")):
            if key not in state:
                # Older trainers omitted hashes until evaluation. Only matching
                # source training may supply an unhashed, completed archive.
                if state["phase"] != "trained" or state["frozen"]["source_sha256"] != protocol.source_hashes():
                    raise ValueError("Unverified legacy training artifacts")
            elif base.sha(neural / name) != state[key]:
                raise ValueError(f"Changed archived neural artifact: {name}")
        base.check_selection(table, base.read(table / "selection.json"))
        selected = calibrated.check_selection(cal)
        if selected["baselines"] != base.read(table / "selection.json")["baselines"]:
            raise ValueError("Changed calibrated baseline specifications")
        for fold in range(len(protocol.folds_for(cfg))):
            if base.sha(table / f"fold{fold}.npz") != base.read(table / f"fold{fold}.json")["model_sha256"]:
                raise ValueError("Changed table fold model")
        paths = [neural / n for n in ("protocol.json", "models.npz", "cv_models.npz")]
        # Archived outcomes across all scenarios are verification references only.
        if (neural / "evaluation.csv").exists():
            paths.append(neural / "evaluation.csv")
        paths += [cal / n for n in ("agent.npz", "selection.json", "cv.csv")]
        paths += [table / n for n in ("selection.json", "baselines.json", "baseline_market.npz")]
        paths += [table / f"fold{f}.{ext}" for f in range(len(protocol.folds_for(cfg)))
                  for ext in ("json", "npz")]
        inputs.update({p.as_posix(): base.sha(p) for p in paths})
        neural_protocols[season] = dict(
            selection=state["selection"], original_source_sha256=state["frozen"]["source_sha256"],
            current_source_matches=state["frozen"]["source_sha256"] == protocol.source_hashes())
    # Comparison metadata carries only settings used by this candidate set.
    # Original training protocols remain intact and are identified by input hashes.
    configs = {season: {key: value for key, value in cfg.items()
                        if key not in ("rainbow", "ppo", "variants")}
               for season, cfg in original["configs"].items()}
    for cfg in configs.values():
        cfg["variants"] = list(RL[:2])
    study = dict(configs=configs, neural_source=str(neural_source),
                 table_source=str(table_source), calibrated_source=str(calibrated_source),
                 rl_candidates=list(RL), core_baselines=list(CORE_BASELINES),
                 source_sha256=protocol.source_hashes(), script_sha256=base.sha(__file__),
                 calibration_sha256=base.sha(calibrated.__file__), input_hashes=inputs,
                  neural_provenance=neural_protocols,
                  table_provenance=dict(original_source_sha256=original["source_sha256"],
                      current_source_matches=original["source_sha256"] == protocol.source_hashes()),
                  calibration_provenance=dict(original_script_sha256=calibration["script_sha256"],
                      current_script_matches=calibration["script_sha256"] == base.sha(calibrated.__file__)),
                 seasonal_selection="highest mean nominal forward-validation clicks, equal date weights",
                 shared_selection="highest mean across seasons of nominal CV / validation reference CV",
                  disclosure="Retrospective comparison on public data, not a blind prospective test. Imported neural "
                            "checkpoints retain their original multi-condition forward-validation "
                            "stopping rule; sampled-Q settings use nominal CV. Current algorithm "
                            "selection replays nominal CV for all methods. No test-based selection "
                            "or guaranteed RL superiority. Separate agents train within each season.")
    path = Path(out) / "study.json"
    if path.exists():
        if base.read(path) != study:
            raise ValueError("Study inputs/code changed; choose a fresh --out")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        protocol.dump(path, study)
    return study


def neural_policy(state, arrays, variant, fold=None):
    """Reconstruct a seed-averaged DQN at its saved checkpoint and campaign map."""
    checkpoint = state["selection"]["iterations"][variant]["episode"]
    spec = dict(state["policy_specs"][variant])
    if fold is not None:
        # A campaign can first appear after a fold's training cutoff.
        train, _ = protocol.folds_for(state["frozen"]["config"])[fold]
        data = ReplayData(state["frozen"]["config"]["data_file"], train)
        spec = protocol.policy_spec(data, variant)
    policies = []
    for seed in state["frozen"]["config"]["seeds"]:
        key = f"{variant}__s{seed}__" if fold is None else protocol.prefix(fold, variant, seed, checkpoint)
        policies.append(protocol.create_policy(protocol.tensors_at(arrays, key), spec))
    return protocol.Ensemble(policies, "dqn")


def validation_records(study, season):
    """Replay chosen source candidates/baselines on earlier-to-later CV folds.

    Each fold refits profile/control statistics from its training prefix. Reserved
    test dates are not passed to ReplayData here. Rows retain dates and campaigns
    so both selection and later verification can be reconstructed.
    """
    cfg = study["configs"][str(season)]
    neural = Path(study["neural_source"]) / f"season{season}"
    table = Path(study["table_source"]) / f"season{season}"
    cal = base.read(Path(study["calibrated_source"]) / f"season{season}" / "selection.json")
    state = base.read(neural / "protocol.json")
    # Use the current declared path after verifying the replay hash.
    state["frozen"]["config"]["data_file"] = cfg["data_file"]
    arrays = base.load_arrays(neural / "cv_models.npz")
    rows = []
    for fold, (train, val) in enumerate(protocol.folds_for(cfg)):
        data = ReplayData(cfg["data_file"], train, val)
        spec = calibrated.inventory_spec(data, base.read(table / f"fold{fold}.json")["spec"])
        agents = [(name, neural_policy(state, arrays, name, fold)) for name in RL[:2]]
        agents.append(("sampled_q", calibrated.CalibratedPolicy(
            base.load_arrays(table / f"fold{fold}.npz"), spec, **cal["settings"])))
        market = fit_control(data)
        agents += [(name, protocol.Baseline(spec, market)) for name, spec in cal["baselines"].items()]
        for date, adv, _, day in base.nominal_days(data, cfg):
            for name, policy in agents:
                result = rollout_replay(day, policy)
                rows.append(dict(fold=fold, date=date, advertiser=adv, policy=name,
                                 clicks=result["clicks"], utilization=result["utilization"]))
    frame = pd.DataFrame(rows)
    # A historical source fingerprint is not rewritten to claim new training.
    # Its selected sampled-Q policy must still reproduce its archived CV rows.
    archived = pd.read_csv(Path(study["calibrated_source"]) / f"season{season}" / "cv.csv")
    columns = ["fold", "date", "advertiser", "clicks", "utilization"]
    keys = columns[:3]
    expected = archived[archived.candidate == cal["candidate"]][columns].sort_values(keys).reset_index(drop=True)
    actual = frame[frame.policy == "sampled_q"][columns].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    return frame


def choices(cv):
    """Return equal-date mean CV scores, the RL winner and a deployable reference.

    Multiple advertisers on a date do not give that date extra selection weight.
    Pandas' sorted policy groups break baseline ties alphabetically.
    """
    scores = cv.groupby(["policy", "fold"]).clicks.mean().groupby("policy").mean()
    reference = scores.drop(list(RL) + ["privileged_linear"], errors="ignore").idxmax()
    primary = max(RL, key=lambda name: scores[name])
    return scores, str(primary), str(reference)


def select(study, season, root):
    """Freeze seasonal choices and export every RL, including non-winning ones."""
    out = Path(root) / f"season{season}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "selection.json").exists():
        check_selected(out)
        return
    cv = validation_records(study, season)
    cv.to_csv(out / "validation.csv", index=False)
    scores, primary, reference = choices(cv)
    cfg = study["configs"][str(season)]
    neural = Path(study["neural_source"]) / f"season{season}"
    state = base.read(neural / "protocol.json")
    arrays = base.load_arrays(neural / "models.npz")
    for variant in RL[:2]:
        metadata = dict(variant=variant, seeds=state["frozen"]["config"]["seeds"],
                        spec=state["policy_specs"][variant],
                        data_sha256=state["frozen"]["data_sha256"],
                        development_dates=cfg["development_dates"], test_dates=cfg["test_dates"],
                        checkpoint=state["selection"]["iterations"][variant]["episode"])
        weights = {k: v for k, v in arrays.items() if k.startswith(variant + "__")}
        protocol.save_arrays(out / f"{variant}.npz", dict(weights, metadata=np.asarray(json.dumps(metadata))))
    cal_path = Path(study["calibrated_source"]) / f"season{season}" / "agent.npz"
    with np.load(cal_path, allow_pickle=False) as z:
        protocol.save_arrays(out / "sampled_q.npz", {k: z[k] for k in z.files})
    saved = dict(primary=primary, reference=reference, validation_scores=scores.to_dict(),
                 baselines=base.read(Path(study["calibrated_source"]) / f"season{season}" / "selection.json")["baselines"],
                 data_sha256=state["frozen"]["data_sha256"],
                 replay_arrays_sha256=replay_input_hash(cfg["data_file"]),
                 hashes={name: base.sha(out / name) for name in
                         ["validation.csv"] + [f"{name}.npz" for name in RL]})
    protocol.dump(out / "selection.json", saved)
    print("SELECTED season", season, primary, {n: round(float(scores[n]), 3) for n in RL},
          "reference", reference, round(float(scores[reference]), 3), flush=True)


def check_selected(out):
    """Check exported bytes and reconstruct the choices from stored CV rows."""
    saved = base.read(Path(out) / "selection.json")
    for name, digest in saved["hashes"].items():
        if base.sha(Path(out) / name) != digest:
            raise ValueError("Changed selected model or validation data")
    scores, primary, reference = choices(pd.read_csv(Path(out) / "validation.csv"))
    if primary != saved["primary"] or reference != saved["reference"]:
        raise ValueError("Selection differs from validation")
    if any(not np.isclose(v, saved["validation_scores"][k]) for k, v in scores.items()):
        raise ValueError("Changed validation scores")
    return saved


def shared_selection(study, root):
    """Select one method across seasons using scale-normalized validation scores.

    Each season contributes one ratio to its reference's CV score. Persisting the
    seasonal selection hashes also prevents changing a season after this choice.
    """
    scores = {name: [] for name in RL}
    hashes = {}
    for season in study["configs"]:
        out = Path(root) / f"season{season}"
        saved = check_selected(out)
        denominator = saved["validation_scores"][saved["reference"]]
        if denominator <= 0:
            raise ValueError("Shared selection needs a positive reference score")
        for name in RL:
            scores[name].append(saved["validation_scores"][name] / denominator)
        hashes[season] = base.sha(out / "selection.json")
    means = {name: float(np.mean(values)) for name, values in scores.items()}
    record = dict(primary=max(RL, key=means.get), normalized_validation_scores=means,
                  selection_sha256=hashes,
                  interpretation="One common algorithm family; separate within-season trained agents. "
                                 "Cross-season validation selection is a historical study, not an "
                                 "earliest-season chronological deployment claim.")
    path = Path(root) / "shared_selection.json"
    if path.exists() and base.read(path) != record:
        raise ValueError("Changed shared validation selection")
    protocol.dump(path, record)
    return record


def load_agent(path):
    """Load a standalone NPZ as a reset(day)/act(obs) policy for ReplayEnv.

    DQN files contain all seed members; sampled-Q files contain tables plus the
    selected calibration. Metadata carries training-fitted campaign information.
    """
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"]))
        arrays = {k: z[k] for k in z.files if k != "metadata"}
    if meta["variant"] == "calibrated_sampled_q":
        return calibrated.CalibratedPolicy(arrays, meta["spec"], **meta["settings"])
    if meta["variant"] not in RL[:2]:
        raise ValueError("Unsupported exported agent")
    return protocol.Ensemble([
        protocol.create_policy(protocol.tensors_at(arrays, f"{meta['variant']}__s{seed}__"), meta["spec"])
        for seed in meta["seeds"]], "dqn")


def evaluation_records(study, season, root):
    """Run exported agents and frozen baselines with paired scenario randomness."""
    cfg = study["configs"][str(season)]
    out = Path(root) / f"season{season}"
    saved = check_selected(out)
    market = base.load_arrays(Path(study["table_source"]) / f"season{season}" / "baseline_market.npz")
    agents = [(name, load_agent(out / f"{name}.npz")) for name in RL]
    agents += [(name, protocol.Baseline(s, market)) for name, s in saved["baselines"].items()]
    data = ReplayData(cfg["data_file"], cfg["development_dates"], (), cfg["test_dates"])
    return scenario_records(data, cfg, season, agents)


def scenario_records(data, cfg, season, agents):
    """Evaluate fixed policies with the same scenario transformations and seeds."""
    rows = []
    for case, changes in protocol.SCENARIOS.items():
        for i, template in enumerate(data.templates["test"]):
            settings = dict(budget_fraction=cfg["evaluation_budget_fraction"])
            settings.update(changes)
            day = make_replay_day(template, seed=3000000 + i * 10000, **settings)
            for name, policy in agents:
                result = rollout_replay(day, policy)
                result.pop("trace")
                if result["violations"] or result["spent"] > day.budget + 1e-7:
                    raise ValueError("Hard budget violated")
                rows.append(dict(season=int(season), scenario=case, date=day.date,
                                 advertiser=day.advertiser, policy=name, **result))
    return pd.DataFrame(rows)


def replay_published(data_root="data", evidence="evidence", agents="agents/comparison"):
    """Reproduce shipped outcomes without training caches or creating new outputs."""
    import build_release

    evidence, agents = Path(evidence), Path(agents)
    manifest = base.read(agents / "manifest.json")
    # Published manifests use project-relative paths. Resolve each entry relative
    # to its declared asset directory, so callers may relocate a complete release.
    expected_paths = {path.as_posix() for path in build_release.AGENTS + build_release.EVIDENCE}
    normalized = {name.replace("\\", "/"): digest for name, digest in manifest.items()}
    if set(normalized) != expected_paths:
        raise ValueError("Incomplete published asset manifest")
    for name, digest in normalized.items():
        path = (evidence if name.startswith("evidence/") else agents) / Path(name).name
        if base.sha(path) != digest:
            raise ValueError(f"Changed published artifact: {path}")
    study = base.read(evidence / "study.json")
    selected = base.read(evidence / "selection.json")
    validation = pd.read_csv(evidence / "validation.csv")
    recorded = pd.read_csv(evidence / "evaluation.csv")
    frames = []
    for season, original in study["configs"].items():
        cfg = dict(original, data_file=(Path(data_root) / f"season{season}" / "replay.npz").as_posix())
        path = Path(cfg["data_file"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing replay input: {path}. Run python -m bidpace.benchmark prepare.")
        saved = selected[season]
        if base.sha(path) != saved["data_sha256"]:
            if replay_input_hash(path) != saved["replay_arrays_sha256"]:
                raise ValueError(f"Published replay data differs for season {season}")
        cv = validation[validation.season == int(season)]
        scores, primary, reference = choices(cv)
        if primary != saved["primary"] or reference != saved["reference"]:
            raise ValueError("Published selection differs from validation")
        if set(scores.index) != set(saved["validation_scores"]) or any(
            not np.isclose(value, saved["validation_scores"][name]) for name, value in scores.items()
        ):
            raise ValueError("Published validation scores changed")
        data = ReplayData(path, cfg["development_dates"], (), cfg["test_dates"])
        policies = [(name, load_agent(agents / f"season{season}_{name}.npz")) for name in RL]
        market = fit_control(data)
        policies += [(name, protocol.Baseline(spec, market)) for name, spec in saved["baselines"].items()]
        frame = scenario_records(data, cfg, season, policies)
        keys = ["season", "scenario", "date", "advertiser", "policy"]
        expected = recorded[recorded.season == int(season)]
        pd.testing.assert_frame_equal(
            frame.sort_values(keys).reset_index(drop=True),
            expected.sort_values(keys).reset_index(drop=True), check_dtype=False,
        )
        frames.append(frame)
        print(f"PASS published season {season}: all policies, all scenarios, selection and hard budgets", flush=True)
    result = pd.concat(frames, ignore_index=True)
    print(f"PASS published replay: {len(result):,} rows; no training caches required", flush=True)
    return result


def evaluate(study, season, root, verify=False):
    """Write held-out rows, or reconstruct and compare a completed evaluation.

    Verification additionally reconstructs CV and, when present, compares neural
    outcomes with historical archives. It never refits or reselects a candidate.
    """
    shared_selection(study, root)  # all planned seasonal selections precede evaluation
    out = Path(root) / f"season{season}"
    if not verify and (out / "evaluation.json").exists():
        return
    frame = evaluation_records(study, season, root)
    if verify:
        audit = base.read(out / "evaluation.json")
        for name, digest in audit["hashes"].items():
            if base.sha(out / name) != digest:
                raise ValueError("Changed evaluation artifact")
        pd.testing.assert_frame_equal(frame, pd.read_csv(out / "evaluation.csv"), check_dtype=False)
        pd.testing.assert_frame_equal(validation_records(study, season),
                                      pd.read_csv(out / "validation.csv"), check_dtype=False)
        archive = Path(study["neural_source"]) / f"season{season}" / "evaluation.csv"
        if archive.exists():
            old = pd.read_csv(archive)
            keys = ["scenario", "date", "advertiser", "policy"]
            old = old[(old.seed == -1) & old.policy.isin(RL[:2])].set_index(keys).sort_index()
            current = frame[frame.policy.isin(RL[:2])].set_index(keys).sort_index()
            pd.testing.assert_frame_equal(current[["clicks", "spent"]], old[["clicks", "spent"]],
                                          check_dtype=False)
        checks = "CV, all scenarios, exported agents"
        if archive.exists():
            checks += ", archived DQN outcomes"
        print(f"PASS season {season}: {checks}", flush=True)
        return
    frame.to_csv(out / "evaluation.csv", index=False)
    protocol.dump(out / "evaluation.json", dict(hashes={
        n: base.sha(out / n) for n in ("evaluation.csv", "selection.json")}))


def summarize(study, root):
    """Combine audited rows into aggregates, paired intervals and explicit claims.

    Highest test baselines are descriptive comparators only. Resampling whole
    dates retains within-date campaign dependence; stress cases reuse those dates.
    """
    root = Path(root)
    shared = shared_selection(study, root)
    frames = []
    for season in study["configs"]:
        out = root / f"season{season}"
        for name, digest in base.read(out / "evaluation.json")["hashes"].items():
            if base.sha(out / name) != digest:
                raise ValueError("Changed results")
        frames.append(pd.read_csv(out / "evaluation.csv"))
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(root / "evaluation.csv", index=False)
    summary = frame.groupby(["season", "scenario", "policy"]).agg(
        clicks=("clicks", "mean"), total_clicks=("clicks", "sum"),
        utilization=("utilization", "mean"), violations=("violations", "sum")).reset_index()
    summary.to_csv(root / "summary.csv", index=False)
    comparisons, decisions = [], []
    rng = np.random.default_rng(7102026)
    for (season, scenario), group in frame.groupby(["season", "scenario"]):
        saved = check_selected(root / f"season{season}")
        baseline_names = [n for n in group.policy.unique() if n not in RL and n != "privileged_linear"]
        means = group.groupby("policy").clicks.mean()
        strongest = max(baseline_names, key=lambda n: means[n])
        for name in RL:
            a = group[group.policy == name].set_index(["date", "advertiser"]).clicks
            for baseline in baseline_names:
                b = group[group.policy == baseline].set_index(["date", "advertiser"]).clicks
                if set(a.index) != set(b.index):
                    raise ValueError("Unpaired evaluation")
                delta = a - b
                clusters = delta.groupby(level="date").agg(["sum", "count"])
                ids = rng.integers(len(clusters), size=(5000, len(clusters)))
                draws = clusters["sum"].to_numpy()[ids].sum(1) / clusters["count"].to_numpy()[ids].sum(1)
                low, high = np.quantile(draws, [0.025, 0.975])
                comparisons.append(dict(season=season, scenario=scenario, rl=name, baseline=baseline,
                                        difference=float(delta.mean()), ci_low=low, ci_high=high,
                                        date_clusters=len(clusters), validation_reference=baseline == saved["reference"]))
        if scenario == "standard":
            decisions.append(dict(season=int(season), selected_rl=saved["primary"],
                                  selected_clicks=float(means[saved["primary"]]), reference=saved["reference"],
                                  reference_clicks=float(means[saved["reference"]]),
                                  highest_observed_baseline=strongest, highest_baseline_clicks=float(means[strongest]),
                                  shared_rl=shared["primary"], shared_clicks=float(means[shared["primary"]])))
    pd.DataFrame(comparisons).to_csv(root / "comparisons.csv", index=False)
    nominal = summary[summary.scenario == "standard"].pivot(index="season", columns="policy", values="clicks")
    nominal.to_csv(root / "nominal.csv")
    flags = dict(
        seasonal_selected_beats_reference_all_seasons=all(d["selected_clicks"] > d["reference_clicks"] for d in decisions),
        seasonal_selected_beats_every_baseline_all_seasons=all(d["selected_clicks"] > d["highest_baseline_clicks"] for d in decisions),
        shared_algorithm_beats_every_baseline_all_seasons=all(d["shared_clicks"] > d["highest_baseline_clicks"] for d in decisions),
        every_rl_beats_every_baseline_all_seasons=all(
            nominal.loc[d["season"], list(RL)].min() > d["highest_baseline_clicks"] for d in decisions))
    protocol.dump(root / "result.json", dict(decisions=decisions, claims=flags,
                                            disclosure=study["disclosure"],
                                            uncertainty="Paired date-cluster bootstrap; few dates; descriptive, unadjusted."))
    plot(summary, root / "nominal.png")
    print(nominal[list(RL) + list(CORE_BASELINES)].round(3).to_string(), flush=True)
    print(f"Shared validation choice: {shared['primary']}. Reports: {root.as_posix()}", flush=True)


def plot(summary, path):
    """Write one nominal figure with the same candidate order in every season."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seasons = sorted(summary.season.unique())
    fig, axes = plt.subplots(1, len(seasons), figsize=(5 * len(seasons), 4), squeeze=False)
    names = list(RL) + list(CORE_BASELINES)
    for ax, season in zip(axes[0], seasons):
        values = summary[(summary.season == season) & (summary.scenario == "standard")].set_index("policy")
        ax.barh(names, values.loc[names, "clicks"], color=["#236f8e"] * len(RL) + ["#929aa2"] * len(CORE_BASELINES))
        ax.set(title=f"Season {season}", xlabel="Recorded clicks / advertiser-day")
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    """Dispatch stages; finish all planned selections before evaluating any season."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["replay", "select", "evaluate", "verify", "summarize", "all"])
    parser.add_argument("--neural-source", default="artifacts/neural")
    parser.add_argument("--table-source", default="artifacts/tabular")
    parser.add_argument("--calibrated-source", default="artifacts/calibration")
    parser.add_argument("--out", default="results")
    parser.add_argument("--train-missing", action="store_true", help="Train missing sources; can take hours on CPU")
    parser.add_argument("--template", default="config.json")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--evidence", default="evidence", help="Published evidence for replay")
    parser.add_argument("--agents", default="agents/comparison", help="Published policies for replay")
    args = parser.parse_args()
    if args.command == "replay":
        if args.train_missing:
            parser.error("replay uses shipped policies; --train-missing is not applicable")
        try:
            replay_published(args.data_root, args.evidence, args.agents)
        except (OSError, ValueError, AssertionError) as error:
            parser.exit(2, f"Replay failed: {error}\n")
        return
    if not args.train_missing and not all(
        (Path(folder) / "study.json").is_file()
        for folder in (args.table_source, args.calibrated_source)
    ):
        parser.error("Training caches are missing. Use 'replay' for the shipped policies, "
                     "or 'all --train-missing' to train from scratch.")
    if args.train_missing:
        train_missing(args)
    study = freeze(args.neural_source, args.table_source, args.calibrated_source, args.out)
    if args.command in ("select", "all"):
        for season in study["configs"]:
            select(study, season, args.out)
        shared_selection(study, args.out)
    if args.command in ("evaluate", "all"):
        for season in study["configs"]:
            evaluate(study, season, args.out)
    if args.command in ("verify", "all"):
        for season in study["configs"]:
            evaluate(study, season, args.out, verify=True)
    if args.command in ("summarize", "all"):
        summarize(study, args.out)


if __name__ == "__main__":
    main()
