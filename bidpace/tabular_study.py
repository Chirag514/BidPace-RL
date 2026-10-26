"""Sampled-Q fitting, baseline tuning and validation-only selection helpers.

The multi-RL runner owns evaluation, verification and reporting for all methods.
"""

import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from . import protocol, tabular
from .benchmark import SEASONS, configuration
from .control import fit_control
from .replay import ReplayData, rollout_replay

CANDIDATES = [dict(reward=reward, reserve=reserve, rate=0.5)
              for reward in tabular.REWARDS for reserve in (0.85, 1.0, 1.15)]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files if k != "metadata"}


def freeze(out, template="config.json", data_root="data"):
    out = Path(out)
    cfg = read(template)
    configs = {str(s): configuration(cfg, s, data_root) for s in SEASONS}
    frozen = dict(
        algorithm="sampled finite-horizon Bellman fitted Q iteration",
        configs=configs, candidates=CANDIDATES,
        selection="highest mean nominal forward-validation clicks; first candidate wins ties",
        reference="highest nominal forward-validation deployable baseline",
        data_sha256={s: sha(c["data_file"]) for s, c in configs.items()},
        source_sha256=protocol.source_hashes(),
        budget_grid=tabular.GRID.tolist(), actions=tabular.ACTIONS.tolist(),
        disclosure="Public historical replication after previous experiments, not a blind test. "
                   "No guarantee of gains; all nominal and stress outcomes retained.",
    )
    target = out / "study.json"
    if target.exists():
        if read(target) != frozen:
            raise ValueError("Study source/config/data changed; choose a fresh --out")
    else:
        out.mkdir(parents=True, exist_ok=True)
        protocol.dump(target, frozen)
    return frozen


def nominal_days(data, cfg):
    return protocol.validation_days(data, dict(cfg, validation_conditions=[
        dict(budget_fraction=cfg["evaluation_budget_fraction"])]))


def select(study, season, out):
    cfg = study["configs"][str(season)]
    out = Path(out) / f"season{season}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "selection.json"
    if path.exists():
        saved = read(path)
        check_selection(out, saved)
        return saved
    started = time.monotonic()
    cv = []
    for fold, (train, validation) in enumerate(protocol.folds_for(cfg)):
        cache = out / f"fold{fold}.json"
        if cache.exists():
            previous = read(cache)
            if sha(out / f"fold{fold}.npz") != previous["model_sha256"]:
                raise ValueError("Changed cached fold model")
            cv.extend(previous["rows"])
            continue
        data = ReplayData(cfg["data_file"], train, validation)
        tensors, spec = tabular.fit(data)
        days = nominal_days(data, cfg)
        rows = []
        for candidate, settings in enumerate(CANDIDATES):
            policy = tabular.TabularPolicy(tensors, spec, **settings)
            for date, adv, _, day in days:
                result = rollout_replay(day, policy)
                rows.append(dict(fold=fold, candidate=candidate, date=date, advertiser=adv,
                                 clicks=result["clicks"], utilization=result["utilization"]))
        protocol.save_arrays(out / f"fold{fold}.npz", tensors)
        protocol.dump(cache, dict(rows=rows, spec=spec,
                                 model_sha256=sha(out / f"fold{fold}.npz")))
        cv.extend(rows)
        print("Season", season, "fold", fold, "nominal CV:",
              pd.DataFrame(rows).groupby("candidate").clicks.mean().to_dict(), flush=True)
        del data, tensors
    cv = pd.DataFrame(cv)
    cv.to_csv(out / "cv.csv", index=False)
    # Equal date weights; campaigns on a date remain together.
    scores = cv.groupby(["candidate", "fold"]).clicks.mean().groupby("candidate").mean()
    candidate = int(scores.idxmax())
    baseline_path = out / "baselines.json"
    if baseline_path.exists():
        baseline = read(baseline_path)
    else:
        folds = [ReplayData(cfg["data_file"], train, val) for train, val in protocol.folds_for(cfg)]
        specs, search = protocol.tune_baselines(folds, cfg, nominal_only=True)
        baseline = dict(specs=specs, search=search)
        protocol.dump(baseline_path, baseline)
        del folds
    search = pd.DataFrame(baseline["search"])
    baseline_scores = {}
    for name, spec in baseline["specs"].items():
        if name == "privileged_linear":
            continue
        key = json.dumps(spec, sort_keys=True)
        baseline_scores[name] = float(search[(search.policy == name) & (search.spec == key)].score.mean())
    reference = max(baseline_scores, key=baseline_scores.get)
    data = ReplayData(cfg["data_file"], cfg["development_dates"])
    tensors, spec = tabular.fit(data)
    metadata = dict(variant="sampled_q", spec=spec, settings=CANDIDATES[candidate],
                    development_dates=cfg["development_dates"], test_dates=cfg["test_dates"],
                    data_sha256=study["data_sha256"][str(season)],
                    source_sha256=study["source_sha256"])
    protocol.save_arrays(out / "agent.npz", dict(tensors, metadata=np.asarray(json.dumps(metadata))))
    protocol.save_arrays(out / "baseline_market.npz", fit_control(data))
    saved = dict(candidate=candidate, settings=CANDIDATES[candidate], spec=spec,
                 validation_scores={str(k): float(v) for k, v in scores.items()},
                 reference=reference, baseline_scores=baseline_scores,
                 baselines=baseline["specs"], selection_seconds=time.monotonic() - started,
                 hashes={name: sha(out / name) for name in
                         ("agent.npz", "baseline_market.npz", "cv.csv", "baselines.json")})
    protocol.dump(path, saved)
    print("FROZEN SELECTION season", season, saved["settings"], "CV", scores[candidate],
          "reference", reference, baseline_scores[reference], flush=True)
    return saved


def check_selection(out, saved):
    for name, expected in saved["hashes"].items():
        if sha(Path(out) / name) != expected:
            raise ValueError(f"Changed selected artifact: {name}")
    cv = pd.read_csv(Path(out) / "cv.csv")
    scores = cv.groupby(["candidate", "fold"]).clicks.mean().groupby("candidate").mean()
    if int(scores.idxmax()) != saved["candidate"] or CANDIDATES[saved["candidate"]] != saved["settings"]:
        raise ValueError("Selection is inconsistent with CV")
    baseline = read(Path(out) / "baselines.json")
    if saved["baselines"] != baseline["specs"]:
        raise ValueError("Changed baseline specifications")
    search = pd.DataFrame(baseline["search"])
    scores = {}
    for name, spec in baseline["specs"].items():
        if name != "privileged_linear":
            key = json.dumps(spec, sort_keys=True)
            scores[name] = float(search[(search.policy == name) & (search.spec == key)].score.mean())
    if saved["reference"] != max(scores, key=scores.get):
        raise ValueError("Reference is inconsistent with baseline validation")

