"""Validation-only recalibration of saved sampled Bellman tables.

The public traffic feature uses raw profile inventory, whereas the fitted MDP
fills never-observed launch hours. This experiment adds a traffic estimate whose
denominator matches that MDP. Q-tables and baseline searches are reused by hash.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from bidpace import protocol, tabular, tabular_study as base
from bidpace.replay import ReplayData, rollout_replay

CANDIDATES = [dict(settings, traffic=traffic) for traffic in ("public", "model")
              for settings in base.CANDIDATES]


def inventory_spec(data, original):
    spec = dict(original)
    inventories = []
    for adv in spec["campaigns"]:
        templates = [t for t in data.templates["train"] if t.advertiser == adv]
        fallback = np.mean([len(hr.prices) for t in templates for hr in t.hours if len(hr.prices)])
        inventory = [np.mean([len(t.hours[h].prices) for t in templates]) for h in range(24)]
        inventories.append([float(n if n > 0 else fallback) for n in inventory])
    spec["model_inventory"] = inventories
    return spec


class CalibratedPolicy(tabular.TabularPolicy):
    def __init__(self, tensors, spec, traffic="model", **settings):
        super().__init__(tensors, spec, **settings)
        self.traffic_mode = traffic

    def reset(self, day):
        super().reset(day)
        self.inventory = np.asarray(self.spec["model_inventory"][self.j])
        self.seen = self.expected_seen = 0.0
        self.last_left = self.budget

    def act(self, obs):
        if self.traffic_mode == "model":
            hour = min(int(round(float(obs[1]) * 24)), 23)
            win = float(obs[3])
            paid = float(obs[7]) * 4 * self.profile.price
            if hour > 0 and win > 1e-8 and paid > 1e-8 and float(obs[7]) < 0.999:
                spend = max(0, self.last_left - float(obs[0]) * self.budget)
                self.seen += spend / paid / win
                self.expected_seen += self.inventory[hour - 1]
            prior = 0.05 * self.inventory.sum()
            traffic = np.clip((self.seen + prior) / (self.expected_seen + prior), 0.05, 4)
            obs = np.asarray(obs).copy()
            obs[8] = traffic / 4
        self.last_left = float(obs[0]) * self.budget
        return super().act(obs)


def freeze(source, out):
    source, out = Path(source), Path(out)
    original = base.read(source / "study.json")
    if original["source_sha256"] != protocol.source_hashes():
        raise ValueError("Upstream implementation changed")
    inputs = {}
    for season, cfg in original["configs"].items():
        if base.sha(cfg["data_file"]) != original["data_sha256"][season]:
            raise ValueError("Upstream replay changed")
        folder = source / f"season{season}"
        base.check_selection(folder, base.read(folder / "selection.json"))
        for name in ["selection.json", "baselines.json", "agent.npz", "baseline_market.npz"] + [
            f"fold{f}.{ext}" for f in range(len(protocol.folds_for(cfg))) for ext in ("json", "npz")
        ]:
            inputs[(folder / name).as_posix()] = base.sha(folder / name)
    frozen = dict(configs=original["configs"], source=source.as_posix(), input_hashes=inputs,
                  source_sha256=protocol.source_hashes(), script_sha256=base.sha(__file__),
                  candidates=CANDIDATES,
                  selection="highest mean nominal forward-validation clicks, equal date weights",
                  disclosure="Historical replication. Traffic recalibration motivated by validation "
                             "action traces after the upstream experiment was evaluated. "
                             "Q-tables and baseline search reused unchanged; no test labels select settings.")
    path = out / "study.json"
    if path.exists():
        if base.read(path) != frozen:
            raise ValueError("Changed calibrated experiment; choose a new --out")
    else:
        out.mkdir(parents=True, exist_ok=True)
        protocol.dump(path, frozen)
    return frozen


def select(study, season, root):
    source = Path(study["source"]) / f"season{season}"
    out = Path(root) / f"season{season}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "selection.json").exists():
        check_selection(out)
        return
    cfg = study["configs"][str(season)]
    rows = []
    for fold, (train, val) in enumerate(protocol.folds_for(cfg)):
        data = ReplayData(cfg["data_file"], train, val)
        spec = inventory_spec(data, base.read(source / f"fold{fold}.json")["spec"])
        tensors = base.load_arrays(source / f"fold{fold}.npz")
        for candidate, settings in enumerate(CANDIDATES):
            policy = CalibratedPolicy(tensors, spec, **settings)
            for date, adv, _, day in base.nominal_days(data, cfg):
                result = rollout_replay(day, policy)
                rows.append(dict(candidate=candidate, fold=fold, date=date, advertiser=adv,
                                 clicks=result["clicks"], utilization=result["utilization"]))
    cv = pd.DataFrame(rows)
    cv.to_csv(out / "cv.csv", index=False)
    scores = cv.groupby(["candidate", "fold"]).clicks.mean().groupby("candidate").mean()
    winner = int(scores.idxmax())
    previous = base.read(source / "selection.json")
    data = ReplayData(cfg["data_file"], cfg["development_dates"])
    spec = inventory_spec(data, previous["spec"])
    metadata = dict(variant="calibrated_sampled_q", settings=CANDIDATES[winner], spec=spec,
                    data_sha256=base.sha(cfg["data_file"]), development_dates=cfg["development_dates"],
                    test_dates=cfg["test_dates"], upstream_agent_sha256=base.sha(source / "agent.npz"))
    protocol.save_arrays(out / "agent.npz", dict(base.load_arrays(source / "agent.npz"),
                                                metadata=np.asarray(json.dumps(metadata))))
    saved = dict(candidate=winner, settings=CANDIDATES[winner], spec=spec,
                 reference=previous["reference"], baselines=previous["baselines"],
                 validation_scores={str(k): float(v) for k, v in scores.items()},
                 baseline_scores=previous["baseline_scores"],
                 hashes={name: base.sha(out / name) for name in ("cv.csv", "agent.npz")})
    protocol.dump(out / "selection.json", saved)
    print("FROZEN", season, saved["settings"], "CV", scores[winner],
          "baseline", previous["baseline_scores"][previous["reference"]], flush=True)


def check_selection(out):
    """Validate cached calibration bytes and reconstruct its CV-only choice."""
    out = Path(out)
    saved = base.read(out / "selection.json")
    for name, expected in saved["hashes"].items():
        if base.sha(out / name) != expected:
            raise ValueError(f"Changed calibrated artifact: {name}")
    cv = pd.read_csv(out / "cv.csv")
    scores = cv.groupby(["candidate", "fold"]).clicks.mean().groupby("candidate").mean()
    if int(scores.idxmax()) != saved["candidate"] or saved["settings"] != CANDIDATES[saved["candidate"]]:
        raise ValueError("Changed calibration validation selection")
    return saved

