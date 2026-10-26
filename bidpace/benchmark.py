"""Seasonal date definitions and chronological iPinYou data preparation."""

import os

# Small NumPy networks use process workers rather than nested BLAS threads.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
from pathlib import Path
from . import fetch, protocol
from .replay import prepare, ReplayData

SEASONS = {
    1: dict(development=list(range(20130311, 20130318)),
            validation=[20130315, 20130316, 20130317],
            test=[20130318, 20130319, 20130320]),
    2: dict(development=list(range(20130606, 20130613)),
            validation=[20130610, 20130611, 20130612],
            test=[20130613, 20130614, 20130615]),
    3: dict(development=list(range(20131019, 20131025)),
            validation=[20131022, 20131023, 20131024],
            test=[20131025, 20131026, 20131027, 20131028]),
}


def configuration(template, season, data_root):
    """Copy the training template and substitute one season's paths and dates."""
    cfg = json.loads(json.dumps(template))
    dates = SEASONS[season]
    cfg.update(
        data_file=(Path(data_root) / f"season{season}" / "replay.npz").as_posix(),
        development_dates=dates["development"],
        cv_validation_dates=dates["validation"],
        test_dates=dates["test"],
        data_scope=f"Public Season {season} chronological replication; all available priced "
                   "impressions/clicks; public leaderboard admitted only after development. "
                   "Prior project results informed this study; not an untouched blind test.",
    )
    return cfg


def prepare_season(cfg, season, data_root):
    target = Path(cfg["data_file"])
    if target.exists():
        # Accept a cache only if the entire requested timeline is present.
        data = ReplayData(target, cfg["development_dates"], (), cfg["test_dates"])
        print("Reusing prepared input:", target, data.audit()["coverage"], flush=True)
        return
    raw = Path(data_root) / f"raw{season}"
    fetch.fetch(raw, cfg["workers"], season, scope="priced", include_test=True)
    prepare(raw, target.parent, cfg["workers"], leaderboard_after=max(cfg["development_dates"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare"])
    parser.add_argument("--out", default="prepared_inputs", help="Preparation configuration directory")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--template", default="config.json")
    parser.add_argument("--season", type=int, choices=list(SEASONS))
    args = parser.parse_args()
    try:
        template = json.loads(Path(args.template).read_text())
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for season in [args.season] if args.season else SEASONS:
            cfg = configuration(template, season, args.data_root)
            prepare_season(cfg, season, args.data_root)
            protocol.dump(out / f"config_season{season}.json", cfg)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Preparation stopped: {error}\n")


if __name__ == "__main__":
    main()
