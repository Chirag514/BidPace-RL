"""Run the complete BidPace-RL experiment from the project directory."""

import os

# Small NumPy networks benefit from separate process workers, not BLAS threads.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
from pathlib import Path
from bidpace import model_based, protocol, reporting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["all", "train", "evaluate", "analyze", "verify", "export", "demo"]
    )
    parser.add_argument("--config", default="config.json", help="experiment configuration")
    parser.add_argument("--out", default="results", help="generated experiment directory")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed folds with matching code, input and configuration",
    )
    parser.add_argument("--agent", help="new export path, or existing standalone agent for demo")
    parser.add_argument(
        "--variant",
        choices=["primary", "empirical_q"],
        default="primary",
        help="agent to export; primary follows validation selection",
    )
    args = parser.parse_args()
    if args.command == "demo" and not args.agent:
        parser.error("demo requires --agent pointing to an exported NPZ file")
    try:
        cfg = json.loads(Path(args.config).read_text())
        out = Path(args.out)
        if args.command in ("all", "train"):
            protocol.train(cfg, out, args.resume)
            model_based.select(cfg, out)
        if args.command in ("all", "evaluate"):
            model_based.evaluate(cfg, out)
            reporting.diagnose(cfg, out)
        if args.command == "analyze":
            protocol.analyze(cfg, out)
            model_based.analyze(cfg, out)
            reporting.diagnose(cfg, out)
        if args.command in ("all", "verify"):
            model_based.verify(cfg, out)
        if args.command == "export":
            target = Path(args.agent) if args.agent else out / "agent.npz"
            if args.variant == "empirical_q":
                reporting.export_empirical(cfg, out, target)
            else:
                model_based.export(cfg, out, target)
        if args.command == "demo":
            model_based.demo(cfg, out, args.agent)
    except (FileNotFoundError, FileExistsError) as exc:
        # Missing input and existing-output errors should explain the remedy.
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
