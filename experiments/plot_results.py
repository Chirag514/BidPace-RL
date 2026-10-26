"""Regenerate both report figures from the published evidence without training."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = {"traffic_dqn": "Traffic DQN", "campaign_dqn": "Campaign DQN",
           "sampled_q": "Sampled Bellman"}
CASES = [("standard", "Nominal"), ("competition_1.5", "Prices ×1.5"),
         ("evening_traffic_3", "Evening traffic ×3"), ("budget_0.15", "Budget 15%"),
         ("budget_0.60", "Budget 60%"), ("quality_0.5", "Click retention 50%"),
         ("coverage_0.5", "Coverage 50%")]


def plot_nominal(nominal, result, output):
    """Compare every RL candidate with the strongest test-observed baseline."""
    nominal = pd.read_csv(nominal).set_index("season")
    decisions = json.loads(Path(result).read_text(encoding="utf8"))["decisions"]
    values = pd.DataFrame({method: [nominal.loc[d["season"], method]
                                    - d["highest_baseline_clicks"] for d in decisions]
                           for method in METHODS}, index=[d["season"] for d in decisions])
    fig, ax = plt.subplots(figsize=(4.5, 2.75))
    try:
        x = np.arange(len(decisions))
        for j, (method, label) in enumerate(METHODS.items()):
            ax.bar(x + (j - 1) * .24, values[method], width=.24, label=label)
        ax.axhline(0, color="#333333", linewidth=.8)
        ax.set_xticks(x, [f"Season {season}" for season in values.index])
        ax.set_ylabel("Clicks/day minus strongest test baseline", fontsize=8)
        ax.legend(fontsize=7, loc="lower left")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=250, bbox_inches="tight", pad_inches=.08)
    finally:
        plt.close(fig)
    return values


def plot_stress(evaluation, output):
    """Weight each advertiser-day equally; return the plotted scenario means."""
    rows = pd.read_csv(evaluation)
    means = rows.groupby(["scenario", "policy"]).clicks.mean()
    values = pd.DataFrame({method: [means.loc[case, method] for case, _ in CASES]
                           for method in METHODS}, index=[case for case, _ in CASES])
    fig, ax = plt.subplots(figsize=(4.5, 3.7))
    try:
        y = np.arange(len(CASES))
        for j, (method, label) in enumerate(METHODS.items()):
            ax.barh(y + (j - 1) * .24, values[method], height=.23, label=label)
        ax.set_yticks(y, [label for _, label in CASES], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Recorded clicks per advertiser-day", fontsize=9)
        ax.tick_params(axis="x", labelsize=9)
        ax.set_xlim(left=0)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=.2)
        # Two rows keep the full legend within the narrow report column.
        legend = ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, 1.21),
                           ncol=2, frameon=False, handlelength=1, columnspacing=1)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(pad=.6)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=250, bbox_inches="tight", pad_inches=.08,
                    bbox_extra_artists=(legend,))
    finally:
        plt.close(fig)
    return values


def generate_figures(evidence="evidence", output="results/figures"):
    """Write the two PNGs embedded in the report and return their paths."""
    evidence, output = Path(evidence), Path(output)
    nominal, stress = output / "nominal.png", output / "stress.png"
    plot_nominal(evidence / "nominal.csv", evidence / "result.json", nominal)
    plot_stress(evidence / "evaluation.csv", stress)
    return nominal, stress


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", default="evidence")
    parser.add_argument("--out", default="results/figures", help="Output directory")
    args = parser.parse_args()
    for path in generate_figures(args.evidence, args.out):
        print(f"Created {path}")
