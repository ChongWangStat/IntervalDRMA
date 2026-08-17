"""Create the direct support-score comparison figure used in the manuscript."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "manuscript" / "images"


def _forest_panel(ax, labels, estimates, lower, upper, colors, xlabel, null=None, digits=3):
    y = np.arange(len(labels))[::-1]
    estimates = np.asarray(estimates, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    for idx, ypos in enumerate(y):
        ax.errorbar(
            estimates[idx],
            ypos,
            xerr=[[estimates[idx] - lower[idx]], [upper[idx] - estimates[idx]]],
            fmt="o",
            color=colors[idx],
            ecolor=colors[idx],
            elinewidth=1.8,
            capsize=3,
            markersize=5.5,
            zorder=3,
        )
        ax.annotate(
            f"{estimates[idx]:.{digits}f} ({lower[idx]:.{digits}f}, {upper[idx]:.{digits}f})",
            (upper[idx], ypos),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color="#333333",
        )
    if null is not None:
        ax.axvline(null, color="#555555", linestyle=":", linewidth=1.2, zorder=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def main() -> None:
    afo = json.loads((RESULTS / "analysis_results.json").read_text(encoding="utf-8"))
    radon = json.loads(
        (RESULTS / "radon_example_results.json").read_text(encoding="utf-8")
    )

    afo_rows = [
        ("All mapped contrasts\nconditional mean (primary)", afo["afo_primary_working_analysis"]),
        ("All mapped contrasts\nmidpoint", afo["afo_primary_midpoint_comparison"]),
        ("OR-only\nrestriction", afo["afo_or_only_restriction"]),
        ("Explicit-support OR\nrestriction", afo["afo_explicit_support_restriction"]),
    ]
    afo_est = [row[1]["random_effects"]["beta"] for row in afo_rows]
    afo_lo = [row[1]["random_effects"]["beta_ci"][0] for row in afo_rows]
    afo_hi = [row[1]["random_effects"]["beta_ci"][1] for row in afo_rows]

    radon_rows = [
        ("Historical Lognormal", radon["primary_historical_distribution"]),
        ("UNSCEAR 2024\nLognormal", radon["unscear_2024_distribution_sensitivity"]),
    ]
    tail_labels = ["Midpoint; open tail\n1.2 x lower bound", "Midpoint; open tail\n1.5 x lower bound", "Midpoint; open tail\n2.0 x lower bound"]
    radon_rows.extend(zip(tail_labels, radon["point_score_sensitivity"]))
    radon_est = [row[1]["random_effects"]["or_per_100_bq_m3"] for row in radon_rows]
    radon_lo = [row[1]["random_effects"]["or_ci_per_100_bq_m3"][0] for row in radon_rows]
    radon_hi = [row[1]["random_effects"]["or_ci_per_100_bq_m3"][1] for row in radon_rows]

    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), constrained_layout=True)
    _forest_panel(
        axes[0],
        [row[0] for row in afo_rows],
        afo_est,
        afo_lo,
        afo_hi,
        ["#1768AC", "#D95F02", "#55A868", "#8172B2"],
        r"Random-effects slope, $\hat{\beta}_1$ per mile",
        null=0.0,
    )
    axes[0].set_title("A. AFO illustration", loc="left", fontweight="bold")

    _forest_panel(
        axes[1],
        [row[0] for row in radon_rows],
        radon_est,
        radon_lo,
        radon_hi,
        ["#1768AC", "#4C92C3", "#D95F02", "#E88D31", "#F2B66D"],
        r"Random-effects OR per 100 Bq/m$^3$",
        null=1.0,
        digits=3,
    )
    axes[1].set_title("B. Residential-radon illustration", loc="left", fontweight="bold")

    for ax in axes:
        ax.margins(x=0.28, y=0.22)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "Score_assumption_comparison.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "Score_assumption_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
