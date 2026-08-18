"""Create the support-mapping fragility figure."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt


def concise_label(row: dict) -> str:
    author = row["author"].replace("Rasmussen", "Rasmussen 2017")
    replacements = (
        ("Nearest poultry CAFO", "poultry-facility distance"),
        ("Nearest goats CAFO", "goat-facility distance"),
        ("Nearest CAFO", "facility distance"),
        ("< or =", "at most"),
        ("Mapped complementary distance support beyond", "mapped distance >"),
        ("Presence of poultry farm within", "poultry farm present within"),
        ("No poultry farm within", "poultry farm absent within"),
        ("Presence of swine farm animals within", "swine farm present within"),
        ("No swine farm animals within", "swine farm absent within"),
        (" (animal-count proxy)", ""),
        (" (binary-buffer proxy)", ""),
    )
    comparison = row["comparison"]
    reference = row["reference"]
    for old, new in replacements:
        comparison = comparison.replace(old, new)
        reference = reference.replace(old, new)
    comparison = re.sub(
        r"(\d+) m < ([^<]+) < (\d+) m",
        lambda match: f"{match.group(2).strip()} {match.group(1)}--{match.group(3)} m",
        comparison,
    )
    reference = re.sub(
        r"(\d+) m < ([^<]+) < (\d+) m",
        lambda match: f"{match.group(2).strip()} {match.group(1)}--{match.group(3)} m",
        reference,
    )
    comparison = re.sub(r"\s*<\s*", " < ", comparison)
    reference = re.sub(r"\s*>\s*", " > ", reference)
    if author == "Rasmussen 2017":
        return "Rasmussen 2017: facility distance < 3 miles vs > 3 miles or no facility within 3 miles"
    text = f"{author}: {comparison} vs {reference}"
    return textwrap.shorten(text, width=92, placeholder="...")


def main(results_file: Path, output_file: Path, top_n: int = 10) -> None:
    results = json.loads(results_file.read_text(encoding="utf-8"))
    rows = results["support_mapping_fragility"]["rows"][:top_n]
    rows = list(reversed(rows))
    values = [1000 * row["beta_change_for_10pct_design_shift"] for row in rows]
    labels = [concise_label(row) for row in rows]
    colors = ["#2F6B9A" if value >= 0 else "#C45A3C" for value in values]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.barh(range(len(rows)), values, color=colors, height=0.68)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(range(len(rows)), labels)
    ax.set_xlabel(r"Change in $\hat\beta_1$ ($\times 10^{-3}$) for a 10% local shift in one mapped contrast")
    ax.set_title("Local fragility of the primary AFO working support mapping")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8.3)
    fig.subplots_adjust(left=0.56, right=0.98, bottom=0.14, top=0.90)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight")
    fig.savefig(output_file.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()
    main(args.results, args.output, args.top_n)
