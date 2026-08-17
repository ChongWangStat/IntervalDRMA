"""Recreate the regional Gamma density figure as vector and high-resolution art."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma

from mixed_support_simulation import REGION_PARAMS


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "manuscript" / "images"


def main() -> None:
    displayed = {
        "Pooled five-region scenario": (2.178, 2.722, "#000000", 3.0),
        "Iowa external scenario": (*REGION_PARAMS["Iowa"], "#377EB8", 2.3),
        "Netherlands proxy": (*REGION_PARAMS["Netherlands"], "#FFB74D", 2.3),
        "North Carolina permit layer": (*REGION_PARAMS["NC"], "#E84A5F", 2.3),
        "Pennsylvania proxy": (*REGION_PARAMS["PA"], "#63B35D", 2.3),
        "Wisconsin permit layer": (*REGION_PARAMS["Wisconsin"], "#7B3294", 2.3),
    }
    x = np.linspace(0.001, 20.0, 3000)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.13, top=0.80)
    for label, (shape, scale, color, linewidth) in displayed.items():
        ax.plot(
            x,
            gamma.pdf(x, a=shape, scale=scale),
            color=color,
            linewidth=linewidth,
            label=rf"{label} ($\alpha={shape:.3g}$, $\theta={scale:.3g}$)",
        )

    fig.suptitle(
        "Estimated Gamma distance distributions by region",
        x=0.10,
        y=0.96,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.10,
        0.90,
        r"Gamma($\alpha$, $\theta$) working models from stored population-weighted fits",
        fontsize=11,
        color="#555555",
    )
    ax.set_xlabel("Distance to recorded facility or proxy site (miles)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 20)
    ax.set_ylim(bottom=0)
    ax.grid(axis="both", color="#E5E5E5", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, fontsize=8.5, loc="upper right")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "Density_plot_Sum_Updated.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "Density_plot_Sum_Updated.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
