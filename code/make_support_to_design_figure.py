"""Create the vector support-to-design framework diagram."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "manuscript" / "images"


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(12.0, 3.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (
            0.025,
            "Reported adjusted\ncontrasts",
            "Study 1: (0, 2] vs (4, 8]\nStudy 2: {1.5} vs {5}\nStudy 3: (1, 10] vs (0, 5]",
            "#E8F1F8",
        ),
        (
            0.275,
            "Outcome-independent\nsupport model $F_k$",
            r"$m_k(S)=E_{F_k}\{h(D)\mid D\in S\}$",
            "#E9F4EE",
        ),
        (
            0.525,
            "Contrast design",
            r"$x_{kj}=m_k(A_{kj})-m_k(B_{kj})$",
            "#FFF3D8",
        ),
        (
            0.775,
            "Conventional synthesis",
            "GLS or profile REML\nworking covariance\nrandom exposure slopes",
            "#EEEAF6",
        ),
    ]
    width = 0.20
    height = 0.52
    y = 0.31
    for x, title, body, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=1.15,
            edgecolor="#1F4E79",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(
            x + width / 2,
            y + 0.37,
            title,
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=9,
            linespacing=1.1,
        )
        ax.text(x + width / 2, y + 0.19, body, ha="center", va="center", linespacing=1.35)

    for left in (0.225, 0.475, 0.725):
        ax.add_patch(
            FancyArrowPatch(
                (left + 0.006, y + height / 2),
                (left + 0.038, y + height / 2),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.25,
                color="#2F6B9A",
            )
        )

    ax.text(0.40, 0.17, "support-model uncertainty", ha="center", va="center", color="#456B52")
    ax.text(
        0.73,
        0.17,
        "synthesis uncertainty conditional on the mapping",
        ha="center",
        va="center",
        color="#5F537F",
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "Support_to_design_framework.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "Support_to_design_framework.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
