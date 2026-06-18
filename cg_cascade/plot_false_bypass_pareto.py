#!/usr/bin/env python
"""Plot the false-bypass/cost frontier used in the cost-aware triage triage paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def main() -> None:
    # Values are the Table 4 TF-IDF triage operating points.
    natural = [
        (0.0, 16.9, r"$\tau=0.01$"),
        (0.1, 23.2, "NP-cal"),
        (0.4, 26.6, r"$\tau=0.03$"),
        (7.8, 43.0, r"$\tau=0.10$"),
    ]
    balanced = [
        (0.1, 10.1, r"$\tau=.01$"),
        (0.2, 15.2, "NP-cal"),
        (0.4, 17.2, r"$\tau=.03$"),
    ]

    fig, ax = plt.subplots(figsize=(4.15, 2.55))

    ax.axvspan(0, 0.5, color="#d8ead2", alpha=0.85, label=r"$\leq 0.5\%$ FB")
    ax.plot(
        [x for x, _, _ in natural],
        [y for _, y, _ in natural],
        marker="o",
        linewidth=1.8,
        markersize=4.5,
        color="#1f77b4",
        label="PromptShield natural",
    )
    ax.plot(
        [x for x, _, _ in balanced],
        [y for _, y, _ in balanced],
        marker="s",
        linewidth=1.5,
        markersize=4.2,
        linestyle="--",
        color="#ff7f0e",
        label="PromptShield balanced",
    )

    label_offsets = {
        r"$\tau=0.01$": (18, -20, "left"),
        "NP-cal": (18, -2, "left"),
        r"$\tau=0.03$": (18, 14, "left"),
        r"$\tau=0.10$": (-10, -20, "right"),
    }
    for x, y, label in natural:
        dx, dy, ha = label_offsets[label]
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            ha=ha,
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.7},
        )

    ax.set_xlabel("Malicious false-bypass rate (%)", fontsize=8.5)
    ax.set_ylabel("LLM-call reduction (%)", fontsize=8.5)
    ax.set_xlim(-0.15, 8.4)
    ax.set_ylim(0, 46)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, alpha=0.28, linewidth=0.6)
    ax.legend(loc="lower right", fontsize=7.5, frameon=True)
    fig.tight_layout(pad=0.25)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "false_bypass_pareto.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "false_bypass_pareto.png", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
