#!/usr/bin/env python
"""Build the paper figure for the controlled representation-alignment test."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    PROJECT_ROOT / "debug/representation_alignment/head003_native_normal_trajectory"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "paper/CVPR2027_WSM/fig/representation_alignment_learning_curves.pdf"
)
REPRESENTATIONS = (
    ("native", "Native", "#1f4e79", "o", 2.8),
    ("vae-channel-cycle-15", "Permutation A", "#c44e52", "s", 1.5),
    ("vae-channel-cycle-01", "Permutation B", "#dd8452", "^", 1.5),
    ("vae-channel-cycle-03", "Permutation C", "#8172b3", "D", 1.5),
    ("vae-channel-cycle-07", "Permutation D", "#55a868", "v", 1.5),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    trajectories: dict[str, list[dict[str, str]]] = {}
    for key, *_ in REPRESENTATIONS:
        trajectories[key] = read_csv(args.input / key / "trajectory.csv")

    native_samples = {
        row["sample_id"]: float(row["normal_mean_deg"])
        for row in read_csv(
            args.input / "native/step_0001000/native_normal_samples.csv"
        )
    }
    paired_differences: list[tuple[int, float]] = []
    for permutation_index, (key, *_rest) in enumerate(REPRESENTATIONS[1:], start=1):
        rows = read_csv(
            args.input / key / "step_0001000/native_normal_samples.csv"
        )
        for row in rows:
            sample_id = row["sample_id"]
            paired_differences.append(
                (
                    permutation_index,
                    float(row["normal_mean_deg"]) - native_samples[sample_id],
                )
            )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.55), gridspec_kw={"wspace": 0.30})

    ax = axes[0]
    for key, label, color, marker, width in REPRESENTATIONS:
        rows = trajectories[key]
        steps = np.asarray([int(row["step"]) for row in rows])
        errors = np.asarray([float(row["normal_mean_deg"]) for row in rows])
        ax.plot(
            steps,
            errors,
            color=color,
            marker=marker,
            linewidth=width,
            markersize=4.0 if key == "native" else 3.2,
            label=label,
            zorder=4 if key == "native" else 2,
        )
    ax.set_title("(a) Learning trajectories")
    ax.set_xlabel("Training updates")
    ax.set_ylabel("Mean normal error (degrees)")
    ax.set_xlim(0, 1020)
    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper right", columnspacing=0.8)

    ax = axes[1]
    for index in range(1, 5):
        values = np.asarray([value for group, value in paired_differences if group == index])
        jitter = np.linspace(-0.11, 0.11, num=len(values))
        ax.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=12,
            alpha=0.72,
            color=REPRESENTATIONS[index][2],
            edgecolors="none",
            zorder=2,
        )
        mean = float(np.mean(values))
        sem = float(np.std(values, ddof=1) / np.sqrt(len(values)))
        ax.errorbar(
            index,
            mean,
            yerr=2.131 * sem,
            fmt="o",
            color="#202020",
            markerfacecolor="white",
            markersize=5,
            capsize=3,
            linewidth=1.2,
            zorder=4,
        )
    ax.axhline(0, color="#303030", linewidth=0.9)
    ax.set_title("(b) Native advantage at update 1,000")
    ax.set_ylabel("Permutation error $-$ native error (degrees)")
    ax.set_xticks(range(1, 5), ["A", "B", "C", "D"])
    ax.set_xlabel("Exactly invertible target permutation")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
