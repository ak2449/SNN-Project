"""Combined FMCW detection-error plot for SNR and clutter sweeps."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


BLUE = "#3B6CA8"
LIGHT_RED = "#D86B6B"
DEFAULT_DIR = Path(__file__).resolve().parent
DEFAULT_SNR_CSV = DEFAULT_DIR / "fmcw_snr_detection_error_summary.csv"
DEFAULT_CLUTTER_CSV = DEFAULT_DIR / "fmcw_clutter_detection_error_summary.csv"
DEFAULT_OUTPUT_PATH = DEFAULT_DIR / "fig_fmcw_snr_clutter_detection_error.png"


def _read_summary_csv(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    key: float(value)
                    for key, value in row.items()
                    if value not in (None, "")
                }
            )
    return rows


def _plot_detection_error(
    ax,
    rows: Sequence[Dict[str, float]],
    x_key: str,
    color: str,
    title: str,
    xlabel: str,
) -> None:
    if not rows:
        raise RuntimeError(f"No rows supplied for {title}.")

    rows = sorted(rows, key=lambda row: row[x_key])
    x_values = np.asarray([row[x_key] for row in rows], dtype=float)
    det_err = np.asarray(
        [row["detection_error_probability"] for row in rows],
        dtype=float,
    )

    ax.plot(x_values, det_err, marker="o", markersize=4, linewidth=2, color=color)
    ax.scatter(x_values, det_err, s=50, color=color, zorder=3)

    for x_value, y_value in zip(x_values, det_err):
        y_offset = -14 if y_value >= 0.95 else 8
        ax.annotate(
            f"{y_value:.3f}",
            xy=(x_value, y_value),
            xytext=(0, y_offset),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_xticks(x_values)
    ax.grid(True, alpha=0.3)


def plot_combined_detection_error(
    snr_rows: Sequence[Dict[str, float]],
    clutter_rows: Sequence[Dict[str, float]],
    output_path: Path,
) -> None:
    mpl_cache_dir = output_path.parent / ".mplcache"
    mpl_cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    _plot_detection_error(
        axes[0],
        snr_rows,
        "snr_db",
        BLUE,
        "FMCW SNR Against Detection Error",
        "SNR (dB)",
    )
    _plot_detection_error(
        axes[1],
        clutter_rows,
        "clutter_fraction",
        LIGHT_RED,
        "FMCW Clutter Against Detection Error",
        "Clutter fraction",
    )

    axes[0].set_ylabel("Detection error probability")
    axes[0].set_ylim(-0.03, 1.05)
    fig.suptitle("FMCW Detection Error Across SNR and Clutter Sweeps")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine FMCW SNR and clutter detection-error plots."
    )
    parser.add_argument("--snr-csv", type=Path, default=DEFAULT_SNR_CSV)
    parser.add_argument("--clutter-csv", type=Path, default=DEFAULT_CLUTTER_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snr_rows = _read_summary_csv(args.snr_csv)
    clutter_rows = _read_summary_csv(args.clutter_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_combined_detection_error(
        snr_rows=snr_rows,
        clutter_rows=clutter_rows,
        output_path=args.output,
    )
    print(f"Saved combined figure to {args.output}")


if __name__ == "__main__":
    main()
