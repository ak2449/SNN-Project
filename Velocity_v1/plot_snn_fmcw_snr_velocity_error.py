"""Overlay SNN and FMCW velocity SNR sweeps on one plot."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "snn_velocity_mplcache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SNN_BLUE = "#3B6CA8"
FMCW_RED = "#D86B6B"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "evaluation_outputs"
DEFAULT_SNN_CSV = DEFAULT_OUTPUT_DIR / "snr_mean_velocity_error_summary.csv"
DEFAULT_FMCW_CSV = DEFAULT_OUTPUT_DIR / "fmcw_snr_mean_velocity_error_summary.csv"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "fig_snn_fmcw_snr_mean_velocity_error.png"


def _read_summary_csv(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed_row: Dict[str, float] = {}
            repeated_header = False
            for key, value in row.items():
                if value in (None, ""):
                    continue
                try:
                    parsed_row[key] = float(value)
                except ValueError:
                    if value == key:
                        repeated_header = True
                        break
                    continue
            if repeated_header:
                continue
            rows.append(parsed_row)
    return rows


def _series_from_rows(rows: Sequence[Dict[str, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sorted_rows = sorted(rows, key=lambda row: row["x_value"])
    snr_db = np.asarray([row["x_value"] for row in sorted_rows], dtype=float)
    mean_abs_error = np.asarray(
        [row["mean_abs_velocity_error_m_s"] for row in sorted_rows],
        dtype=float,
    )
    std_abs_error = np.asarray(
        [row["std_abs_velocity_error_m_s"] for row in sorted_rows],
        dtype=float,
    )
    return snr_db, mean_abs_error, std_abs_error


def plot_snn_fmcw_snr_velocity_error(
    snn_rows: Sequence[Dict[str, float]],
    fmcw_rows: Sequence[Dict[str, float]],
    output_path: Path,
    log_y: bool = False,
) -> None:
    if not snn_rows:
        raise RuntimeError("No SNN summary rows were supplied.")
    if not fmcw_rows:
        raise RuntimeError("No FMCW summary rows were supplied.")

    snn_snr, snn_mae, snn_std = _series_from_rows(snn_rows)
    fmcw_snr, fmcw_mae, fmcw_std = _series_from_rows(fmcw_rows)

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.errorbar(
        snn_snr,
        snn_mae,
        yerr=snn_std,
        marker="o",
        linewidth=2,
        capsize=4,
        color=SNN_BLUE,
        ecolor=SNN_BLUE,
        elinewidth=1.4,
        label="SNN velocity MAE +/- SD",
    )
    ax.errorbar(
        fmcw_snr,
        fmcw_mae,
        yerr=fmcw_std,
        marker="s",
        linewidth=2,
        capsize=4,
        color=FMCW_RED,
        ecolor=FMCW_RED,
        elinewidth=1.4,
        label="FMCW velocity MAE +/- SD",
    )
    ax.scatter(snn_snr, snn_mae, s=52, color=SNN_BLUE, zorder=3)
    ax.scatter(fmcw_snr, fmcw_mae, s=52, color=FMCW_RED, zorder=3)

    ax.set_title("SNR Against Mean Absolute Velocity Error")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Mean absolute velocity error (m/s)")
    ax.set_xticks(sorted(set(snn_snr.tolist()) | set(fmcw_snr.tolist())))
    if log_y:
        ax.set_yscale("log")
    else:
        ax.set_yscale("linear")
    ax.grid(True, alpha=0.3, which="both" if log_y else "major")
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay SNN and FMCW SNR-vs-velocity-error summaries."
    )
    parser.add_argument("--snn-csv", type=Path, default=DEFAULT_SNN_CSV)
    parser.add_argument("--fmcw-csv", type=Path, default=DEFAULT_FMCW_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use a log y-axis instead of the default linear y-axis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snn_rows = _read_summary_csv(args.snn_csv)
    fmcw_rows = _read_summary_csv(args.fmcw_csv)
    plot_snn_fmcw_snr_velocity_error(
        snn_rows=snn_rows,
        fmcw_rows=fmcw_rows,
        output_path=args.output,
        log_y=args.log_y,
    )
    print(f"Saved combined SNN/FMCW velocity figure to {args.output}")


if __name__ == "__main__":
    main()
