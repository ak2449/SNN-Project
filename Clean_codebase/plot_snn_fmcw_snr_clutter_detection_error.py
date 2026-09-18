"""Overlay SNN and FMCW range detection-error sweeps.

This combines the existing SNN SNR/clutter detection-error points with the
generated FMCW SNR/clutter summary CSVs into one two-panel comparison plot.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


SNR_BLUE = "#3B6CA8"
CLUTTER_RED = "#D86B6B"
FMCW_ORANGE = "#F28E2B"
SNN_ALPHA = 1
FMCW_ALPHA = 0.6
DEFAULT_DIR = Path(__file__).resolve().parent
DEFAULT_FMCW_SNR_CSV = DEFAULT_DIR / "fmcw_snr_detection_error_summary.csv"
DEFAULT_FMCW_CLUTTER_CSV = DEFAULT_DIR / "fmcw_clutter_detection_error_summary.csv"
DEFAULT_OUTPUT_PATH = DEFAULT_DIR / "fig_snn_fmcw_snr_clutter_detection_error.png"

DEFAULT_SNN_SNR_ROWS = [
    {"config_id": 5, "snr_db": 0.0, "det_err": 0.004},
    {"config_id": 4, "snr_db": -5.0, "det_err": 0.006},
    {"config_id": 7, "snr_db": 10.0, "det_err": 0.008},
    {"config_id": 8, "snr_db": 15.0, "det_err": 0.008},
    {"config_id": 6, "snr_db": 5.0, "det_err": 0.008},
    {"config_id": 3, "snr_db": -10.0, "det_err": 0.036},
    {"config_id": 2, "snr_db": -15.0, "det_err": 0.588},
    {"config_id": 1, "snr_db": -20.0, "det_err": 1.000},
]

DEFAULT_SNN_CLUTTER_ROWS = [
    {"config_id": 1, "clutter_fraction": 0.0, "det_err": 0.000},
    {"config_id": 2, "clutter_fraction": 0.25, "det_err": 0.002},
    {"config_id": 3, "clutter_fraction": 0.5, "det_err": 0.018},
    {"config_id": 4, "clutter_fraction": 0.75, "det_err": 0.436},
    {"config_id": 5, "clutter_fraction": 1.0, "det_err": 0.830},
]


def _read_numeric_csv(path: Path) -> List[Dict[str, float]]:
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


def _load_rows(
    optional_csv: Optional[Path],
    fallback_rows: Sequence[Dict[str, float]],
) -> List[Dict[str, float]]:
    if optional_csv is None:
        return [dict(row) for row in fallback_rows]
    return _read_numeric_csv(optional_csv)


def _value_from_row(row: Dict[str, float]) -> float:
    if "detection_error_probability" in row:
        return row["detection_error_probability"]
    if "det_err" in row:
        return row["det_err"]
    raise KeyError(
        "Expected either 'detection_error_probability' or 'det_err' in row."
    )


def _series_from_rows(
    rows: Sequence[Dict[str, float]],
    x_key: str,
) -> Tuple[np.ndarray, np.ndarray]:
    usable_rows = [row for row in rows if x_key in row]
    if not usable_rows:
        raise RuntimeError(f"No rows with '{x_key}' were supplied.")

    sorted_rows = sorted(usable_rows, key=lambda row: row[x_key])
    x_values = np.asarray([row[x_key] for row in sorted_rows], dtype=float)
    det_err = np.asarray([_value_from_row(row) for row in sorted_rows], dtype=float)
    return x_values, det_err


def _plot_method(
    ax,
    rows: Sequence[Dict[str, float]],
    x_key: str,
    label: str,
    color: str,
    alpha: float,
    marker: str,
) -> np.ndarray:
    x_values, det_err = _series_from_rows(rows, x_key)
    ax.plot(
        x_values,
        det_err,
        marker=marker,
        markersize=5,
        linewidth=2,
        color=color,
        alpha=alpha,
        label=label,
    )
    ax.scatter(x_values, det_err, s=52, color=color, alpha=alpha, zorder=3)
    return x_values


def _plot_comparison_axis(
    ax,
    snn_rows: Sequence[Dict[str, float]],
    fmcw_rows: Sequence[Dict[str, float]],
    x_key: str,
    title: str,
    xlabel: str,
    snn_color: str,
    fmcw_color: str = FMCW_ORANGE,
) -> None:
    snn_x = _plot_method(
        ax=ax,
        rows=snn_rows,
        x_key=x_key,
        label="SNN",
        color=snn_color,
        alpha=SNN_ALPHA,
        marker="o",
    )
    fmcw_x = _plot_method(
        ax=ax,
        rows=fmcw_rows,
        x_key=x_key,
        label="FMCW",
        color=fmcw_color,
        alpha=FMCW_ALPHA,
        marker="s",
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_xticks(sorted(set(snn_x.tolist()) | set(fmcw_x.tolist())))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


def plot_snn_fmcw_detection_error(
    snn_snr_rows: Sequence[Dict[str, float]],
    snn_clutter_rows: Sequence[Dict[str, float]],
    fmcw_snr_rows: Sequence[Dict[str, float]],
    fmcw_clutter_rows: Sequence[Dict[str, float]],
    output_path: Path,
) -> None:
    mpl_cache_dir = output_path.parent / ".mplcache"
    mpl_cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3), sharey=True)

    _plot_comparison_axis(
        ax=axes[0],
        snn_rows=snn_snr_rows,
        fmcw_rows=fmcw_snr_rows,
        x_key="snr_db",
        title="SNR Against Detection Error",
        xlabel="SNR (dB)",
        snn_color=SNR_BLUE,
    )
    _plot_comparison_axis(
        ax=axes[1],
        snn_rows=snn_clutter_rows,
        fmcw_rows=fmcw_clutter_rows,
        x_key="clutter_fraction",
        title="Clutter Against Detection Error",
        xlabel="Clutter fraction",
        snn_color=CLUTTER_RED,
    )

    axes[0].set_ylabel("Detection error probability")
    axes[0].set_ylim(-0.03, 1.05)
    fig.suptitle("SNN and FMCW Detection Error Across SNR and Clutter Sweeps")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay SNN and FMCW SNR/clutter detection-error plots."
    )
    parser.add_argument("--fmcw-snr-csv", type=Path, default=DEFAULT_FMCW_SNR_CSV)
    parser.add_argument(
        "--fmcw-clutter-csv",
        type=Path,
        default=DEFAULT_FMCW_CLUTTER_CSV,
    )
    parser.add_argument(
        "--snn-snr-csv",
        type=Path,
        default=None,
        help="Optional SNN SNR CSV. Defaults to the existing plotted SNN values.",
    )
    parser.add_argument(
        "--snn-clutter-csv",
        type=Path,
        default=None,
        help="Optional SNN clutter CSV. Defaults to the existing plotted SNN values.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snn_snr_rows = _load_rows(args.snn_snr_csv, DEFAULT_SNN_SNR_ROWS)
    snn_clutter_rows = _load_rows(
        args.snn_clutter_csv,
        DEFAULT_SNN_CLUTTER_ROWS,
    )
    fmcw_snr_rows = _read_numeric_csv(args.fmcw_snr_csv)
    fmcw_clutter_rows = _read_numeric_csv(args.fmcw_clutter_csv)

    plot_snn_fmcw_detection_error(
        snn_snr_rows=snn_snr_rows,
        snn_clutter_rows=snn_clutter_rows,
        fmcw_snr_rows=fmcw_snr_rows,
        fmcw_clutter_rows=fmcw_clutter_rows,
        output_path=args.output,
    )
    print(f"Saved combined SNN/FMCW detection-error figure to {args.output}")


if __name__ == "__main__":
    main()
