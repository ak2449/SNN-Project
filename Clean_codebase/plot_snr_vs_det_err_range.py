import matplotlib.pyplot as plt

BLUE = "#3B6CA8"
LIGHT_RED = "#D86B6B"

SNR_ROWS = [
    {"config_id": 5, "snr_db": 0, "det_err": 0.004},
    {"config_id": 4, "snr_db": -5, "det_err": 0.006},
    {"config_id": 7, "snr_db": 10, "det_err": 0.008},
    {"config_id": 8, "snr_db": 15, "det_err": 0.008},
    {"config_id": 6, "snr_db": 5, "det_err": 0.008},
    {"config_id": 3, "snr_db": -10, "det_err": 0.036},
    {"config_id": 2, "snr_db": -15, "det_err": 0.588},
    {"config_id": 1, "snr_db": -20, "det_err": 1.000},
]

CLUTTER_ROWS = [
    {"config_id": 1, "clutter_fraction": 0.0, "det_err": 0.000},
    {"config_id": 2, "clutter_fraction": 0.25, "det_err": 0.002},
    {"config_id": 3, "clutter_fraction": 0.5, "det_err": 0.018},
    {"config_id": 4, "clutter_fraction": 0.75, "det_err": 0.436},
    {"config_id": 5, "clutter_fraction": 1.0, "det_err": 0.830},
]

SCREENSHOT_ROWS = SNR_ROWS


def plot_detection_error(ax, rows, x_key, color, title, xlabel) -> None:
    rows = sorted(rows, key=lambda row: row[x_key])
    x_values = [row[x_key] for row in rows]
    det_err = [row["det_err"] for row in rows]

    ax.plot(x_values, det_err, marker="o", linewidth=2, color=color)
    ax.scatter(x_values, det_err, s=55, color=color, zorder=3)

    for x_val, y_val in zip(x_values, det_err):
        y_offset = -14 if y_val >= 0.95 else 8
        ax.annotate(
            f"{y_val:.3f}",
            xy=(x_val, y_val),
            xytext=(0, y_offset),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_xticks(x_values)
    ax.grid(True, alpha=0.3)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    plot_detection_error(
        axes[0],
        SNR_ROWS,
        "snr_db",
        BLUE,
        "SNR Against Detection Error",
        "SNR (dB)",
    )
    plot_detection_error(
        axes[1],
        CLUTTER_ROWS,
        "clutter_fraction",
        LIGHT_RED,
        "Clutter Against Detection Error",
        "Clutter Fraction",
    )

    axes[0].set_ylabel("Detection Error")
    axes[0].set_ylim(-0.02, 1.05)
    fig.suptitle("Detection Error Across SNR and Clutter Ranges")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()


if __name__ == "__main__":
    main()
