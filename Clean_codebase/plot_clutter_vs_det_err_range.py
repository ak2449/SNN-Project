import matplotlib.pyplot as plt

LIGHT_RED = "#D86B6B"

SCREENSHOT_ROWS = [
    {"config_id": 1, "clutter_fraction": 0.0, "det_err": 0.000},
    {"config_id": 2, "clutter_fraction": 0.25, "det_err": 0.002},
    {"config_id": 3, "clutter_fraction": 0.5, "det_err": 0.018},
    {"config_id": 4, "clutter_fraction": 0.75, "det_err": 0.436},
    {"config_id": 5, "clutter_fraction": 1.0, "det_err": 0.830},
]


def main() -> None:
    rows = sorted(SCREENSHOT_ROWS, key=lambda row: row["clutter_fraction"])
    clutter_fraction = [row["clutter_fraction"] for row in rows]
    det_err = [row["det_err"] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(clutter_fraction, det_err, marker="o", linewidth=2, color=LIGHT_RED)
    ax.scatter(clutter_fraction, det_err, s=55, color=LIGHT_RED, zorder=3)

    for x_val, y_val in zip(clutter_fraction, det_err):
        ax.annotate(
            f"{y_val:.3f}",
            xy=(x_val, y_val),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    ax.set_title("Clutter Against Detection Error")
    ax.set_xlabel("Clutter Fraction")
    ax.set_ylabel("Detection Error")
    ax.set_xticks(clutter_fraction)
    ax.set_ylim(-0.02, 0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
