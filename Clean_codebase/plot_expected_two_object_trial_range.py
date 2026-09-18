from pathlib import Path
import os

import numpy as np

MPL_CACHE_DIR = Path(__file__).with_name(".mplcache")
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


OUTPUT_PATH = Path(__file__).with_name("fig_expected_two_object_trial_range.png")

BLUE = "#3B6CA8"
RED = "#D86B6B"
GREEN = "#3E8F65"
DARK = "#333333"
GREY = "#8A8A8A"


def gaussian(x_values: np.ndarray, center: float, sigma: float, height: float = 1.0) -> np.ndarray:
    return height * np.exp(-0.5 * ((x_values - center) / sigma) ** 2)


def plot_geometry(ax: plt.Axes) -> None:
    bat_x = 0.0
    r1 = 1.50
    r2 = 1.80

    ax.hlines(0, bat_x, 2.12, color="#D0D0D0", linewidth=2)
    ax.scatter([bat_x], [0], s=140, color=DARK, zorder=3)
    ax.scatter([r1, r2], [0, 0], s=160, color=[BLUE, GREEN], zorder=3)

    ax.annotate("", xy=(r1, 0.16), xytext=(bat_x, 0.16), arrowprops={"arrowstyle": "<->", "color": BLUE})
    ax.annotate("", xy=(r2, -0.16), xytext=(bat_x, -0.16), arrowprops={"arrowstyle": "<->", "color": GREEN})
    ax.annotate("", xy=(r2, 0.34), xytext=(r1, 0.34), arrowprops={"arrowstyle": "<->", "color": RED})

    ax.text(bat_x, -0.08, "pulse", ha="center", va="top", fontsize=9)
    ax.text(r1, -0.08, "object 1", ha="center", va="top", fontsize=9, color=BLUE)
    ax.text(r2, -0.08, "object 2", ha="center", va="top", fontsize=9, color=GREEN)
    ax.text((bat_x + r1) / 2.0, 0.20, "r1", ha="center", va="bottom", fontsize=9, color=BLUE)
    ax.text((bat_x + r2) / 2.0, -0.29, "r2", ha="center", va="top", fontsize=9, color=GREEN)
    ax.text((r1 + r2) / 2.0, 0.38, "separation", ha="center", va="bottom", fontsize=9, color=RED)

    ax.set_xlim(-0.15, 2.2)
    ax.set_ylim(-0.42, 0.52)
    ax.set_title("Clean Two-Object Range Trial")
    ax.axis("off")


def plot_close_population(ax: plt.Axes) -> None:
    delays = np.linspace(80, 100, 600)
    true_delays = [87.0, 89.0]
    response = gaussian(delays, true_delays[0], 2.5, 1.0) + gaussian(delays, true_delays[1], 2.5, 0.92)
    response = response / response.max()
    centroid = float(np.sum(delays * response) / np.sum(response))

    ax.plot(delays, response, color=BLUE, linewidth=2)
    for delay in true_delays:
        ax.axvline(delay, color=DARK, linestyle="--", linewidth=1.1)
    ax.scatter([centroid], [np.interp(centroid, delays, response)], s=65, color=RED, zorder=4)
    ax.text(centroid + 0.4, 0.86, "one fused centroid", color=RED, fontsize=9)

    ax.set_title("Small Separation: Fused Response")
    ax.set_xlabel("Delay-bank neuron")
    ax.set_ylabel("Population activity")
    ax.set_ylim(0, 1.08)
    ax.grid(True, alpha=0.25)


def plot_resolved_population(ax: plt.Axes) -> None:
    delays = np.linspace(80, 124, 800)
    true_delays = [87.0, 117.0]
    response_1 = gaussian(delays, true_delays[0], 2.7, 1.0)
    response_2 = gaussian(delays, true_delays[1], 2.7, 0.78)
    response = response_1 + response_2
    response = response / response.max()

    window_1 = delays < 101
    window_2 = delays > 103
    centroid_1 = float(np.sum(delays[window_1] * response[window_1]) / np.sum(response[window_1]))
    centroid_2 = float(np.sum(delays[window_2] * response[window_2]) / np.sum(response[window_2]))

    ax.plot(delays, response, color=GREEN, linewidth=2)
    for delay in true_delays:
        ax.axvline(delay, color=DARK, linestyle="--", linewidth=1.1)
    ax.scatter(
        [centroid_1, centroid_2],
        [np.interp(centroid_1, delays, response), np.interp(centroid_2, delays, response)],
        s=65,
        color=RED,
        zorder=4,
    )
    ax.text(centroid_1 - 5.8, 0.88, "centroid 1", color=RED, fontsize=9)
    ax.text(centroid_2 - 3.6, 0.68, "centroid 2", color=RED, fontsize=9)

    ax.set_title("Large Separation: Two Centroids")
    ax.set_xlabel("Delay-bank neuron")
    ax.set_ylabel("Population activity")
    ax.set_ylim(0, 1.08)
    ax.grid(True, alpha=0.25)


def plot_expected_error(ax: plt.Axes) -> None:
    separation_cm = np.linspace(1, 50, 300)
    transition_cm = 7.0
    slope_cm = 1.4
    expected_error = 1.0 / (1.0 + np.exp((separation_cm - transition_cm) / slope_cm))
    expected_error = 0.03 + 0.94 * expected_error

    ax.plot(separation_cm, expected_error, color=RED, linewidth=2.4)
    ax.fill_between(separation_cm, expected_error, 0, color=RED, alpha=0.12)
    ax.axvline(transition_cm, color=DARK, linestyle=":", linewidth=1.2)
    ax.text(transition_cm + 1.0, 0.63, "resolution threshold", fontsize=9, color=DARK)

    ax.set_title("Expected Detection Error Trend")
    ax.set_xlabel("Object separation (cm)")
    ax.set_ylabel("Detection error probability")
    ax.set_xlim(1, 50)
    ax.set_ylim(0, 1.03)
    ax.grid(True, alpha=0.25)


def main() -> None:
    fig = plt.figure(figsize=(11.2, 7.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.78, 1.0], hspace=0.42, wspace=0.28)

    geometry_ax = fig.add_subplot(grid[0, :])
    close_ax = fig.add_subplot(grid[1, 0])
    resolved_ax = fig.add_subplot(grid[1, 1])

    plot_geometry(geometry_ax)
    plot_close_population(close_ax)
    plot_resolved_population(resolved_ax)

    inset_ax = fig.add_axes([0.36, 0.54, 0.28, 0.19])
    plot_expected_error(inset_ax)
    inset_ax.tick_params(labelsize=8)
    inset_ax.title.set_fontsize(9)
    inset_ax.xaxis.label.set_fontsize(8)
    inset_ax.yaxis.label.set_fontsize(8)

    fig.suptitle(
        "Expected Population-Centroid Decoding Behavior for a Clean Two-Object Trial",
        fontsize=14,
        y=0.98,
    )
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
