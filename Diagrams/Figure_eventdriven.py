import matplotlib.pyplot as plt
import numpy as np

# Time axis
t = np.linspace(0, 1.0, 1000)

# Continuous "dense" input signal (stylized bar + wiggle)
dense = 0.65 + 0.08*np.sin(2*np.pi*6*t) + 0.03*np.sin(2*np.pi*17*t)
nd = 0.65  + 0.08*np.sin(2*np.pi*4*t)

# Sparse spikes (event-driven)
spike_times = np.array([0.08, 0.18, 0.33, 0.47, 0.62, 0.76, 0.91])


def zero_crossing_times(t, x):
    t = np.asarray(t)
    x = np.asarray(x)

    # Indices where sign flips between samples
    flip_idx = np.where(np.diff(np.signbit(x)))[0]

    # Linear interpolation to estimate the actual crossing time
    t_cross = t[flip_idx] - x[flip_idx] * (t[flip_idx+1] - t[flip_idx]) / (x[flip_idx+1] - x[flip_idx])

    # If you want to also include exact zeros:
    exact_zero_idx = np.where(x == 0)[0]
    if exact_zero_idx.size:
        t_cross = np.sort(np.concatenate([t_cross, t[exact_zero_idx]]))

    return t_cross

sti = zero_crossing_times(t, 0.08*np.sin(2*np.pi*4*t))



fig, ax = plt.subplots(figsize=(9, 3))

# Plot dense input as a thick band
ax.plot(t, nd, linewidth=6, solid_capstyle='round',color ='orange')
ax.text(1.02, np.mean(nd), "continuous input\n(dense)", va='center', ha='left', fontsize=11)

# Plot sparse spikes as vertical lines on a baseline
baseline = 0.2
ax.hlines(baseline, 0, 1, linewidth=1)
for st in sti:
    ax.vlines(st, baseline, baseline+0.35, linewidth=2)

ax.text(1.02, baseline+0.18, "spikes\n(events)", va='center', ha='left', fontsize=11)

# Arrow indicating compute only on spikes
ax.annotate("compute only\non spikes",
            xy=(0.62, baseline+0.35), xycoords='data',
            xytext=(0.62, 0.75), textcoords='data',
            ha='center', va='bottom',
            arrowprops=dict(arrowstyle='->', lw=1.5))

# Cosmetics
ax.set_xlim(0.05, 0.95)
ax.set_ylim(0, 1.0)
ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)

plt.tight_layout()
plt.show()
