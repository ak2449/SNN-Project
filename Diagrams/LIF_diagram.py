import matplotlib.pyplot as plt
import numpy as np

# Time axis
dt = 0.001
T = 1.0
t = np.arange(0, T, dt)

# Stylized input current (piecewise)
I = np.zeros_like(t)
I[(t > 0.05) & (t < 0.25)] = 1.2
I[(t > 0.35) & (t < 0.55)] = 1.0
I[(t > 0.65) & (t < 0.85)] = 1.3

# LIF parameters (arbitrary units for illustration)
tau = 0.03
V_rest = 0.0
V_reset = 0.0
V_th = 1.0
R = 1.0

# Simulate LIF
V = np.zeros_like(t)
spikes = np.zeros_like(t)

for i in range(1, len(t)):
    dV = (-(V[i-1] - V_rest) + R * I[i-1]) * (dt / tau)
    V[i] = V[i-1] + dV
    if V[i] >= V_th:
        spikes[i] = 1
        V[i] = V_reset

spike_times = t[spikes > 0]

# Plot
fig = plt.figure(figsize=(9, 4))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.15)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)

# Top: membrane potential and threshold
ax1.plot(t, V, linewidth=2)
ax1.axhline(V_th, linestyle="--", linewidth=1.5)

# Spike indicators
for st in spike_times:
    ax1.vlines(st, V_reset, V_th, linewidth=1.5)
ax1.text(T * 0.98, V_th, "threshold", ha="right", va="bottom", fontsize=10)

# Annotations
ax1.annotate("leaky integration",
             xy=(0.12, 0.6), xycoords="data",
             xytext=(0.18, 0.8), textcoords="data",
             arrowprops=dict(arrowstyle="->", lw=1.2),
             fontsize=10)

if len(spike_times) > 0:
    st0 = spike_times[0]
    ax1.annotate("spike + reset",
                 xy=(st0, V_th), xycoords="data",
                 xytext=(st0 + 0.06, 0.25), textcoords="data",
                 arrowprops=dict(arrowstyle="->", lw=1.2),
                 fontsize=10)

ax1.set_ylabel("V(t)")
ax1.set_yticks([0, V_th])
ax1.set_yticklabels(["V_reset", "V_th"])
ax1.set_xlim(0, T)
ax1.set_title("Leaky Integrate-and-Fire (LIF) neuron cartoon")
ax1.spines[['top', 'right']].set_visible(False)
ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

# Input current on twin axis
ax1b = ax1.twinx()
ax1b.plot(t, I, linewidth=1, alpha=0.75)
ax1b.set_ylabel("input I(t)")
ax1b.set_yticks([])
ax1b.spines[['top', 'right']].set_visible(False)

# Bottom: spike raster
ax2.hlines(0, 0, T, linewidth=1)
for st in spike_times:
    ax2.vlines(st, -0.4, 0.4, linewidth=2)
ax2.set_ylim(-1, 1)
ax2.set_yticks([])
ax2.set_xlabel("time")
ax2.text(T * 0.98, 0.6, "spikes", ha="right", va="center", fontsize=10)
ax2.spines[['top', 'right', 'left']].set_visible(False)

plt.tight_layout()
plt.show()
