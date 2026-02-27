import matplotlib.pyplot as plt
import numpy as np

# --- Time axis ---
dt = 0.001
T = 1.0
t = np.arange(0, T, dt)

# --- Input spikes (delta current events) ---
in_spikes = np.array([0.12, 0.45, 0.52,0.56, 0.80])

# --- LIF parameters (illustrative) ---
tau_m = 0.08          # membrane time constant (s)
V_rest = 0.0
V_reset = 0.0
V_th = 1.0
w = 0.55              # jump per input spike

# --- Simulate LIF correctly ---
V = np.zeros_like(t)
spikes_out = np.zeros_like(t)

for i in range(1, len(t)):
    # exponential decay toward rest
    V[i] = V[i-1] + (-(V[i-1] - V_rest)) * (dt / tau_m)

    # add instantaneous jumps at input spike times
    if np.any(np.isclose(t[i], in_spikes, atol=dt/2)):
        V[i] += w

    # threshold + reset
    if V[i] >= V_th:
        spikes_out[i] = 1.0
        V[i] = V_reset

out_spike_times = t[spikes_out > 0]

# --- Plot stacked cartoon ---
fig, axes = plt.subplots(
    3, 1, figsize=(9, 4),
    sharex=True,
    gridspec_kw=dict(height_ratios=[1, 1.5, 1], hspace=0.12)
)

# Top: I_in spikes
ax = axes[0]
ax.hlines(0, 0, 1, linewidth=1)
for st in in_spikes:
    ax.vlines(st, 0, 1, linewidth=2)
    ax.plot(st, 1, markersize=10)
ax.set_ylim(-0.1, 1.2)
ax.set_yticks([])
ax.set_xticks([])
ax.set_frame_on(False)
ax.text(-0.03, 0.5, r"$I_{in}$", va="center", ha="right", fontsize=14, transform=ax.transAxes)

# Middle: U_mem
ax = axes[1]
ax.plot(t, V, linewidth=2)
ax.hlines(V_th, 0, 1, linestyle="--", linewidth=1.5)
ax.text(0.98, V_th, r"$V_{th}$", ha="right", va="bottom", fontsize=10)
ax.vlines(0.56, 0, V_th, linewidth=2)

# vertical dotted guides at input spikes and output spikes
for st in in_spikes:
    ax.vlines(st, 0, V_th*1.1, linestyle=":", linewidth=1)
for st in out_spike_times:
    ax.vlines(st, 0, V_th*1.1, linestyle=":", linewidth=1.2)

# annotate leak + reset
ax.annotate(
    "leaky integration",
    xy=(0.18, 0.55), xycoords="data",
    xytext=(0.26, 0.8), textcoords="data",
    arrowprops=dict(arrowstyle="->", lw=1.2),
    fontsize=10
)
if len(out_spike_times) > 0:
    st0 = out_spike_times[0]
    ax.annotate(
        "spike + reset",
        xy=(st0, V_th*0.98), xycoords="data",
        xytext=(st0+0.06, 0.25), textcoords="data",
        arrowprops=dict(arrowstyle="->", lw=1.2),
        fontsize=10
    )

ax.set_ylim(-0.05, V_th*1.2)
ax.set_yticks([])
ax.set_xticks([])
ax.set_frame_on(False)
ax.text(-0.03, 0.5, r"$U_{mem}$", va="center", ha="right", fontsize=14, transform=ax.transAxes)

# Bottom: V_out spikes
ax = axes[2]
ax.hlines(0, 0, 1, linewidth=1)
for st in out_spike_times:
    ax.vlines(st, 0, 1, linewidth=2)
    ax.plot(st, 1, marker="^", markersize=10)
ax.set_ylim(-0.1, 1.2)
ax.set_yticks([])
ax.set_xticks([])
ax.set_frame_on(False)
ax.text(-0.03, 0.5, r"$V_{out}$", va="center", ha="right", fontsize=14, transform=ax.transAxes)

# Time label
axes[2].annotate(
    "t",
    xy=(0.95, -0.02), xycoords="axes fraction",
    ha="left", va="top", fontsize=18
)

plt.tight_layout()
plt.show()

