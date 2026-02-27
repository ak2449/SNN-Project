import matplotlib.pyplot as plt
import numpy as np

# --- Time axis ---
dt = 0.001
T = 1.0
t = np.arange(0, T, dt)

# --- Input spikes ---
in_spikes = np.array([0.32, 0.45])

# Discrete spike train
xin = np.zeros_like(t)
for st in in_spikes:
    idx = int(st / dt)
    if 0 <= idx < len(xin):
        xin[idx] = 1.0

# --- Synaptic kernels ---
def exp_kernel(tau, length=0.25):
    kt = np.arange(0, length, dt)
    return np.exp(-kt / tau)

k_exc = exp_kernel(tau=0.02, length=0.15)
k_inh = exp_kernel(tau=0.05, length=0.25)
inh_delay = int(0.01 / dt)  # 10 ms delay

# Currents
I_exc = np.convolve(xin, k_exc, mode="same")
xin_delayed = np.roll(xin, inh_delay)
xin_delayed[:inh_delay] = 0.0
I_inh = np.convolve(xin_delayed, k_inh, mode="same")

# --- LIF parameters ---
tau_m = 0.06
V_rest = 0.0
V_reset = 0.0
V_th = 1.0
R = 1.0
w_inh = 0.5

# --- Modified LIF simulation ---
V = np.zeros_like(t)
spk = np.zeros_like(t)
for i in range(1, len(t)):
    I_total = I_exc[i-1] - w_inh * I_inh[i-1]
    dV = (-(V[i-1] - V_rest) + R * I_total) * (dt / tau_m)
    V[i] = V[i-1] + dV
    if V[i] >= V_th:
        spk[i] = 1.0
        V[i] = V_reset

spike_times = t[spk > 0]

# --- Plot option 2 ---
fig, ax = plt.subplots(figsize=(9, 3.6))

# Membrane potential
ax.plot(t, V, linewidth=2, label=r"$U_{mem}(t)$")
ax.hlines(0.3, 0, T, linestyle="--", linewidth=1.5, label=r"$U_{th}$")

# Scale currents for visual overlay
# (purely for illustration; keeps them small and below membrane trace)
exc_scale = 0.35 / (I_exc.max() + 1e-9)
inh_scale = 0.35 / (I_inh.max() + 1e-9)

ax.plot(t, I_exc * exc_scale - 0.25, linewidth=1.5, alpha=0.9, label=r"$I_{exc}(t)$")
ax.plot(t, -I_inh * inh_scale - 0.25, linewidth=1.5, alpha=0.9, label=r"$-I_{inh}(t)$ ")

# Input spike markers along baseline
# for st in in_spikes:
#     ax.vlines(st, -0.45, -0.35, linewidth=1)

# Output spike markers
for st in spike_times:
    ax.vlines(st, 0, V_th, linewidth=1.5)
    ax.plot(st, V_th, marker="^", markersize=8)

# Labels/annotations
ax.text(-0.02, 0.4, r"$U_{mem}$", transform=ax.transAxes, ha="right", va="center", fontsize=12)
ax.text(-0.02, 0.6, r"$U_{th}$", transform=ax.transAxes, ha="right", va="center", fontsize=12)
ax.text(-0.02, 0.25, r"$I_{exc}, I_{inh}$", transform=ax.transAxes, ha="right", va="center", fontsize=11)

if len(spike_times) > 0:
    st0 = spike_times[0]
    ax.annotate(
        "spike",
        xy=(st0, V_th),
        xytext=(st0+0.05, V_th*1.05),
        arrowprops=dict(arrowstyle="->", lw=1.2),
        fontsize=10
    )

ax.set_xlim(0, 0.7)
ax.set_ylim(-0.7, 1)
ax.set_yticks([])
ax.set_xticks([])
# ax.set_frame_on(False)
ax.set_title("Modified LIF with excitatory & inhibitory currents")

# Simple legend outside
ax.legend(loc="upper right", frameon=False, fontsize=9)

plt.tight_layout()
plt.show()