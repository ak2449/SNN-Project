import torch
import torch.nn as nn
import snntorch as snn
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------
# Single delay-tuned FMFM neuron (step-by-step capable)
# -------------------------------
class FMFMNeuron(nn.Module):
    def __init__(self, delay, chosen_w1, ute=1.01, beta=0.9, threshold=1.0):
        super().__init__()
        self.beta = float(beta)
        self.chosen_w1 = float(chosen_w1)
        self.ute = float(ute)
        self.threshold = float(threshold)

        self.w1, self.w2 = self.calculate_weights(delay)

        self.fc = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.fc.weight[:] = torch.tensor([[self.w1, self.w2]], dtype=torch.float32)

        # explicit threshold for clarity
        self.lif = snn.Leaky(beta=beta, threshold=threshold)

    def calculate_weights(self, delay: int):
        # Design condition at echo time te:
        # U(te) = beta^delay * w1 + w2 = ute  => w2 = ute - beta^delay * w1
        w1 = self.chosen_w1
        w2 = self.ute - (self.beta ** int(delay)) * w1
        return float(w1), float(w2)

    def init_state(self, batch_size: int, device: str):
        # snntorch returns a tensor state; we keep it and pass it back each step
        mem = self.lif.init_leaky()
        if isinstance(mem, torch.Tensor):
            mem = mem.to(device)
        return mem

    def step(self, x_t, mem):
        """
        x_t: [B,2] spikes at time t
        mem: neuron state
        returns: spk_t [B,1], mem_next
        """
        cur = self.fc(x_t)          # [B,1]
        spk, mem = self.lif(cur, mem)
        return spk, mem


# -------------------------------
# Pulse-echo generator
# -------------------------------
def generate_pulse_echo(num_steps=50, delay=10, batch_size=1, t_pulse=5, device="cpu"):
    spikes = torch.zeros(num_steps, batch_size, 2, device=device)
    t_echo = t_pulse + delay
    if 0 <= t_pulse < num_steps:
        spikes[t_pulse, :, 0] = 1.0  # FM1
    if 0 <= t_echo < num_steps:
        spikes[t_echo, :, 1] = 1.0   # FM3
    print(spikes)
    return spikes


# -------------------------------
# Spiking readout that triggers global inhibition (WTA)
# -------------------------------
class SpikingWTAReadout(nn.Module):
    """
    Readout neuron integrates spikes from the bank.
    When it spikes, we activate a global inhibition flag to suppress
    *future* spikes from the bank, yielding a clean winner.
    """
    def __init__(self, num_inputs, beta=0.9, threshold=1.0):
        super().__init__()
        self.readout = snn.Leaky(beta=beta, threshold=threshold)

        # Excitatory weights from bank spikes -> readout current
        self.fc_exc = nn.Linear(num_inputs, 1, bias=False)
        with torch.no_grad():
            self.fc_exc.weight[:] = torch.ones(1, num_inputs)

    def init_state(self, device="cpu"):
        mem = self.readout.init_leaky()
        if isinstance(mem, torch.Tensor):
            mem = mem.to(device)
        return mem

    def step(self, bank_spk_t, mem):
        """
        bank_spk_t: [B,N] spikes at time t from bank
        returns: ro_spk [B,1], ro_mem_next
        """
        cur = self.fc_exc(bank_spk_t)     # [B,1]
        spk, mem = self.readout(cur, mem)
        return spk, mem


# -------------------------------
# Full network: bank + spiking WTA readout + inhibition
# -------------------------------
class FMFM_WTA_Network(nn.Module):
    def __init__(
        self,
        delays,
        chosen_w1=0.7,
        ute=1.01,
        beta=0.9,
        threshold_bank=1.0,
        beta_readout=0.9,
        threshold_readout=1.0,
    ):
        super().__init__()
        self.delays = np.array(delays, dtype=int)

        self.bank = nn.ModuleList([
            FMFMNeuron(delay=d, chosen_w1=chosen_w1, ute=ute, beta=beta, threshold=threshold_bank)
            for d in self.delays
        ])

        self.readout = SpikingWTAReadout(
            num_inputs=len(self.delays),
            beta=beta_readout,
            threshold=threshold_readout,
        )


    @torch.no_grad()
    def forward(self, spike_seq):
        T, B, _ = spike_seq.shape
        device = spike_seq.device
        N = len(self.bank)

        bank_mem = [neuron.init_state(B, device=device) for neuron in self.bank]
        ro_mem = self.readout.init_state(device=device)

        bank_spk_rec = []
        ro_spk_rec = []

        inhibit = False
        inhibit_time = None
        winner_idx = None

        for t in range(T):
            x_t = spike_seq[t]
            spk_t_list = []
            
            # 1. Run all neurons first
            active_indices_at_t = [] # Track everyone who spiked this step
            
            for i, neuron in enumerate(self.bank):
                spk_i, bank_mem[i] = neuron.step(x_t, bank_mem[i])
                
                if inhibit:
                    spk_i = torch.zeros_like(spk_i)
                
                if spk_i.sum() > 0:
                    active_indices_at_t.append(i)
                    
                spk_t_list.append(spk_i)

            bank_spk_t = torch.cat(spk_t_list, dim=1)
            bank_spk_rec.append(bank_spk_t)

            # 2. Check for winner among the active indices
            if (not inhibit) and (winner_idx is None) and (len(active_indices_at_t) > 0):
                # We have spikes! But which one is the true winner?
                # We need the one associated with the SMALLEST delay value.
                
                # Get the delay values for all neurons that spiked
                spiking_delays = [self.delays[i] for i in active_indices_at_t]
                
                # Find the minimum delay (this avoids the "early arrival" problem)
                min_delay = min(spiking_delays)
                
                # Find which index that delay belongs to
                # Note: np.where returns a tuple, we take [0][0]
                true_winner_local_idx = np.where(self.delays == min_delay)[0][0]
                
                winner_idx = true_winner_local_idx

            # 3. Readout / Inhibition logic
            ro_spk_t, ro_mem = self.readout.step(bank_spk_t, ro_mem)
            ro_spk_rec.append(ro_spk_t.squeeze(-1))

            if (not inhibit) and (ro_spk_t.sum() > 0):
                inhibit = True
                inhibit_time = t

        bank_spk = torch.stack(bank_spk_rec, dim=0)
        readout_spk = torch.stack(ro_spk_rec, dim=0)

        return bank_spk, readout_spk, inhibit_time, winner_idx


    @torch.no_grad()
    def decode(self, spike_seq):
        bank_spk, readout_spk, inhibit_time, winner_idx = self.forward(spike_seq)

        if winner_idx is None:
            return None, bank_spk, readout_spk, inhibit_time, winner_idx

        pred_delay = int(self.delays[winner_idx])
        return pred_delay, bank_spk, readout_spk, inhibit_time, winner_idx


# -------------------------------
# Demo / test
# -------------------------------
if __name__ == "__main__":
    device = "cpu"

    candidate_delays = np.arange(1, 11)  # bank delays 1..10
    net = FMFM_WTA_Network(
        delays=candidate_delays,
        chosen_w1=0.7,
        ute=1.01,
        beta=0.9,
        threshold_bank=1.0,
        beta_readout=0.9,
        threshold_readout=1.0,
    ).to(device)

    unknown_delay = 5
    spike_seq = generate_pulse_echo(num_steps=50, delay=unknown_delay, batch_size=1, device=device)

    pred_delay, bank_spk, readout_spk, inhibit_time, winner_idx = net.decode(spike_seq)

    print(f"True delay: {unknown_delay}")
    print(f"Predicted delay: {pred_delay}")
    print(f"Inhibition triggered at t = {inhibit_time}")
    if winner_idx is not None:
        print(f"Winner index: {winner_idx}, winner tuned delay: {candidate_delays[winner_idx]}")


    # fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # # Plot readout spike train
    # # plt.figure(figsize=(8, 2.5))
    # ax[0].plot(readout_spk[:, 0].cpu().numpy(), marker="o")
    # ax[0].set_title("Readout Neuron")
    # ax[0].x_label("Time step")
    # ax[0].y_label("Spike")
    # ax[0].xlim(0,20)
    # ax[0].grid(True)

    # # Plot bank raster (T x N)
    # # ax[1].figure(figsize=(9, 4))
    # ax[1].imshow(bank_spk[:, 0, :].cpu().numpy(), aspect="auto", origin="lower", interpolation="nearest")
    # ax[1].set_title("Spike Bank")
    # ax[1].xlabel("Tuned delay")
    # ax[1].ylabel("Time step")
    # ax[1].xticks(np.arange(len(candidate_delays)), candidate_delays)
    # ax[1].colorbar(label="Spike")
    # plt.show()


    fig, ax = plt.subplots(1, 2, figsize=(12, 5)) # Removed sharey=True

    # 1. Plot readout spike train
    # ax.plot expects x and y data. If passing only one array, it plots against index.
    ax[0].plot(readout_spk[:, 0].cpu().numpy(), marker="o")
    ax[0].set_title("Readout Neuron")
    ax[0].set_xlabel("Time step")     # Fixed: .xlabel -> .set_xlabel
    ax[0].set_ylabel("Spike")         # Fixed: .ylabel -> .set_ylabel
    ax[0].set_xlim(0, 20)             # Fixed: .xlim -> .set_xlim
    ax[0].grid(True)

    # 2. Plot bank raster
    # Capture the image object (im) to make the colorbar
    im = ax[1].imshow(
        bank_spk[:, 0, :].cpu().numpy(), 
        aspect="auto", 
        origin="lower", 
        interpolation="nearest"
    )

    ax[1].set_title("Spike Bank")
    ax[1].set_xlabel("Tuned delay")   # Fixed: .xlabel -> .set_xlabel
    ax[1].set_ylabel("Time step")     # Fixed: .ylabel -> .set_ylabel
    ax[1].set_ylim(0, 20) 


    # Fixed: xticks requires two steps in OO-interface
    ax[1].set_xticks(np.arange(len(candidate_delays)))
    ax[1].set_xticklabels(candidate_delays)

    # Fixed: Add colorbar using the image object
    fig.colorbar(im, ax=ax[1], label="Spike") 

    plt.tight_layout()
    plt.show()