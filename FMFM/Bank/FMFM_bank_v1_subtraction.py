import torch
import numpy as np
import matplotlib.pyplot as plt
import snntorch as snn
import torch.nn as nn

# -------------------------------
# Define FMFMNeuron with delay-based weight initialization
# -------------------------------
class FMFMNeuron(nn.Module):
    def __init__(self,delay,chosen_w1,ute = 1.01, beta=0.95):
        super().__init__()

        self.beta = beta     # Decay factor
        self.chosen_w1 = chosen_w1  # Chosen w1 for initialization
        self.ute = ute       # Membrane potential at echo time
        
        self.w1, self.w2 = self.calculate_weights(delay)

        # 2 input channels: FM1, FM3
        self.fc = nn.Linear(2, 1, bias=False)

        # initialise weights so FM1 is a "priming" current,
        # FM3 is a strong trigger
        with torch.no_grad():
            self.fc.weight[:] = torch.tensor([self.w1, self.w2])  # [w_FM1, w_FM3]y

        # Leaky integrate-and-fire neuron
        self.lif = snn.Leaky(beta=beta)

    def forward(self, spike_seq):
        """
        spike_seq: [num_steps, batch_size, 2]
        Returns:
            spk_rec: [num_steps, batch_size, 1]
        """
        num_steps, batch_size, _ = spike_seq.shape
        mem = self.lif.init_leaky()

        spk_rec = []
        for t in range(num_steps):
            cur = self.fc(spike_seq[t])       # current from FM1 + FM3
            spk, mem = self.lif(cur, mem)     # LIF dynamics
            spk_rec.append(spk)

        spk_rec = torch.stack(spk_rec, dim=0)
        return spk_rec

    def calculate_weights(self, delay):
        """
        Calculate the weights w1 and w2 based on the given delay.
        This follows the manual method described in the explanation.
        """
        # Calculate w1: We choose w1 to be less than Uth, for simplicity, we choose w1 = chosen_w1
        w1 = self.chosen_w1
        
        # Calculate w2 using the equation Ute = beta * w1^d + w2
        # We want Ute > Uth, so we solve for w2
        w2 = self.ute - (self.beta**delay) * (w1)
        print(w1, w2)
        return w1, w2



# -------------------------------
# Generate pulse sequence (for testing purposes)
# -------------------------------
def generate_pulse_echo(num_steps=50, delay=10, batch_size=1):
    spikes = torch.zeros(num_steps, batch_size, 2)

    t_pulse = 5
    t_echo = t_pulse + delay
    if t_echo < num_steps:
        spikes[t_pulse, :, 0] = 1.0  # FM1 channel
        spikes[t_echo, :, 1] = 1.0   # FM3 channel

    return spikes


# -------------------------------
# Define FMFMNeuronBank (ensemble of neurons with different delays)
# -------------------------------
class FMFMNeuronBank:
    def __init__(self, beta=0.9, delays=np.linspace(1, 10, 10, dtype=int)):
        # Create a list of neurons with different delays and weights
        self.neurons = [FMFMNeuron(delay=target_delay, beta=beta,chosen_w1 = 0.7) for delay in delays]
        self.delays = delays

    def get_output_for_delay(self, target_delay):
        """
        Subtract neuron outputs to isolate the spikes corresponding to the target delay
        """
        # Get spike outputs for each neuron
        spike_outputs = []
        for i, delay in enumerate(self.delays):
            spike_seq = generate_pulse_echo(delay=delay)  # Generate spikes with the given delay
            spike_output = self.neurons[i](spike_seq)    # Get the spike output for the delay
            spike_outputs.append(spike_output)

        # Now we want to subtract outputs of all neurons except the one with the target delay
        # Find the index where the delay matches the target delay
        target_index = np.where(self.delays == target_delay)[0][0]
        target_output = spike_outputs[target_index]  # Get the output for the target delay
        other_outputs = [spike_outputs[i] for i, delay in enumerate(self.delays) if delay != target_delay]

        # Subtract all the other outputs from the target output to isolate spikes at the target delay
        for other_output in other_outputs:
            target_output -= other_output  # Subtract other outputs

        return target_output


# -------------------------------
# Testing: Isolate the spikes for target_delay = 5
# -------------------------------
device = 'cpu'
target_delay = 4
delays = np.linspace(1, 10, 10, dtype=int)

# Create the neuron bank with delays
neuron_bank = FMFMNeuronBank(beta=0.9, delays=delays)

# Get output for target delay
spk_output_target_delay = neuron_bank.get_output_for_delay(target_delay=target_delay)

# Convert to numpy for plotting
spk_output_target_delay = spk_output_target_delay.detach().cpu().numpy()

# Plot the results
plt.figure(figsize=(10, 6))
plt.imshow(spk_output_target_delay[:, 0, :], aspect='auto', cmap='binary', origin='lower', interpolation='nearest')
plt.title(f"Spike Output for Target Delay = {target_delay}")
plt.ylim(0,20)
plt.xlabel('Batch Size')
plt.ylabel('Time Steps')
plt.colorbar(label="Spike Intensity")
plt.show()

