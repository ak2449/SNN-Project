
% Base Radar Parameters (Constant for the whole dataset)
c = 3e8;                        % Speed of light (m/s)
fc = 77e9;                      % Carrier frequency (Hz)
lambda = c / fc;                % Wavelength (m)

% Chirp parameters
bw = 150e6;                     % Bandwidth (Hz)
T_chirp = 10e-6;                % Chirp duration (s)
k = bw / T_chirp;               % Chirp rate (slope)
fs = 2 * bw;                    % Sample rate (Hz)

% Frame parameters
n_chirps = 64;                  % Number of chirps in a frame
T_frame = T_chirp * n_chirps;   % Total frame time

% Time vector for ONE chirp
t = 0 : 1/fs : T_chirp - 1/fs;      % Time vector for ONE chirp
n_samples = length(t);              % Number of samples per chirp

% Generate the constant Tx signal (only need to do this once)
tx_phase = 2 * pi * (fc * t + (k/2) * t.^2);
Tx = exp(1j * tx_phase);

% Dataset Generation Parameters (Customize for your training)
n_dataset_samples = 100;      

% Scene variability
max_targets = 3;              % Maximum number of targets in any given scene
range_limits = [10, 250];     % [min, max] initial range (m)
vel_limits = [-30, 30];       % [min, max] velocity (m/s) (negative is approaching)
rcs_limits = [0.1, 20];       % [min, max] Radar Cross Section
SNR_dB = 15;                  % Signal-to-Noise Ratio (dB) for awgn

% RDM Processing Parameters
% Use the same dimensions for FFT as the signal
N_fft_range = n_samples;
N_fft_doppler = n_chirps;

% Create windows for sidelobe reduction
win_range = hann(n_samples);
win_doppler = hann(n_chirps);
% Combine into a 2D window (n_chirps x n_samples)
win_2d = win_doppler * win_range'; 


% Pre-allocate Data Arrays (for speed)
% RDMs are real-valued (after abs). Using 'single' saves memory.
dataset_data = zeros(n_dataset_samples, n_chirps, n_samples, 'single');

% dataset_labels will be (n_dataset_samples, max_targets)
% We pad with zeros if n_targets < max_targets
dataset_labels_range = zeros(n_dataset_samples, max_targets);
dataset_labels_velocity = zeros(n_dataset_samples, max_targets);

fprintf('Starting RDM dataset generation for %d samples...\n', n_dataset_samples);

% Main Generation Loop
for k = 1:n_dataset_samples
    
    % --- 1. Generate Ground Truth for this sample ---
    n_targets = randi([1, max_targets]);
    r_targets = range_limits(1) + (range_limits(2) - range_limits(1)) * rand(1, n_targets);
    v_targets = vel_limits(1) + (vel_limits(2) - vel_limits(1)) * rand(1, n_targets);
    rcs_targets = rcs_limits(1) + (rcs_limits(2) - rcs_limits(1)) * rand(1, n_targets);
    
    % Store Ground Truth Labels (with padding)
    dataset_labels_range(k, 1:n_targets) = r_targets;
    dataset_labels_velocity(k, 1:n_targets) = v_targets;
    
    % Simulate the Radar Frame (Mix_matrix) for these targets 
    Mix_matrix = zeros(n_chirps, n_samples, 'like', 1j);
    
    for i = 1:n_chirps % Loop for each chirp
        
        Rx_total = zeros(1, n_samples); 
        time_at_chirp_start = (i-1) * T_chirp;
        
        for j = 1:n_targets
            r0 = r_targets(j);
            v0 = v_targets(j);
            current_range = r0 + v0 * time_at_chirp_start;
            tau = (2 * current_range) / c;
            t_rx = t - tau;
            rx_phase = 2 * pi * (fc * t_rx + (k/2) * t_rx.^2);
            Rx_j = exp(1j * rx_phase);
            att = sqrt(rcs_targets(j)) / (current_range^2); 
            Rx_total = Rx_total + (att * Rx_j);
        end
        
        Rx_noisy = awgn(Rx_total, SNR_dB, 'measured');
        Mix_matrix(i, :) = Tx .* conj(Rx_noisy);
    end
    
    % NEW: Convert Mix_matrix to Normalized RDM
    
    
    % Apply 2D window
    Mix_windowed = Mix_matrix .* win_2d;
    
    % Range-FFT (along dimension 2)
    range_fft = fft(Mix_windowed, N_fft_range, 2);
    
    % Doppler-FFT (along dimension 1)
    rdm = fft(range_fft, N_fft_doppler, 1);
    
    % Shift and take magnitude
    rdm_shifted = fftshift(rdm);
    rdm_abs = abs(rdm_shifted);
    
    % Normalize the RDM to [0, 1] for the SNN
    % (Adding 1e-6 to prevent divide-by-zero if RDM is all zeros)
    rdm_min = min(rdm_abs(:));
    rdm_max = max(rdm_abs(:));
    rdm_normalized = (rdm_abs - rdm_min) / (rdm_max - rdm_min + 1e-6);

    % Store the final RDM in the dataset array 
    dataset_data(k, :, :) = single(rdm_normalized); % Store as single precision
    
    % Report progress ---
    if mod(k, 100) == 0
        fprintf('... generated sample %d of %d\n', k, n_dataset_samples);
    end
    
end

fprintf('Dataset generation complete!\n');
fprintf('Variables created:\n');
fprintf('  - dataset_data:         [%d, %d, %d] (Output RDMs, real/normalized)\n', size(dataset_data));
fprintf('  - dataset_labels_range:   [%d, %d] (Labels, real)\n', size(dataset_labels_range));
fprintf('  - dataset_labels_velocity: [%d, %d] (Labels, real)\n', size(dataset_labels_velocity));


% Visualize one example from the dataset 
figure;
example_idx = randi(n_dataset_samples);
imagesc(range_axis, vel_axis, dataset_data(:, :, example_idx));
title(['Example RDM (Sample #', num2str(example_idx), ')']);
xlabel('Range (m)');
ylabel('Velocity (m/s)');
colorbar;
axis xy; 
xlim([0, max_range]);

% Display the labels for this example
disp(['Labels for sample #', num2str(example_idx), ':']);
disp(squeeze(label_dataset(example_idx, :, :)));

% Save the dataset to a file
% save('radar_rdm_dataset.mat', 'dataset_data', 'dataset_labels_range', 'dataset_labels_velocity', '-v7.3');
% disp('Dataset saved to radar_rdm_dataset.mat');