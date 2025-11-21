% Parameters
c = 3e8;                        % Speed of light (m/s)
fc = 77e9;                      % Carrier frequency (Hz)
lambda = c / fc;                % Wavelength (m)

% Chirp parameters
bw = 150e6;                     % Bandwidth
T_chirp = 10e-6;                % Chirp duration (s)
k = bw / T_chirp;               % Chirp rate (slope)
fs = 2 * bw;                    % Sample rate (Hz)

% Frame parameters for velocity 
n_chirps = 64;                  % Number of chirps in a frame
T_frame = T_chirp * n_chirps;   % Total frame time

% Generate Tx Time Vector
t = 0 : 1/fs : T_chirp - 1/fs;      % Time vector for ONE chirp
n_samples = length(t);              % Number of samples per chirp

% FFT parameters (calculated once)
n_fft_range = 2^nextpow2(n_samples); 
n_fft_vel = 2^nextpow2(n_chirps);  

% Create Axes for Plotting (calculated once)
f_beat_axis = fs * (0:(n_fft_range/2)) / n_fft_range;
range_axis = (f_beat_axis * c) / (2 * k);
fs_doppler = 1 / T_chirp; 
doppler_freq_axis = linspace(-fs_doppler/2, fs_doppler/2, n_fft_vel);
vel_axis = (doppler_freq_axis * lambda) / 2;

% --- NEW: Dataset Generation Parameters ---
N_samples = 100;        % Number of RDM/label pairs to generate
max_targets = 3;        % Maximum number of targets per scene

% Randomization bounds
min_range = 10;         % m
max_range = 250;        % m
max_vel = 50;           % m/s (allows for -50 to +50 m/s)
min_rcs = 0.5;          % m^2
max_rcs = 20;           % m^2
min_snr = 10;            % dB
max_snr = 15;           % dB

% --- NEW: Initialize Data Storage ---
% X_data: The RDM images. We store the log-magnitude of the first half
% of the range FFT. Dimensions: (Velocity Bins, Range Bins, Samples)
rdm_dataset = zeros(n_fft_vel, n_fft_range/2 + 1, N_samples);

% Y_data: The labels. Dimensions: (Samples, Max Targets, 2)
% The last dimension stores [range, velocity]
label_dataset = zeros(N_samples, max_targets, 2);

disp(['--- Starting Dataset Generation (', num2str(N_samples), ' samples) ---']);

% --- NEW: Main Generation Loop ---
for k = 1:N_samples
    
    % --- NEW: Randomize Scenario ---
    n_targets = randi([1, max_targets]);
    target_range = min_range + (max_range - min_range) * rand(1, n_targets);
    target_vel = -max_vel + (2 * max_vel) * rand(1, n_targets);
    target_rcs = min_rcs + (max_rcs - min_rcs) * rand(1, n_targets);
    SNR_dB = min_snr + (max_snr - min_snr) * rand;

    % --- NEW: Store Labels ---
    % Pad with zeros if n_targets < max_targets
    current_labels = zeros(max_targets, 2);
    current_labels(1:n_targets, 1) = target_range;
    current_labels(1:n_targets, 2) = target_vel;
    label_dataset(k, :, :) = current_labels;

    % Generate Tx
    tx_phase = 2 * pi * (fc * t + (k/2) * t.^2);
    Tx = exp(1j * tx_phase);
    
    % Simulate Chirp Frame
    Mix_matrix = zeros(n_chirps, n_samples);
    for i = 1:n_chirps % Loop for each chirp
        Rx_total = zeros(1, n_samples); 
        time_at_chirp_start = (i-1) * T_chirp;
        
        % iterate through each target
        for j = 1:n_targets
            
            % Get this target's properties
            r0 = target_range(j);
            v0 = target_vel(j);
            
            % Calculate this target's current range
            current_range = r0 + v0 * time_at_chirp_start;
            
            % Calculate time delay for this target
            tau = (2 * current_range) / c;
            
            % Create the delayed time vector
            t_rx = t - tau;
            
            % Generate this target's reflected signal phase
            rx_phase = 2 * pi * (fc * t_rx + (k/2) * t_rx.^2);
            Rx_j = exp(1j * rx_phase);
            
            % Apply attenuation based on range and RCS
            att = sqrt(target_rcs(j)) / (current_range^2);
            
            % Add this target's signal to the total
            Rx_total = Rx_total + (att * Rx_j);
        end
        
        % Add noise
        Rx_noisy = awgn(Rx_total, SNR_dB, 'measured');
        
        % Mix the total received signal with the transmitted
        Mix_matrix(i, :) = Tx .* conj(Rx_noisy);
    end
    
    % Process the Signal (2D FFT)
    range_fft = fft(Mix_matrix, n_fft_range, 2); 
    rdm = fftshift(fft(range_fft, n_fft_vel, 1), 1);
    
    % Get the absolute value of the useful half of the RDM
    rdm_abs = abs(rdm(:, 1:n_fft_range/2+1));
    
    % Convert to dB, adding epsilon to avoid log(0)
    rdm_db = 10*log10(rdm_abs);
    
    % Normalize the RDM
    % Clips the noise floor 60dB below the peak.
    % peak_val = max(rdm_db, [], 'all');
    % rdm_db_clipped = max(rdm_db, peak_val - 60);
    
    % Store the final processed RDM
    rdm_dataset(:, :, k) = rdm_db;

    % Progress update
    if mod(k, 100) == 0
        disp(['Generated sample ', num2str(k), ' of ', num2str(N_samples)]);
    end
end

disp('--- Dataset Generation Complete ---');

% --- NEW: Save the dataset to a file ---
disp('Saving dataset to radar_dataset.mat...');
save('radar_dataset_rdm.mat', 'rdm_dataset', 'label_dataset', 'range_axis', 'vel_axis', '-v7.3');
disp('Done.');

% --- NEW: Visualize one example from the dataset ---
figure;
example_idx = randi(N_samples);
imagesc(range_axis, vel_axis, rdm_dataset(:, :, example_idx));
title(['Example RDM (Sample #', num2str(example_idx), ')']);
xlabel('Range (m)');
ylabel('Velocity (m/s)');
colorbar;
axis xy; 
xlim([0, max_range]);

% Display the labels for this example
disp(['Labels for sample #', num2str(example_idx), ':']);
disp(squeeze(label_dataset(example_idx, :, :)));