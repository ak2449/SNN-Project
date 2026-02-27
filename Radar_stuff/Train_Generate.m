rng('default'); % For reproducible results

% Base Radar Parameters
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

% Generate the constant Tx signal
tx_phase = 2 * pi * (fc * t + (k/2) * t.^2);
Tx = exp(1j * tx_phase);

% Dataset Generation Parameters
n_dataset_samples = 100;      

% Scene variability
max_targets = 3;              % Maximum number of targets in any given scene
range_limits = [10, 250];     % [min, max] initial range (m)
vel_limits = [-30, 30];       % [min, max] velocity (m/s) (negative is approaching)
rcs_limits = [0.1, 20];       % [min, max] Radar Cross Section
SNR_dB = 15;                  % Signal-to-Noise Ratio (dB) for awgn

% Pre-allocate Data Arrays (for speed)
% dataset_data will be (n_dataset_samples, n_chirps, n_samples)
dataset_data = zeros(n_dataset_samples, n_chirps, n_samples, 'like', 1j);

% dataset_labels will be (n_dataset_samples, max_targets)
% We pad with zeros if n_targets < max_targets
dataset_labels_range = zeros(n_dataset_samples, max_targets);
dataset_labels_velocity = zeros(n_dataset_samples, max_targets);

fprintf('Starting dataset generation for %d samples...\n', n_dataset_samples);

% Main Generation Loop
for k = 1:n_dataset_samples
    
    % --- 1. Generate Ground Truth for this sample ---
    
    % Randomly decide how many targets for this scene (1 to max_targets)
    n_targets = randi([1, max_targets]);
    
    % Generate random properties for these targets
    r_targets = range_limits(1) + (range_limits(2) - range_limits(1)) * rand(1, n_targets);
    v_targets = vel_limits(1) + (vel_limits(2) - vel_limits(1)) * rand(1, n_targets);
    rcs_targets = rcs_limits(1) + (rcs_limits(2) - rcs_limits(1)) * rand(1, n_targets);
    
    % --- 2. Store Ground Truth Labels (with padding) ---
    dataset_labels_range(k, 1:n_targets) = r_targets;
    dataset_labels_velocity(k, 1:n_targets) = v_targets;
    
    % --- 3. Simulate the Radar Frame (Mix_matrix) for these targets ---
    Mix_matrix = zeros(n_chirps, n_samples);
    
    for i = 1:n_chirps % Loop for each chirp
        
        Rx_total = zeros(1, n_samples); 
        time_at_chirp_start = (i-1) * T_chirp;
        
        % Iterate through each generated target
        for j = 1:n_targets
            
            % Get this target's properties
            r0 = r_targets(j);
            v0 = v_targets(j);
            
            % Calculate this target's current range at the start of this chirp
            current_range = r0 + v0 * time_at_chirp_start;
            
            % Calculate time delay for this target
            tau = (2 * current_range) / c;
            
            % Create the delayed time vector
            t_rx = t - tau;
            
            % Generate this target's reflected signal phase
            rx_phase = 2 * pi * (fc * t_rx + (k/2) * t_rx.^2);
            Rx_j = exp(1j * rx_phase);
            
            % Apply attenuation (Amplitude ~ sqrt(RCS)/Range^2)
            att = sqrt(rcs_targets(j)) / (current_range^2); 
            
            % Add this target's signal to the total
            Rx_total = Rx_total + (att * Rx_j);
        end
        
        % Add noise
        Rx_noisy = awgn(Rx_total, SNR_dB, 'measured');
        
        % Mix the total received signal with the transmitted
        Mix_matrix(i, :) = Tx .* conj(Rx_noisy);
    end
    
    % Store the final Mix_matrix in the 3D dataset array 
    dataset_data(k, :, :) = Mix_matrix;
    
    % Progress
    if mod(k, 100) == 0
        fprintf('... generated sample %d of %d\n', k, n_dataset_samples);
    end
    
end

fprintf('Dataset generation complete!\n');
fprintf('Variables created:\n');
fprintf('  - dataset_data:         [%d, %d, %d] (Input Data, complex)\n', size(dataset_data));
fprintf('  - dataset_labels_range:   [%d, %d] (Labels, real)\n', size(dataset_labels_range));
fprintf('  - dataset_labels_velocity: [%d, %d] (Labels, real)\n', size(dataset_labels_velocity));

save('radar_training_dataset.mat', 'dataset_data', 'dataset_labels_range', 'dataset_labels_velocity', '-v7.3');
disp('Dataset saved to radar_training_dataset.mat');