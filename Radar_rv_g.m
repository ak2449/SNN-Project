% --- 1. Define Parameters ---
c = 3e8;                        % Speed of light (m/s)
fc = 77e9;                      % Carrier frequency (Hz) (Crucial for Doppler)
lambda = c / fc;                % Wavelength (m)

% Chirp parameters
bw = 15e6;                     % Chirp bandwidth (Hz)
T_chirp = 10e-6;                % Chirp duration (s)
k = bw / T_chirp;               % Chirp rate (slope)
fs = 2 * bw;                    % Sample rate (Hz)

% --- Frame parameters for velocity ---
n_chirps = 64;                  % Number of chirps in a frame (determines velocity resolution)
T_frame = T_chirp * n_chirps;   % Total frame time (we'll assume no dead time for simplicity)

% Scenario parameters
target_range = 200;            % Target's initial range (m)
target_vel = 40;               % Target's velocity (m/s) (Negative = approaching)

% --- 2. Generate Transmitted Signal (Tx) ---
t = 0 : 1/fs : T_chirp - 1/fs;      % Time vector for ONE chirp
n_samples = length(t);              % Number of samples per chirp

% Generate the phase for one Tx chirp (we can reuse this)
tx_phase = 2 * pi * (fc * t + (k/2) * t.^2);
Tx = exp(1j * tx_phase);


% --- 3. Simulate Chirp Frame ---
% --- Create a matrix to hold all mixed signals ---
% Rows = chirps, Columns = time samples
Mix_matrix = zeros(n_chirps, n_samples);

for i = 1:n_chirps
    % Calculate the time at the start of this chirp
    time_at_chirp_start = (i-1) * T_chirp;
    
    % Update the target's range based on its velocity
    % We use the range at the *start* of the chirp for this basic simulation
    current_range = target_range + target_vel * time_at_chirp_start;
    
    % Calculate round-trip time delay for this chirp
    tau = (2 * current_range) / c;
    
    % Create the delayed time vector
    t_rx = t - tau;
    
    % Generate the received signal (Rx) for this chirp
    rx_phase = 2 * pi * (fc * t_rx + (k/2) * t_rx.^2);
    Rx = exp(1j * rx_phase);

    % Add noise
    att = 1 / (target_range^2);
    Rx_noisy = att*awgn(Rx, 10,'measured');
    
    % Mix Tx and Rx to get the beat signal for this chirp
    Mix_matrix(i, :) = Tx .* conj(Rx_noisy);
end

% --- 4. Process the Signal (2D FFT) ---

% --- 4a. 1st FFT (Range FFT) ---
n_fft_range = 2^nextpow2(n_samples); % FFT points for range
range_fft = fft(Mix_matrix, n_fft_range, 2); % FFT along dimension 2 (rows)

% --- 4b. 2nd FFT (Doppler FFT) ---
n_fft_vel = 2^nextpow2(n_chirps);  % FFT points for velocity
% Apply FFT along dimension 1 (columns) and shift for centered Doppler
rdm = fftshift(fft(range_fft, n_fft_vel, 1), 1);

% --- 5. Create Axes for Plotting ---

% Range axis (same as before)
f_beat_axis = fs * (0:(n_fft_range/2)) / n_fft_range;
range_axis = (f_beat_axis * c) / (2 * k);

% --- NEW: Velocity axis ---
% The sampling frequency in the "Doppler dimension" is 1/T_chirp
fs_doppler = 1 / T_chirp; 
% The FFT-shifted frequency axis for Doppler
doppler_freq_axis = linspace(-fs_doppler/2, fs_doppler/2, n_fft_vel);

% Convert Doppler frequency to velocity
% v = f_doppler * lambda / 2
vel_axis = (doppler_freq_axis * lambda) / 2;

% --- 6. Visualize the Range-Doppler Map (RDM) ---
figure;

% We only need to plot the first half of the range FFT results
rdm_plot = abs(rdm(:, 1:n_fft_range/2+1));

% Plot in dB
imagesc(range_axis, vel_axis, 10*log10(rdm_plot));
title('Range-Doppler Map (RDM)');
xlabel('Range (m)');
ylabel('Velocity (m/s)');
colorbar;
axis xy; % Puts 0-velocity at the bottom, which is conventional

% --- 7. Find and Print the Peak ---
[max_val, max_idx] = max(rdm_plot(:));
[vel_idx, range_idx] = ind2sub(size(rdm_plot), max_idx);

fprintf('Detected target at:\n');
fprintf('  Range:    %.2f m\n', range_axis(range_idx));
fprintf('  Velocity: %.2f m/s\n', vel_axis(vel_idx));