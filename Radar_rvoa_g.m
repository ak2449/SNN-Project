
% --- 1. Define Parameters ---
c = 3e8;                        % Speed of light (m/s)
fc = 77e9;                      % Carrier frequency (Hz)
lambda = c / fc;                % Wavelength (m)

% --- NEW: Array Parameters ---
n_antennas = 16;                % Number of receiver antennas
d = lambda / 2;                 % Antenna spacing (Nyquist)

% Chirp parameters
bw = 15e6;                      % Bandwidth (15 MHz)
T_chirp = 10e-6;                % Chirp duration (s)
k = bw / T_chirp;               % Chirp rate (slope)
fs = 2 * bw;                    % Sample rate (Hz)

% Frame parameters
n_chirps = 64;                  % Number of chirps in a frame
n_samples = T_chirp * fs;       % Number of samples per chirp

% Scenario parameters
% <-- MODIFIED: Added angle to targets -->
target_range = [100, 75,  50];  % Targets' initial ranges (m)
target_vel   = [40,  -15, 10];  % Targets' velocities (m/s)
target_angle = [-20, 10,  30];  % <-- NEW: Target angles (degrees)
target_rcs   = [1,   0.8, 0.5]; % Relative "brightness"
n_targets    = length(target_range);

% --- 2. Generate Transmitted Signal (Tx) ---
t = (0 : n_samples - 1) / fs;   % Time vector for ONE chirp
tx_phase = 2 * pi * (fc * t + (k/2) * t.^2);
Tx = exp(1j * tx_phase);

% --- 3. Simulate Chirp Frame ---
% <-- MODIFIED: Matrix is now 3D (Chirps, Samples, Antennas) -->
Mix_matrix = zeros(n_chirps, n_samples, n_antennas);

for i = 1:n_chirps % Loop for each chirp
    
    % This will hold the sum of all reflections for this one chirp
    Rx_chirp_total = zeros(n_samples, n_antennas); 
    
    time_at_chirp_start = (i-1) * T_chirp;

    for j = 1:n_targets % Loop for each target
        
        % Get this target's properties
        r0 = target_range(j);
        v0 = target_vel(j);
        theta_deg = target_angle(j);
        theta_rad = deg2rad(theta_deg); % Convert angle to radians
        
        % Calculate this target's current range
        current_range = r0 + v0 * time_at_chirp_start;
        
        % Calculate time delay for the *first* antenna (reference)
        tau = (2 * current_range) / c;
        
        % Generate this target's reflected signal
        t_rx = t - tau;
        rx_phase = 2 * pi * (fc * t_rx + (k/2) * t_rx.^2);
        Rx_j = exp(1j * rx_phase);
        
        % Apply attenuation
        att = target_rcs(j) / (current_range^4);
        
        % <-- NEW: Calculate and apply phase shift for each antenna -->
        for ant_idx = 1:n_antennas
            % Calculate the extra path length for this antenna
            % This is the simple ULA phase shift model
            path_diff = (ant_idx - 1) * d * sin(theta_rad);
            
            % Calculate the phase shift (at carrier frequency)
            % This is the round-trip phase shift (2 * path / lambda)
            spatial_phase_shift = exp(1j * 2 * pi * path_diff * 2 / lambda);
            
            % Add this target's signal to this antenna's total
            Rx_chirp_total(:, ant_idx) = Rx_chirp_total(:, ant_idx) + (att * Rx_j.' * spatial_phase_shift);
        end
    end
    
    % Add noise ONCE to the combined signal for all antennas
    Rx_noisy = awgn(Rx_chirp_total, -10, 'measured');
    
    % Mix the total received signal with the reference
    % We use conj(Tx) here so positive velocity = positive Doppler
    Mix_matrix(i, :, :) = Tx .* conj(Rx_noisy);
end

% --- 4. Process the Signal (3D FFT) ---
n_fft_range = 2^nextpow2(n_samples);
n_fft_vel = 2^nextpow2(n_chirps);
n_fft_angle = 2^nextpow2(n_antennas);

% 1st FFT (Range)
range_fft = fft(Mix_matrix, n_fft_range, 2);
% 2nd FFT (Doppler)
rd_map = fft(range_fft, n_fft_vel, 1);
% 3rd FFT (Angle) - and fftshift to center 0 degrees
rda_cube = fftshift(fft(rd_map, n_fft_angle, 3), 3);


% --- 5. Create Axes for Plotting ---
% Range axis
f_beat_axis = fs * (0:(n_fft_range-1)/2) / n_fft_range;
range_axis = (f_beat_axis * c) / (2 * k);

% Velocity axis
fs_doppler = 1 / T_chirp; 
doppler_freq_axis = linspace(-fs_doppler/2, fs_doppler/2, n_fft_vel);
vel_axis = (doppler_freq_axis * lambda) / 2;

% --- NEW: Angle axis ---
% The FFT axis is in "normalized angular frequency" u = 2*pi*d*sin(theta)/lambda
% Since d = lambda/2, u = pi*sin(theta)
% We fftshifted, so u goes from -pi to pi.
% This means sin(theta) goes from -1 to 1.
angle_axis_rad = asin(linspace(-1, 1, n_fft_angle));
angle_axis_deg = rad2deg(angle_axis_rad);


% --- 6. Visualize the Range-Doppler-Angle Cube ---
% A 3D cube is hard to plot. Let's show a 2D slice:
% A "Range-Angle" map, showing a top-down view.
% We "collapse" the Doppler dimension by taking the max value.

% Take the magnitude
rdm_plot = abs(rda_cube(:, 1:n_fft_range/2, :));
range_axis = range_axis(1:n_fft_range/2);

% Collapse velocity dimension by taking the max
range_angle_map = squeeze(max(rdm_plot, [], 1));

figure;
imagesc(range_axis, angle_axis_deg, 10*log10(range_angle_map.'));
title('Range-Angle Map (Top-Down View)');
xlabel('Range (m)');
ylabel('Angle (degrees)');
colorbar;
axis xy;
ylim([-90 90]); % Show full -90 to +90 degree range
xlim([0 250]);  % Set x-axis limit as requested

% % --- 7. Find and Print the 3 Strongest Peaks ---
% num_peaks_to_find = 3;
% 
% % We will search the 3D cube
% rdm_search_plot = abs(rda_cube);
% 
% % Define suppression neighborhood
% range_suppression_bins = 3; 
% vel_suppression_bins = 3;
% angle_suppression_bins = 2;
% 
% fprintf('Detected %d strongest targets:\n', num_peaks_to_find);
% 
% for k = 1:num_peaks_to_find
%     [max_val, max_idx] = max(rdm_search_plot(:));
%     if max_val < 1e-6
%         fprintf('  (No more significant targets found)\n');
%         break;
%     end
% 
%     % <-- MODIFIED: ind2sub now returns 3 indices -->
%     [vel_idx, range_idx, angle_idx] = ind2sub(size(rdm_search_plot), max_idx);
% 
%     % Print this peak's info
%     fprintf('  Target %d:\n', k);
%     fprintf('    Range:    %.2f m\n', range_axis(range_idx));
%     fprintf('    Velocity: %.2f m/s\n', vel_axis(vel_idx));
%     fprintf('    Angle:    %.2f deg\n', angle_axis_deg(angle_idx));
% 
%     % --- Suppress this peak in 3D ---
%     r_min = max(1, range_idx - range_suppression_bins);
%     r_max = min(size(rdm_search_plot, 2), range_idx + range_suppression_bins);
%     v_min = max(1, vel_idx - vel_suppression_bins);
%     v_max = min(size(rdm_search_plot, 1), vel_idx + vel_suppression_bins);
%     a_min = max(1, angle_idx - angle_suppression_bins);
%     a_max = min(size(rdm_search_plot, 3), angle_idx + angle_suppression_bins);
% 
%     rdm_search_plot(v_min:v_max, r_min:r_max, a_min:a_max) = 0;
% end