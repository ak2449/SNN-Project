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

% Scenario parameters
target_range = [200, 75,  50];     % Target initial ranges (m)
target_vel = [20,  -15, 10];       % Target velocities (m/s)
target_rcs = [10,   0.8, 0.5];     % Radar Cross Section - relates to reflectivity of object
n_targets = length(target_range);  % Number of targets

% Generate Tx
t = 0 : 1/fs : T_chirp - 1/fs;      % Time vector for ONE chirp
n_samples = length(t);              % Number of samples per chirp
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
        att = target_rcs(j) / (current_range^4); 
        
        % Add this target's signal to the total
        Rx_total = Rx_total + (att * Rx_j);
    end
    % Add noise
    Rx_noisy = awgn(Rx_total, 10, 'measured');
    
    % Mix the total received signal with the transmitted
    Mix_matrix(i, :) = Tx .* conj(Rx_noisy);
end

% Process the Signal (2D FFT)
n_fft_range = 2^nextpow2(n_samples); 
range_fft = fft(Mix_matrix, n_fft_range, 2); 
n_fft_vel = 2^nextpow2(n_chirps);  
rdm = fftshift(fft(range_fft, n_fft_vel, 1), 1);

% Create Axes for Plotting 
f_beat_axis = fs * (0:(n_fft_range/2)) / n_fft_range;
range_axis = (f_beat_axis * c) / (2 * k);

fs_doppler = 1 / T_chirp; 
doppler_freq_axis = linspace(-fs_doppler/2, fs_doppler/2, n_fft_vel);
vel_axis = (doppler_freq_axis * lambda) / 2;

% Visualize the Range-Doppler Map (RDM)
figure;
rdm_plot = abs(rdm(:, 1:n_fft_range/2+1));
imagesc(range_axis, vel_axis, 10*log10(rdm_plot));
title('Range-Doppler Map (RDM)');
xlabel('Range (m)');
ylabel('Velocity (m/s)');
colorbar;
axis xy; 
xlim([0, 250]);

% Find the Peak - only finds strongest target
[max_val, max_idx] = max(rdm_plot(:));
[vel_idx, range_idx] = ind2sub(size(rdm_plot), max_idx);
fprintf('Detected STRONGEST target at:\n');
fprintf('  Range:    %.2f m\n', range_axis(range_idx));
fprintf('  Velocity: %.2f m/s\n', vel_axis(vel_idx));


% Find and Print the 3 Strongest Peaks
% 
% num_peaks_to_find = 3;
% 
% % Make a copy of the RDM plot, as we will be modifying it
% rdm_search_plot = rdm_plot;
% 
% % Set a "suppression neighborhood" size. This prevents finding multiple
% % bins from the same target. You may need to tune these values.
% range_suppression_bins = 5;  % Suppress +/- 3 bins in range
% vel_suppression_bins = 5;    % Suppress +/- 3 bins in velocity
% 
% fprintf('Detected %d strongest targets:\n', num_peaks_to_find);
% 
% for k = 1:num_peaks_to_find
%     % Find the single strongest peak in the current search plot
%     [max_val, max_idx] = max(rdm_search_plot(:));
% 
%     % Check if the peak is just noise (value is near zero)
%     if max_val < 1e-6 % A small threshold to stop searching in empty noise
%         fprintf('  (No more significant targets found)\n');
%         break;
%     end
% 
%     % Convert the linear index to (row, col) / (velocity, range)
%     [vel_idx, range_idx] = ind2sub(size(rdm_search_plot), max_idx);
% 
%     % Print this peak's info
%     fprintf('  Target %d:\n', k);
%     fprintf('    Range:    %.2f m\n', range_axis(range_idx));
%     fprintf('    Velocity: %.2f m/s\n', vel_axis(vel_idx));
%     fprintf('    Strength: %.4f (linear magnitude)\n', max_val);
% 
%     % --- Suppress this peak so we don't find it again ---
%     % Define the boundaries of the neighborhood to suppress
%     r_min = max(1, range_idx - range_suppression_bins);
%     r_max = min(size(rdm_search_plot, 2), range_idx + range_suppression_bins);
%     v_min = max(1, vel_idx - vel_suppression_bins);
%     v_max = min(size(rdm_search_plot, 1), vel_idx + vel_suppression_bins);
% 
%     % Set this whole neighborhood to 0 in the search plot
%     rdm_search_plot(v_min:v_max, r_min:r_max) = 0;
% end