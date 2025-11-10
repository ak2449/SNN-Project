clear; clc; close all;
% --- 1. Define Parameters ---
c = 3e8;                        % Speed of light (m/s)
fc = 77e9;                      % Carrier frequency (Hz)
bw = 300e6;                     % Chirp bandwidth (Hz)
T = 10e-6;                      % Chirp duration (s)
fs = 2 * bw;                    % Sample rate (Hz)

target_range = 50;              % Target range (m)
target_rcs = 1;                 % Target radar cross section (arbitrary for this demo)

% --- 2. Generate Transmitted Signal (Tx) ---
t = 0 : 1/fs : T - 1/fs;        % Time vector for one chirp
k = bw / T;                     % Chirp rate (slope)

% Phase of the transmitted signal
tx_phase = 2 * pi * (fc * t + (k/2) * t.^2);
Tx = cos(tx_phase);

% --- 3. Simulate Received Signal (Rx) ---

% Calculate round-trip time delay
tau = (2 * target_range) / c;

% Create the time vector for the received signal
t_rx = t - tau;

% Phase of the received signal
% We use the delayed time vector 't_rx' in the same phase equation
rx_phase = 2 * pi * (fc * t_rx + (k/2) * t_rx.^2);
Rx = cos(rx_phase);

% Attenuate the signal (optional, based on 1/R^4)
% For a simple demo, we can skip complex path loss and just add noise.
Rx = awgn(Rx, 20, 'measured');   % Add Additive White Gaussian Noise (SNR = 20 dB

% --- 4. Process the Signal (Mixing and FFT) ---

% Mix the Tx and Rx signals (element-wise multiplication)
Mix = Tx .* Rx;

% Perform an FFT on the mixed (beat) signal to find its frequency
n_fft = 1024; % Number of FFT points
Y = fft(Mix, n_fft);
P2 = abs(Y/n_fft);
P1 = P2(1:n_fft/2+1);
P1(2:end-1) = 2*P1(2:end-1);

% Create the frequency axis for the FFT plot
f_axis = fs * (0:(n_fft/2)) / n_fft;

% --- 5. Visualize the Result ---

% Convert the beat frequency axis to a range axis
range_axis = (f_axis * c) / (2 * k);

% Plot the Range-FFT
figure;
plot(range_axis, P1);
title('Radar Range-FFT Plot');
xlabel('Range (m)');
ylabel('Signal Strength');
grid on;

[max_val, max_idx] = max(P1);
fprintf('Detected target at %.2f meters.\n', range_axis(max_idx));