

%% -------------------------- Config --------------------------
numFrames      = 100;         % how many RDM examples to generate
maxTargets     = 4;            % max targets per frame (0..maxTargets, sampled)
targetRange_m  = [20, 250];    % range span for sampling (m)
targetVel_mps  = [-40, 40];    % velocity span for sampling (m/s)
rcsRange       = [0.3, 15];    % random RCS range (linear scale)
snr_dB_range   = [5, 20];      % per-frame SNR (dB), uniform random

% Radar / waveform (close to your original)
c       = 3e8;
fc      = 77e9;
lambda  = c / fc;

bw      = 150e6;               % Bandwidth
T_chirp = 10e-6;               % Chirp duration (s)
k       = bw / T_chirp;        % Chirp slope
fs      = 2 * bw;              % ADC sample rate (Hz)

n_chirps   = 64;               % chirps per frame (slow-time)
T_frame    = T_chirp * n_chirps;

% FFT sizing
t = 0 : 1/fs : T_chirp - 1/fs;  % one-chirp fast-time
n_samples   = numel(t);

nFFT_range  = 2^nextpow2(n_samples);
nFFT_vel    = 2^nextpow2(n_chirps);

% Axes
f_beat_axis = fs * (0:(nFFT_range/2)) / nFFT_range;     % keep positive half incl DC
range_axis  = (f_beat_axis * c) / (2 * k);              % meters

fs_doppler        = 1 / T_chirp;                        % PRF in slow-time
doppler_freq_axis = linspace(-fs_doppler/2, fs_doppler/2, nFFT_vel);
vel_axis          = (doppler_freq_axis * lambda) / 2;   % m/s

nRangeBins = numel(range_axis);
nVelBins   = numel(vel_axis);

% Output tensors
RDMs   = zeros(nVelBins, nRangeBins, numFrames, 'single'); % [vel × range × frame]
Y_mask = false(nVelBins, nRangeBins, numFrames);           % same size, multi-target mask

% Per-frame label list (variable #targets) stored as struct array
Y_list(numFrames) = struct('range_m', [], 'vel_mps', [], 'rcs', [], ...
                           'range_bin', [], 'vel_bin', [], 'snr_dB', []);

%% ---------------------- Fixed Transmit ----------------------
tx_phase = 2 * pi * (fc * t + (k/2) * t.^2);
Tx = exp(1j * tx_phase);       % 1 × n_samples

%% ---------------------- Helper functions --------------------
nearest_idx = @(axis_vals, x) max(1, min(numel(axis_vals), round(interp1(axis_vals, 1:numel(axis_vals), x, 'nearest', 'extrap'))));

% Inverse mapping for labels:
% Range → beat frequency → range bin
f_b_from_R = @(R) (2 * k / c) * R;                % Hz
% Velocity → Doppler frequency (slow-time) → vel bin
f_d_from_v = @(v) (2 * v / lambda);               % Hz

%% ------------------- Main dataset loop ----------------------
for fIdx = 1:numFrames
    % Randomize frame configuration
    snr_dB = snr_dB_range(1) + (snr_dB_range(2)-snr_dB_range(1))*rand;
    n_tgts = randi([0, maxTargets]);  % allow empty scenes too

    % Pre-allocate slow-time/fast-time matrix
    Mix = zeros(n_chirps, n_samples);

    % Sample targets
    if n_tgts > 0
        R0  = targetRange_m(1) + (targetRange_m(2)-targetRange_m(1))*rand(1, n_tgts);
        V0  = targetVel_mps(1) + (targetVel_mps(2)-targetVel_mps(1))*rand(1, n_tgts);
        RCS = rcsRange(1)      + (rcsRange(2)-rcsRange(1))*rand(1, n_tgts);
    else
        R0 = []; V0 = []; RCS = [];
    end

    for i = 1:n_chirps
        Rx_total = zeros(1, n_samples);
        t_chirp_start = (i-1) * T_chirp;

        for j = 1:n_tgts
            % Target kinematics (constant velocity)
            R_curr = R0(j) + V0(j) * t_chirp_start;
            tau    = (2 * R_curr) / c;
            t_rx   = t - tau;

            % Reflected phase and signal
            rx_phase = 2 * pi * (fc * t_rx + (k/2) * t_rx.^2);
            Rx_j = exp(1j * rx_phase);

            % Simple attenuation: R^-4 scaled by RCS
            att = RCS(j) / max(R_curr, 1)^4;   % guard against R=0
            Rx_total = Rx_total + att * Rx_j;
        end

        % Add noise (per slow-time sample)
        Rx_noisy = awgn(Rx_total, snr_dB, 'measured');
        Mix(i, :) = Tx .* conj(Rx_noisy);  % de-chirp
    end

    % 2D FFT → Range–Doppler Map
    R_fft = fft(Mix, nFFT_range, 2);                     % fast-time FFT (range)
    RDM   = fftshift(fft(R_fft, nFFT_vel, 1), 1);        % slow-time FFT (velocity)
    RDM   = RDM(:, 1:(nFFT_range/2+1));                  % keep positive range

    % Magnitude → dB → normalize 0–1 per frame (robust)
    RDM_mag = abs(RDM);
    RDM_db  = 10*log10(RDM_mag + eps);

    % Normalize: per-frame percentile scaling (helps NN training)
    p1  = prctile(RDM_db(:), 1);
    p99 = prctile(RDM_db(:), 99);
    RDM_norm = (RDM_db - p1) ./ max(p99 - p1, 1e-6);
    RDM_norm = min(max(RDM_norm, 0), 1);                 % clip to [0,1]

    % Store
    RDMs(:, :, fIdx) = single(RDM_norm);

    % Build label mask & list using true kinematics → nearest bins
    if n_tgts > 0
        rng_bins = zeros(1, n_tgts);
        vel_bins = zeros(1, n_tgts);

        for j = 1:n_tgts
            % Use range/velocity at the START of the frame for labeling
            R_lab = R0(j);                 % could also use mid-frame; be consistent
            v_lab = V0(j);

            % Map to bins
            f_b   = f_b_from_R(R_lab);                             % Hz
            dF_r  = fs / nFFT_range;                               % Hz/bin along range FFT (full)
            r_idx = nearest_idx(f_beat_axis, f_b);                 % because we kept only positive half

            f_d   = f_d_from_v(v_lab);                             % Hz
            v_idx = nearest_idx(doppler_freq_axis, f_d);           % along slow-time FFT

            rng_bins(j) = r_idx;
            vel_bins(j) = v_idx;

            % Write a 1-pixel label; optionally, thicken with a small blob
            Y_mask(v_idx, r_idx, fIdx) = true;
        end

        % Save per-frame continuous labels + bins
        Y_list(fIdx).range_m   = R0;
        Y_list(fIdx).vel_mps   = V0;
        Y_list(fIdx).rcs       = RCS;
        Y_list(fIdx).range_bin = rng_bins;
        Y_list(fIdx).vel_bin   = vel_bins;
        Y_list(fIdx).snr_dB    = snr_dB;
    else
        Y_list(fIdx).range_m   = [];
        Y_list(fIdx).vel_mps   = [];
        Y_list(fIdx).rcs       = [];
        Y_list(fIdx).range_bin = [];
        Y_list(fIdx).vel_bin   = [];
        Y_list(fIdx).snr_dB    = snr_dB;
    end

    if mod(fIdx, 50) == 0
        fprintf('Generated %d / %d frames\n', fIdx, numFrames);
    end
end

%% -------------------------- Save ----------------------------
meta = struct();
meta.c = c; meta.fc = fc; meta.lambda = lambda;
meta.bw = bw; meta.T_chirp = T_chirp; meta.k = k; meta.fs = fs;
meta.n_chirps = n_chirps; meta.T_frame = T_frame;
meta.nFFT_range = nFFT_range; meta.nFFT_vel = nFFT_vel;
meta.snr_dB_range = snr_dB_range;
meta.range_axis = range_axis; meta.vel_axis = vel_axis;
meta.description = 'RDMs normalized to [0,1] per frame using 1–99th percentile. Y_mask has ones at target bins.';

save('rdm_dataset.mat', 'RDMs', 'Y_mask', 'Y_list', 'range_axis', 'vel_axis', 'meta', '-v7.3');

fprintf('Saved dataset to rdm_dataset.mat\n');

%% ---------------------- Quick visual check ------------------
% Visualize a random frame and its labels
idx = randi(numFrames);
R = RDMs(:, :, idx);
M = Y_mask(:, :, idx);

figure; 
imagesc(range_axis, vel_axis, R); axis xy;
title(sprintf('RDM (frame %d)', idx));
xlabel('Range (m)'); ylabel('Velocity (m/s)'); colorbar;
xlim([0, targetRange_m(2)]);



hold on;
[y_lab, x_lab] = find(M);
plot(range_axis(x_lab), vel_axis(y_lab), 'w+', 'MarkerSize', 8, 'LineWidth', 1.5);
hold off;