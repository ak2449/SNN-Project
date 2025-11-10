% A SCRIPT TO SIMULATE RADAR DETECTIONS OVER TIME 📡

clear; clc; close all;

%% 1. Define Radar Parameters
radarFreq = 77e9; % Radar frequency in Hz (77 GHz is common for automotive)
prf = 1e3;        % Pulse repetition frequency in Hz
fs = 2e9;         % Sampling rate in Hz (must be > bandwidth)
bandwidth = 1e9;  % Bandwidth in Hz
pulseWidth = 1e-6; % Pulse width in seconds

% Create a radar data generator object
% NOTE: This requires the Automated Driving Toolbox or Radar Toolbox
radar = radarDataGenerator('SensorIndex', 1, ...
                           'UpdateRate', prf, ... % Set the update rate to the PRF
                           'OperatingFrequency', radarFreq, ...
                           'SampleRate', fs, ...
                           'Bandwidth', bandwidth, ...
                           'PulseWidth', pulseWidth, ...
                           'HasOcclusion', false, ... % Keep simulation simple
                           'HasFalseAlarms', false); % No false alarms for this demo

%% 2. Define Target Parameters
targetPos = [100; 0; 0];  % Target position in meters [x; y; z]
targetVel = [-20; 0; 0]; % Target velocity in m/s [vx; vy; vz] (moving towards radar)

%% 3. Run the Simulation Loop
numPulses = 10; % Number of pulses to simulate
timeStep = 1 / prf;    % Time between each pulse
simulationTime = 0;    % Initialize simulation time

fprintf('--- Starting Radar Simulation ---\n');

for i = 1:numPulses
    % Create a struct for the target's current state (pose)
    % This is the required input format
    targetPose = struct('Position', targetPos, 'Velocity', targetVel);
    
    % Generate detection data for the current time step
    % The function needs the target pose(s) and the current time
    detections = radar(targetPose, simulationTime);
    
    % --- Process and Display Detections ---
    if isempty(detections)
        fprintf('Pulse %d at %.4f s: No detection\n', i, simulationTime);
    else
        % Detections are returned as a cell array of objectDetection objects
        detectedPos = detections{1}.Measurement(1:3); % Extract [x; y; z]
        fprintf('Pulse %d at %.4f s: Detected at range %.2f m\n', ...
                i, simulationTime, norm(detectedPos));
    end
    
    % --- Update Simulation State for Next Loop ---
    % Advance the simulation time
    simulationTime = simulationTime + timeStep;
    
    % Update the target's position based on its velocity
    targetPos = targetPos + targetVel * timeStep;
end