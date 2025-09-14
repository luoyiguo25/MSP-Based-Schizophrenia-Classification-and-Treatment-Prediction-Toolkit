%--------------------------------------------------------------------------
% Brain Region Ablation AUC Visualization
% Purpose: Calculate relative AUC changes after brain region removal and 
%          visualize results using plotBrain function
%--------------------------------------------------------------------------

%% Clear Workspace & Load Data
clc; clear; close all;

% Load CSV data containing brain region ablation results
data = readtable('analysis_brains_contributions_t2.csv');

% Extract brain region names and raw AUC values
brain_regions = data.BrainRemoved;  % Brain region labels
auc_values = data.AUC;              % Corresponding AUC scores


%% Preprocess AUC Values (Relative Change Calculation)
% Get baseline AUC (from "All Brains" condition, last row)
baseline_auc = auc_values(end);

% Calculate relative AUC change (%) compared to baseline
% Formula: (AUC_after_removal - AUC_baseline) / AUC_baseline * 100
relative_auc_change = (auc_values - baseline_auc) / baseline_auc * 100;

% Remove baseline data ("All Brains") from visualization dataset
brain_regions = brain_regions(1:end-1);          % Keep only ablated regions
relative_auc_change = relative_auc_change(1:end-1);  % Corresponding AUC changes


%% Define Color Map (Blue-Red Gradient for Positive-Negative Changes)
% Color scheme: Dark Blue → Light Blue → White → Light Red → Dark Red
cm = [0.0471    0.3294    0.6471; ... % #0C54A5 (Dark Blue: Strong Negative Change)
      0.2157    0.5451    0.7882; ... % #378BC9 (Medium Blue)
      0.6784    0.8667    0.9647; ... % #ADDCF6 (Light Blue)
      0.7569    0.8627    0.9373; ... % #C1DCEF (Very Light Blue)
      0.9608    0.9725    0.9804; ... % #F5F8FA (Near White: No Change)
      0.9333    0.6353    0.6431; ... % #EEA2A4 (Light Red)
      0.9294    0.3529    0.3961; ... % #ED5A65 (Medium Red)
      0.9373    0.2157    0.3176; ... % #EF3751 (Dark Red)
      0.8196    0.1020    0.1765];    % #D11A2D (Strong Red: Strong Positive Change)

% Interpolate color map to get smoother gradient
cm = interp1(cm, 1:0.01:size(cm, 1));


%% Visualize AUC Changes on Brain Atlas
% Visualize real AUC changes (t2 time point)
plotBrain(brain_regions, ...
          relative_auc_change, ...
          cm, ...
          'atlas', 'aparc_aseg', ...       % Use aparc_aseg brain atlas
          'savePath', 'Brainfigures/t2', ...% Save figures to this path
          'limits', [-max(abs(relative_auc_change)) max(abs(relative_auc_change))]); % Symmetric color limits


%% Generate & Visualize Random AUC Values (Control Group)
rng('default');  % Set random seed for reproducibility
num_regions = length(brain_regions);
% Generate random values (simulate control data, 0.5-1.0 normalized range)
random_auc = 0.5 + 0.5 * rand(num_regions, 1);  

% Visualize random AUC values (for comparison)
plotBrain(brain_regions, ...
          random_auc, ...
          cm, ...
          'atlas', 'aparc_aseg', ...
          'savePath', 'Brainfigures/random', ...
          'limits', [-max(abs(random_auc)) max(abs(random_auc))]); % Symmetric color limits