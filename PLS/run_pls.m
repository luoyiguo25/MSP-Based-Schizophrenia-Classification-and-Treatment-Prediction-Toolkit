clc; clear; close all;

% -------------------------------------------------------------------------
% Configuration
% -------------------------------------------------------------------------
dataname = '_CT';               % Data type identifier
time_point = 1;                 % Time point to analyze (1 or 2)
output_path = sprintf('./PLS_results_panss_PNG_t%d', time_point);  % Output directory

% Create output directory if it doesn't exist
if ~exist(output_path, 'dir')
    mkdir(output_path);
end

% -------------------------------------------------------------------------
% Step 1: Load and preprocess data
% -------------------------------------------------------------------------
% Load similarity data with PANSS measurements
data = load(sprintf('data/similarity%s_panss_PNG.mat', dataname));

% Extract key variables from loaded data
similarity = data.similarity;   % Similarity scores matrix
group = data.group;             % Group labels
panss = data.panss;             % Total PANSS scores
P = data.P;                     % Positive symptoms
N = data.N;                     % Negative symptoms
G = data.G;                     % Cognitive symptoms
name = data.name;               % Subject identifiers
disease = data.disease;         % List of disease categories

% Select data for specified time point (t1 or t2)
time_indices = group == time_point;
similarity_patients = similarity(time_indices, :);
panss_patients = panss(time_indices);
pos_patients = P(time_indices);
neg_patients = N(time_indices);
cog_patients = G(time_indices);
group_patients = group(time_indices);
name_patients = name(time_indices);

% Handle missing values (replace NaNs with column means)
column_means = nanmean(similarity_patients);
nan_indices = isnan(similarity_patients);
for col = 1:size(similarity_patients, 2)
    similarity_patients(nan_indices(:, col), col) = column_means(col);
end

% -------------------------------------------------------------------------
% Prepare PLS input data
% -------------------------------------------------------------------------
% Behavioral data: [Total PANSS, Positive symptoms, Negative symptoms, Cognitive symptoms]
Y = [panss_patients, pos_patients, neg_patients, cog_patients];
grouping = group_patients;      % Group information

% Configure PLS options
pls_opts.behav_type = 'behavior';          % Type of behavioral data
pls_opts.grouped_PLS = 0;                  % No grouped PLS analysis
pls_opts.normalization_behav = 1;          % Normalize behavioral data
pls_opts.normalization_img = 1;            % Normalize imaging data
pls_opts.nPerms = 200;                     % Number of permutations
pls_opts.nBootstraps = 100;                % Number of bootstraps

% Configure save options
save_opts.output_path = output_path;       % Output directory
save_opts.img_type = 'barPlot';            % Type of image output
save_opts.grouped_plots = 0;               % No grouped plots

% -------------------------------------------------------------------------
% PLS analysis
% -------------------------------------------------------------------------
input.brain_data = similarity_patients;
input.behav_data = Y;
input.grouping = grouping;

[input, pls_opts, save_opts] = myPLS_initialize(input, pls_opts, save_opts);
res = myPLS_analysis(input, pls_opts);

save(sprintf('%s/pls%s.mat', output_path, dataname), ...
     'res', 'similarity_patients', 'Y', 'grouping', 'name_patients', 'disease');
