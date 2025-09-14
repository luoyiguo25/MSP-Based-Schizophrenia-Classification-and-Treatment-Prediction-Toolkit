clc; clear; close all;

% -------------------------------------------------------------------------
% Configuration and Initialization
% -------------------------------------------------------------------------
rng(42);  % Set random seed for reproducibility
time_point = 1;  % Time point (t1) to analyze
num_folds = 5;   % Number of cross-validation folds
dataname = '_SV';  % Data type identifier

% Create output directory path
output_path = fullfile('.', sprintf('PLS_ndyx_5folds_t%d', time_point));
if ~exist(output_path, 'dir')
    mkdir(output_path);  % Create directory if it doesn't exist
end

% -------------------------------------------------------------------------
% Step 1: Load and preprocess data
% -------------------------------------------------------------------------
% Load similarity data
data = load(fullfile('../similarity_ndyx', sprintf('similarity%s.mat', dataname)));

% Extract key variables from loaded data
similarity = data.similarity;
group = data.group;
rate = data.rate;
t1panss = data.t1panss;
t2panss = data.t2panss;
diffpanss = data.diffpanss;
name = data.name;
disease = data.disease;

% Remove 'epilepsy' data (6th column)
similarity(:, 6) = [];
diseaseDescription = disease;
diseaseDescription(6) = [];

% Select only data for specified time point (t1)
patient_indices = group == time_point;
similarity_patients = similarity(patient_indices, :);
rate_patients = rate(patient_indices);
t1panss_patients = t1panss(patient_indices);
t2panss_patients = t2panss(patient_indices);
diffpanss_patients = diffpanss(patient_indices);
group_patients = group(patient_indices);
name_patients = name(patient_indices);

% Handle missing values (replace NaNs with column means)
column_means = nanmean(similarity_patients);
nan_indices = isnan(similarity_patients);
for col = 1:size(similarity_patients, 2)
    similarity_patients(nan_indices(:, col), col) = column_means(col);
end

% -------------------------------------------------------------------------
% Step 2: Perform 5-fold cross-validation
% -------------------------------------------------------------------------
% Generate fold indices for cross-validation
fold_indices = crossvalind('Kfold', size(similarity_patients, 1), num_folds);
pls_results = cell(num_folds, 1);  % Store results for each fold

% Process each fold
for fold = 1:num_folds
    fprintf('Processing fold %d/%d...\n', fold, num_folds);
    
    % Split data into training and testing sets
    train_indices = fold_indices ~= fold;
    test_indices = fold_indices == fold;
    
    % Training data
    X_train = similarity_patients(train_indices, :);
    Y_train = (t2panss_patients(train_indices) - t1panss_patients(train_indices)) ...
             ./ t1panss_patients(train_indices);  % Relative change in PANSS
    G_train = group_patients(train_indices);
    
    % Testing data
    X_test = similarity_patients(test_indices, :);
    Y_test = (t2panss_patients(test_indices) - t1panss_patients(test_indices)) ...
            ./ t1panss_patients(test_indices);  % Relative change in PANSS
    G_test = group_patients(test_indices);
    
    % ---------------------------------------------------------------------
    % Step 3: Configure and run PLS analysis
    % ---------------------------------------------------------------------
    % Prepare input data structure
    input.brain_data = X_train;     % Brain imaging features
    input.behav_data = Y_train;     % Behavioral measure (PANSS change)
    input.grouping = G_train;       % Group information
    
    % Configure PLS options
    pls_opts.behav_type = 'behavior';          % Type of behavioral data
    pls_opts.grouped_PLS = 0;                  % No grouped PLS analysis
    pls_opts.normalization_behav = 1;          % Normalize behavioral data
    pls_opts.normalization_img = 1;            % Normalize imaging data
    pls_opts.nPerms = 10000;                   % Number of permutations
    pls_opts.nBootstraps = 5000;               % Number of bootstraps
    
    % Configure save options
    save_opts.output_path = output_path;       % Output directory
    save_opts.img_type = 'barPlot';            % Type of image output
    save_opts.grouped_plots = 0;               % No grouped plots
    
    % Initialize and run PLS analysis
    [input, pls_opts, save_opts] = myPLS_initialize(input, pls_opts, save_opts);
    pls_results{fold} = myPLS_analysis(input, pls_opts);
end

% -------------------------------------------------------------------------
% Step 4: Aggregate and analyze cross-validation results
% -------------------------------------------------------------------------
% Collect latent variables across all folds
all_Lx = [];  % Latent variables for brain data
all_Ly = [];  % Latent variables for behavioral data

for fold = 1:num_folds
    all_Lx = [all_Lx; pls_results{fold}.Lx];
    all_Ly = [all_Ly; pls_results{fold}.Ly];
end

% Calculate mean p-values across all folds
mean_LC_pvals = zeros(size(pls_results{1}.LC_pvals));
for fold = 1:num_folds
    mean_LC_pvals = mean_LC_pvals + pls_results{fold}.LC_pvals;
end
mean_LC_pvals = mean_LC_pvals / num_folds;

fprintf('Average LC p-values across all folds:\n');
disp(mean_LC_pvals);
