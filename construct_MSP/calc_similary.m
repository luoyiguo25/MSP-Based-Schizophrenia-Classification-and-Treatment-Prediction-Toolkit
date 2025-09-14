clc; clear; close all;

% Define data type (Surface Area in this case)
dataname = 'SA';

% -------------------------------------------------------------------------
% Step 1: Load ENIGMA summary statistics
% -------------------------------------------------------------------------
enigma_data = load('../data/enigma_stats_adult.mat');
d_subvol = enigma_data.cohensd_SA;  % Extract Surface Area Cohen's d values

% -------------------------------------------------------------------------
% Step 2: Load z-score data (both genders)
% -------------------------------------------------------------------------
% Load female and male z-score data
zdata_female = readtable('../centileBrain/SA_female/zscore_SurfaceArea_female.csv');
zdata_male = readtable('../centileBrain/SA_male/zscore_SurfaceArea_male.csv');

% Combine female and male data
zdata_combined = cat(1, zdata_female, zdata_male);
zval = table2array(zdata_combined);  % Convert to numeric array

% -------------------------------------------------------------------------
% Step 3: Load subject information and align data
% -------------------------------------------------------------------------
% Load subject demographic data
person_female = readtable('../centileBrain/data_centileBrain_subcort_female.xlsx');
person_male = readtable('../centileBrain/data_centileBrain_subcort_male.xlsx');
person_combined = cat(1, person_female, person_male);

% Load ndyx dataset for alignment
ndyx_data = load('../data/ndyx_VOL_SA_CT_SUBVOL_COV.mat');

% Find indices to align subjects between datasets
[~, align_indices] = ismember(person_combined.subjects, ndyx_data.subjects);

% -------------------------------------------------------------------------
% Step 4: Extract relevant clinical measures
% -------------------------------------------------------------------------
group = ndyx_data.group(align_indices, :);          % Group classification
subject_ids = ndyx_data.subjects(align_indices, :); % Subject identifiers
rate = ndyx_data.rate(align_indices, :);            % Rate measure
t1panss = ndyx_data.t1panss(align_indices, :);      % PANSS score at time 1
t2panss = ndyx_data.t2panss(align_indices, :);      % PANSS score at time 2
diffpanss = t1panss - t2panss;                      % Change in PANSS scores
pos = ndyx_data.t2panss_pos(align_indices, :);      % Positive symptoms
neg = ndyx_data.t2panss_neg(align_indices, :);      % Negative symptoms
cog = ndyx_data.t2panss_cog(align_indices, :);      % Cognitive symptoms
diseases = enigma_data.diseaseDescriptions;         % List of diseases from ENIGMA

% -------------------------------------------------------------------------
% Step 5: Calculate similarity (correlation) between z-scores and ENIGMA data
% -------------------------------------------------------------------------
num_diseases = numel(enigma_data.diseaseDescriptions);
num_subjects = size(zval, 1);
similarity = zeros(num_subjects, num_diseases);  % Initialize similarity matrix

% Compute correlation for each disease and subject
for disease_idx = 1:num_diseases
    enigma_vector = d_subvol(:, disease_idx);  % ENIGMA Cohen's d values for current disease
    
    for subject_idx = 1:num_subjects
        zscore_vector = zval(subject_idx, :)';  % z-score vector for current subject
        similarity(subject_idx, disease_idx) = corr(enigma_vector, zscore_vector);
    end
end

% -------------------------------------------------------------------------
% Step 6: Remove outliers and save results
% -------------------------------------------------------------------------
% Replace outliers with NaN for each disease
for disease_idx = 1:size(similarity, 2)
    current_similarity = similarity(:, disease_idx);
    outlier_indices = isoutlier(current_similarity);
    current_similarity(outlier_indices) = nan;
    similarity(:, disease_idx) = current_similarity;
end

% Save results to MAT file
save(strcat('../centileSimilarity/similarity_', dataname, '.mat'), ...
     "similarity", "group", "subject_ids", "rate", ...
     "t1panss", "t2panss", "diffpanss", "diseases", ...
     "pos", "cog", "neg", "-mat");
