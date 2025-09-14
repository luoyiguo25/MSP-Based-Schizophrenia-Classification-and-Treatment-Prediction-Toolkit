clc; clear; close all;

% -------------------------------------------------------------------------
% Configuration
% -------------------------------------------------------------------------
dataname = 'SV+CT';  % Data type: 'SV', 'CT', 'SA', or 'SV+CT'
output_file = sprintf('../similarity_ndyx/similarity_removebrain_%s.mat', dataname);

% -------------------------------------------------------------------------
% Step 1: Load ENIGMA summary statistics
% -------------------------------------------------------------------------
enigma_data = load('../data/enigma_stats_adult.mat');

% -------------------------------------------------------------------------
% Step 2: Load and prepare z-score data based on data type
% -------------------------------------------------------------------------
switch dataname
    case 'SV'
        % Subcortical Volume data
        enigma_values = enigma_data.cohensd_SUBVOL;
        
        % Load z-score data (female + male)
        zdata_female = readtable('../centileBrain/SV_female/zscore_SubcorticalVolume_female.csv');
        zdata_male = readtable('../centileBrain/SV_male/zscore_SubcorticalVolume_male.csv');
        zdata_combined = cat(1, zdata_female, zdata_male);
        
        % Reorder columns for SV data
        zdata_combined = zdata_combined(:, [1:2:13, 2:2:14]);
        
    case 'CT'
        % Cortical Thickness data
        enigma_values = enigma_data.cohensd_CT;
        
        % Load z-score data (female + male)
        zdata_female = readtable('../centileBrain/CT_female/zscore_CorticalThickness_female.csv');
        zdata_male = readtable('../centileBrain/CT_male/zscore_CorticalThickness_male.csv');
        zdata_combined = cat(1, zdata_female, zdata_male);
        
    case 'SA'
        % Surface Area data
        enigma_values = enigma_data.cohensd_SA;
        
        % Load z-score data (female + male)
        zdata_female = readtable('../centileBrain/SA_female/zscore_SurfaceArea_female.csv');
        zdata_male = readtable('../centileBrain/SA_male/zscore_SurfaceArea_male.csv');
        zdata_combined = cat(1, zdata_female, zdata_male);
        
    case 'SV+CT'
        % Combined Subcortical Volume + Cortical Thickness data
        enigma_sv = enigma_data.cohensd_SUBVOL;
        enigma_ct = enigma_data.cohensd_CT;
        enigma_values = cat(1, enigma_sv, enigma_ct);  % Combine vertically
        
        % Load and combine SV z-scores
        zdata_sv_female = readtable('../centileBrain/SV_female/zscore_SubcorticalVolume_female.csv');
        zdata_sv_male = readtable('../centileBrain/SV_male/zscore_SubcorticalVolume_male.csv');
        zdata_sv = cat(1, zdata_sv_female, zdata_sv_male);
        zdata_sv = zdata_sv(:, [1:2:13, 2:2:14]);  % Reorder SV columns
        
        % Load and combine CT z-scores
        zdata_ct_female = readtable('../centileBrain/CT_female/zscore_CorticalThickness_female.csv');
        zdata_ct_male = readtable('../centileBrain/CT_male/zscore_CorticalThickness_male.csv');
        zdata_ct = cat(1, zdata_ct_female, zdata_ct_male);
        
        % Combine SV and CT z-scores horizontally
        zdata_combined = cat(2, zdata_sv, zdata_ct);
        
    otherwise
        error('Invalid dataname. Supported types: ''SV'', ''CT'', ''SA'', ''SV+CT''');
end

% Convert z-score table to numeric array
zval = table2array(zdata_combined);

% -------------------------------------------------------------------------
% Step 3: Load subject metadata and align data
% -------------------------------------------------------------------------
% Load subject demographic data (female + male)
subjects_female = readtable('../centileBrain/data_centileBrain_subcort_female.xlsx');
subjects_male = readtable('../centileBrain/data_centileBrain_subcort_male.xlsx');
subjects_combined = cat(1, subjects_female, subjects_male);

% Load ndyx dataset for alignment
ndyx_data = load('../data/ndyx_VOL_SA_CT_SUBVOL_COV.mat');

% Find indices to align subjects between datasets
[~, align_indices] = ismember(subjects_combined.subjects, ndyx_data.subjects);

% Extract clinical and demographic variables
group = ndyx_data.group(align_indices, :);          % Group classification
subject_ids = ndyx_data.subjects(align_indices, :); % Subject identifiers
rate = ndyx_data.rate(align_indices, :);            % Rate measure
t1panss = ndyx_data.t1panss(align_indices, :);      % PANSS score at time 1
t2panss = ndyx_data.t2panss(align_indices, :);      % PANSS score at time 2
diffpanss = t1panss - t2panss;                      % Change in PANSS scores
pos = ndyx_data.t2panss_pos(align_indices, :);      % Positive symptoms
neg = ndyx_data.t2panss_neg(align_indices, :);      % Negative symptoms
cog = ndyx_data.t2panss_cog(align_indices, :);      % Cognitive symptoms
diseases = enigma_data.diseaseDescriptions;         % List of diseases

% Define brain region labels based on data type
if strcmp(dataname, 'SV')
    brain_regions = ndyx_data.subcortexDescriptions;
elseif strcmp(dataname, 'SV+CT')
    brain_regions = cat(1, ndyx_data.subcortexDescriptions, ndyx_data.regionDescriptions);
else  % CT or SA
    brain_regions = ndyx_data.regionDescriptions;
end

% -------------------------------------------------------------------------
% Step 4: Calculate similarity with sequential brain region removal
% -------------------------------------------------------------------------
num_diseases = numel(diseases);          % Number of diseases
num_regions = size(enigma_values, 1);    % Number of brain regions
num_subjects = size(zval, 1);            % Number of subjects

% Initialize 3D similarity matrix: [subjects x diseases x regions_removed]
similarity = zeros(num_subjects, num_diseases, num_regions);

% Calculate similarity with one brain region removed at a time
for disease_idx = 1:num_diseases
    % Get ENIGMA values for current disease
    enigma_vector = enigma_values(:, disease_idx);
    
    for region_idx = 1:num_regions
        % Remove current region from ENIGMA vector
        enigma_no_region = enigma_vector;
        enigma_no_region(region_idx) = [];
        
        for subject_idx = 1:num_subjects
            % Remove corresponding region from subject's z-score vector
            zscore_vector = zval(subject_idx, :)';
            zscore_no_region = zscore_vector;
            zscore_no_region(region_idx) = [];
            
            % Calculate correlation (similarity) between the modified vectors
            similarity(subject_idx, disease_idx, region_idx) = corr(enigma_no_region, zscore_no_region);
        end
    end
end

% -------------------------------------------------------------------------
% Step 5: Handle outliers in similarity scores
% -------------------------------------------------------------------------
for disease_idx = 1:num_diseases
    for region_idx = 1:num_regions
        % Replace outliers with NaN
        current_similarity = similarity(:, disease_idx, region_idx);
        outlier_mask = isoutlier(current_similarity);
        current_similarity(outlier_mask) = nan;
        similarity(:, disease_idx, region_idx) = current_similarity;
    end
end

% -------------------------------------------------------------------------
% Step 6: Save results
% -------------------------------------------------------------------------
save(output_file, ...
     "similarity", "group", "subject_ids", "rate", ...
     "t1panss", "t2panss", "diffpanss", "diseases", ...
     "pos", "cog", "neg", "brain_regions", "-mat");

fprintf('Results saved to: %s\n', output_file);
