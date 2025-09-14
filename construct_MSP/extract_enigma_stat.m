clc; clear; close all;

% Add ENIGMA toolbox to MATLAB path (includes all subdirectories)
addpath(genpath('./ENIGMA-2.0.0/matlab'));


%% -------------------------------------------------------------------------
% Step 1: Load and process ENIGMA summary statistics
% -------------------------------------------------------------------------
disease_list = {'22q', 'adhd', 'asd', 'bipolar', 'depression', 'epilepsy', ...
                'ocd', 'schizophrenia'};

res = extract_enigma_data(disease_list);

ndyx_data = load('../data/ndyx_CT_SUBVOL_COV.mat');

% -------------------------------------------------------------------------
% Step 2: Align ENIGMA data with ndyx data (match region order)
% -------------------------------------------------------------------------

[~, cortex_idx] = ismember(ndyx_data.regionDescriptions, res.regionDescriptions);
res.cohensd_CT = res.cohensd_CT(cortex_idx, :);          % Cohen's d for Cortical Thickness
res.cohensd_SA = res.cohensd_SA(cortex_idx, :);          % Cohen's d for Surface Area
res.pfdr_CT = res.pfdr_CT(cortex_idx, :);                % FDR-corrected p-values (CT)
res.regionDescriptions = res.regionDescriptions(cortex_idx);  % Aligned region labels

[~, subcortex_idx] = ismember(ndyx_data.subcortexDescriptions, res.subcortexDescriptions);
res.cohensd_SUBVOL = res.cohensd_SUBVOL(subcortex_idx, :);  % Cohen's d for Subcortical Volume
res.pfdr_SUBVOL = res.pfdr_SUBVOL(subcortex_idx, :);        % FDR-corrected p-values (SubVol)
res.subcortexDescriptions = res.subcortexDescriptions(subcortex_idx);  % Aligned subcortex labels

save('../data/enigma_stats_adult.mat', '-struct', 'res');


%% -------------------------------------------------------------------------
% Function: extract_enigma_data
% Purpose:  Extracts and standardizes summary statistics (Cohen's d, p-values)
%           from ENIGMA dataset for specified diseases, plus generates 
%           cortical/subcortical effect size visualizations
% Input:    disease_list - Cell array of disease names to process
% Output:   res          - Structure containing standardized ENIGMA data:
%                           - cohensd_CT:        Cortical Thickness Cohen's d (68 regions x N diseases)
%                           - cohensd_SA:        Surface Area Cohen's d (68 regions x N diseases)
%                           - cohensd_SUBVOL:    Subcortical Volume Cohen's d (16 regions x N diseases)
%                           - pfdr_CT:           FDR-corrected p-values (CT) (68xN)
%                           - pfdr_SUBVOL:       FDR-corrected p-values (SubVol) (16xN)
%                           - regionDescriptions: Cortical region labels (68x1 cell)
%                           - subcortexDescriptions: Subcortical region labels (16x1 cell)
%                           - diseaseDescriptions: List of processed diseases (Nx1 cell)
% -------------------------------------------------------------------------
function res = extract_enigma_data(disease_list)
    % Initialize output structure fields
    num_diseases = numel(disease_list);
    res.cohensd_CT = zeros(68, num_diseases);    % 68 cortical regions (standard ENIGMA)
    res.cohensd_SA = zeros(68, num_diseases);    % 68 cortical regions
    res.cohensd_SUBVOL = zeros(16, num_diseases);% 16 subcortical regions (standard ENIGMA)
    res.pfdr_CT = zeros(68, num_diseases);       % FDR p-values for CT
    res.pfdr_SUBVOL = zeros(16, num_diseases);   % FDR p-values for SubVol
    res.diseaseDescriptions = disease_list;      % Store list of processed diseases

    % Process each disease sequentially
    for disease_idx = 1:num_diseases
        current_disease = disease_list{disease_idx};
        fprintf('Processing disease: %s\n', current_disease);

        enigma_stats = load_summary_stats(current_disease);

        % -----------------------------------------------------------------
        % Step 1: Extract relevant metrics (CT, SA, SubVol) based on disease
        % Note: ENIGMA dataset uses different table names for different diseases
        % -----------------------------------------------------------------
        switch current_disease
            case '22q'
                % 22q11.2 deletion syndrome: Direct table names
                ct_table = enigma_stats.CortThick_case_vs_controls;
                sa_table = enigma_stats.CortSurf_case_vs_controls;
                subvol_table = enigma_stats.SubVol_case_vs_controls;
                cortex_labels = sa_table.Structure;  % Get cortical region labels from SA table

            case 'adhd'
                % ADHD: Use "allages" tables (combines child/adult data)
                ct_table = enigma_stats.CortThick_case_vs_controls_allages;
                sa_table = enigma_stats.CortSurf_case_vs_controls_allages;
                subvol_table = enigma_stats.SubVol_case_vs_controls_allages;
                cortex_labels = sa_table.Structure;

            case 'asd'
                % ASD: Use meta-analysis tables; SA data unavailable (fill with NaN)
                ct_table = enigma_stats.CortThick_case_vs_controls_meta_analysis;
                sa_table = table(nan(68, 1), 'VariableNames', {'d_icv'});  % Dummy SA table
                subvol_table = enigma_stats.SubVol_case_vs_controls_meta_analysis;
                cortex_labels = ct_table.Structure;  % Get labels from CT table

            case 'bipolar'
                % Bipolar disorder: Use adult tables; SubVol = Type I bipolar
                ct_table = enigma_stats.CortThick_case_vs_controls_adult;
                sa_table = enigma_stats.CortSurf_case_vs_controls_adult;
                subvol_table = enigma_stats.SubVol_case_vs_controls_typeI;
                cortex_labels = sa_table.Structure;

            case 'depression'
                % Major depression: Use adult tables
                ct_table = enigma_stats.CortThick_case_vs_controls_adult;
                sa_table = enigma_stats.CortSurf_case_vs_controls_adult;
                subvol_table = enigma_stats.SubVol_case_vs_controls;
                cortex_labels = sa_table.Structure;

            case 'epilepsy'
                % Epilepsy: Use "allepilepsy" tables; SA data unavailable (fill with NaN)
                ct_table = enigma_stats.CortThick_case_vs_controls_allepilepsy;
                sa_table = table(nan(68, 1), 'VariableNames', {'d_icv'});  % Dummy SA table
                subvol_table = enigma_stats.SubVol_case_vs_controls_allepilepsy;
                cortex_labels = ct_table.Structure;

            case 'ocd'
                % OCD: Use adult tables
                ct_table = enigma_stats.CortThick_case_vs_controls_adult;
                sa_table = enigma_stats.CortSurf_case_vs_controls_adult;
                subvol_table = enigma_stats.SubVol_case_vs_controls_adult;
                cortex_labels = sa_table.Structure;

            case 'schizophrenia'
                % Schizophrenia: Direct table names
                ct_table = enigma_stats.CortThick_case_vs_controls;
                sa_table = enigma_stats.CortSurf_case_vs_controls;
                subvol_table = enigma_stats.SubVol_case_vs_controls;
                cortex_labels = sa_table.Structure;

            otherwise
                error('Unsupported disease: %s. Update switch-case to include this disease.', current_disease);
        end

        % -----------------------------------------------------------------
        % Step 2: Extract Cohen's d (effect size) and FDR p-values
        % -----------------------------------------------------------------
        res.cohensd_CT(:, disease_idx) = ct_table.d_icv;
        res.pfdr_CT(:, disease_idx) = ct_table.fdr_p;

        res.cohensd_SA(:, disease_idx) = sa_table.d_icv;

        res.cohensd_SUBVOL(:, disease_idx) = subvol_table.d_icv;
        res.pfdr_SUBVOL(:, disease_idx) = subvol_table.fdr_p;

        % -----------------------------------------------------------------
        % Step 3: Generate and save visualization plots
        % -----------------------------------------------------------------
        % Plot 1: Subcortical Volume effect size (Cohen's d)
        fig_subvol = figure('Units', 'centimeters', 'Position', [0, 0, 12, 6]);  % 12x6 cm figure
        max_effect_subvol = max(abs(res.cohensd_SUBVOL(:, disease_idx)));  % Dynamic color range
        plot_subcortical(res.cohensd_SUBVOL(:, disease_idx), ...
                        'color_range', [-max_effect_subvol, max_effect_subvol]);
        % Save plot as high-resolution PNG (300 DPI)
        exportgraphics(fig_subvol, ...
                      ['../figures2/SV_', current_disease, '.png'], ...
                      'Resolution', 300);
        close(fig_subvol);  % Close figure to free memory

        % Plot 2: Cortical Thickness effect size (Cohen's d)
        fig_ct = figure('Units', 'centimeters', 'Position', [0, 0, 12, 6]);  % 12x6 cm figure
        max_effect_ct = max(abs(res.cohensd_CT(:, disease_idx)));  % Dynamic color range
        ct_surface_data = parcel_to_surface(res.cohensd_CT(:, disease_idx), 'aparc_fsa5');
        plot_cortical(ct_surface_data, ...
                     'color_range', [-max_effect_ct, max_effect_ct]);
        exportgraphics(fig_ct, ...
                      ['../figures2/CT_', current_disease, '.png'], ...
                      'Resolution', 300);
        close(fig_ct);  % Close figure to free memory
    end

    % -----------------------------------------------------------------
    % Step 4: Standardize cortical region labels (LH/RH prefixes)
    % Convert ENIGMA's default labels (e.g., 'bankssts_lh') to 'ctx-lh-bankssts'
    % -----------------------------------------------------------------
    standardized_cortex_labels = cell(68, 1);
    for label_idx = 1:34  % Left hemisphere (first 34 regions)
        label_parts = strsplit(cortex_labels{label_idx}, '_');
        standardized_cortex_labels{label_idx} = ['ctx-lh-', label_parts{2}];
    end
    for label_idx = 35:68  % Right hemisphere (last 34 regions)
        label_parts = strsplit(cortex_labels{label_idx}, '_');
        standardized_cortex_labels{label_idx} = ['ctx-rh-', label_parts{2}];
    end
    res.regionDescriptions = standardized_cortex_labels;

    % -----------------------------------------------------------------
    % Step 5: Define standardized subcortical region labels
    % Matches ENIGMA's default 16 subcortical regions (left → right order)
    % -----------------------------------------------------------------
    res.subcortexDescriptions = {
        'Left-Accumbens-area';
        'Left-Amygdala';
        'Left-Caudate';
        'Left-Hippocampus';
        'Left-Pallidum';
        'Left-Putamen';
        'Left-Thalamus';
        'Left-Lateral-Ventricle';
        'Right-Accumbens-area';
        'Right-Amygdala';
        'Right-Caudate';
        'Right-Hippocampus';
        'Right-Pallidum';
        'Right-Putamen';
        'Right-Thalamus';
        'Right-Lateral-Ventricle'
    };

res.pfdr_CT = CT_p;
res.cohensd_CT = CT_d;
res.cohensd_SA = SA_d;
res.diseaseDescriptions = diseases;
res.cohensd_SUBVOL = SUBVOL_d;
res.pfdr_SUBVOL = SUBVOL_p;
res.regionDescriptions = regionDescriptions;

end