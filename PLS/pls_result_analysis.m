clc, clear, close all

%% Load PLS analysis results
load("PLS_results_t1/pls_SV_exclude_epilepsy.mat");

%% 1. Identify significant components
% Calculate cumulative explained variance
for ii = 1:numel(res.explCovLC)
    sumExpl(ii, 1) = sum(res.explCovLC(1:ii));
end

% Determine minimum components explaining 80% variance
[~, nComp] = min(pdist2(sumExpl, 0.80));

% FDR correction to identify significant latent components
idx_sigLC = find(mafdr(res.LC_pvals(1:nComp), 'BHFDR', true) < 0.05);

fprintf('# Significant LC indices: \n');
disp(idx_sigLC);
fprintf('Variance explained by LC1: %.4f\n', sumExpl(1));

%% 2. Visualize latent variable relationship (LC1)
% Scatter plot and regression line for similarity vs behavioral composite scores
II = 1; % Select first latent component
grouplabel = cell(size(res.grouping));
grouplabel(res.grouping == 1) = repmat({'SCZ'}, nnz(res.grouping == 1), 1);

% Create gramm graphics object
g = gramm('x', res.Lx(:, II), 'y', res.Ly(:, II), 'color', grouplabel);
g.geom_point();      % Scatter plot
g.stat_glm();        % Add regression line
g.set_color_options('map','d3_20');
g.set_point_options('base_size', 3);
g.set_names('x', 'Similarity Composite Scores', 'y', 'Behavioral Composite Scores', 'color', 'Group');
g.axe_property('FontSize', 10);
g.set_text_options('legend_title_scaling', 1, 'legend_scaling', 1);

% Plot figure
figure('Unit', 'centimeters', 'Position', [0 0 8 6]);
g.draw();

% Calculate correlation for schizophrenia group
[r, p] = corr(res.Lx(res.grouping == 1, 1), res.Ly(res.grouping == 1, 1));
fprintf('Lx vs Ly correlation - r=%.4f, p=%.4f\n', r, p);

%% 3. Brain feature contributions analysis (X variables)
for ii = 1:numel(idx_sigLC)
    II = idx_sigLC(ii);
    X_salience(:, II) = res.V(:, II);                    % Salience
    X_loadings(:, II) = res.LC_img_loadings(:, II);      % Loadings
    X_zval(:, II) = X_salience(:, II) ./ res.boot_results.Vb_std(:, II); % Z-values
    X_pval(:, II) = 2 * normcdf(-abs(X_zval(:, II)));    % P-values
    X_pvaladj(:, II) = mafdr(X_pval(:, II), 'BHFDR', true); % FDR-corrected P-values
end

% Count significant features
sig_features = nnz(X_pvaladj(:, II) < 0.05);
sig_percentage = sig_features / numel(X_pvaladj(:, II)) * 100;
pos_sig = nnz((X_pvaladj(:, II) < 0.05) & (X_salience(:, II) > 0));
neg_sig = nnz((X_pvaladj(:, II) < 0.05) & (X_salience(:, II) < 0));

fprintf('Significant features: %d (%.2f%%)\n', sig_features, sig_percentage);
fprintf('Positive significant features: %d, Negative significant features: %d\n', pos_sig, neg_sig);

%% 4. Disease type Z-value visualization
clear g;    
II = 1;
g = gramm('x', diseaseDescription, 'y', X_zval(:, II), ...
    'color', X_pvaladj(:, II) < 0.05);
g.geom_bar();  % Bar plot showing contributions of each disease type

g.axe_property('FontSize', 10);
g.set_names('x', '', 'y', 'Z-score', 'color', 'Significance');
g.set_color_options('map', [0.7,0.7,0.7; 0.1804 0.4824 0.7216], ...
                   'n_color', 2, 'n_lightness', 1);
g.no_legend();

figure('Unit', 'centimeters', 'Position', [0 0 11 6]);
g.draw();

%% 5. Clinical symptom contributions analysis (Y variables)
for ii = 1:numel(idx_sigLC)
    II = idx_sigLC(ii);
    Y_salience(:, II) = res.U(:, II);                    % Salience
    Y_loadings(:, II) = res.LC_behav_loadings(:, II);    % Loadings
    Y_zval(:, II) = Y_salience(:, II) ./ res.boot_results.Ub_std(:, II); % Z-values
    Y_pval(:, II) = 2 * normcdf(-abs(Y_zval(:, II)));    % P-values
    Y_pvaladj(:, II) = mafdr(Y_pval(:, II), 'BHFDR', true); % FDR-corrected P-values
end

% Clinical symptom labels
rates = {'Rating'; 'PANSS(t1)'; 'PANSS(t2)'; 'ΔPANSS'}';

%% 6. Clinical symptom Z-value visualization
clear g;    
g = gramm('x', rates, 'y', Y_zval(:, II), ...
          'color', Y_pvaladj(:, II) < 0.05);
g.geom_bar();  % Bar plot showing contributions of each clinical symptom

g.axe_property('FontSize', 10);
g.set_names('x', '', 'y', 'Z-score', 'color', 'Significance');
g.axe_property('YLim', [-2 4]);
g.set_color_options('map', [0.7,0.7,0.7; 0.1804 0.4824 0.7216], ...
                   'n_color', 2, 'n_lightness', 1);
g.no_legend();

figure('Unit', 'centimeters', 'Position', [0 0 8 6]);
g.draw();