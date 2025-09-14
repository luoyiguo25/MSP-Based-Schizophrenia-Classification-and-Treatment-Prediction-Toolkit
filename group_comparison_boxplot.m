% Group comparison and longitudinal comparison of healthy and patient groups with boxplots

clc; clear; close all;

% -------------------------------------------------------------------------
% Configuration
% -------------------------------------------------------------------------
dataname = 'SV';  % Data type identifier (e.g., 'SV', 'CT', 'SA')

% Define disease to visualize (options: 'Schizophrenia', 'Bipolar', 'Depression', '22q', 'ocd')
target_disease = '22q';

% Define custom colors in RGB format
colorsRGB = [
    116/255, 155/255, 225/255;   % Primary blue
    172/255, 201/255, 233/255;   % Light blue
    242/255, 221/255, 179/255];  % Beige

% -------------------------------------------------------------------------
% Step 1: Load data
% -------------------------------------------------------------------------
data_path = strcat('../similarity_ndyx/similarity_', dataname, '.mat');
loaded_data = load(data_path);

% Extract key variables from loaded data
similarity = loaded_data.similarity;  % Similarity scores matrix
group = loaded_data.group;            % Group labels (0=HC, 1=SCZ(t1), 2=SCZ(t2))
diseases = loaded_data.disease;       % List of disease names

% -------------------------------------------------------------------------
% Step 2: Statistical analysis - Group differences (optional)
% -------------------------------------------------------------------------
num_diseases = size(similarity, 2);
[ttval1, pp1, ttval2, pp2] = calculate_group_statistics(similarity, group, num_diseases);

fdr_pp1 = mafdr(pp1, 'BHFDR', true);  % FDR for baseline comparison
fdr_pp2 = mafdr(pp2, 'BHFDR', true);  % FDR for follow-up comparison
 
disp('FDR-corrected similarity score difference (HC vs SCZ baseline):');
disp(table(diseases', ttval1, pp1, fdr_pp1, 'VariableNames', {'Disease', 'Tstat', 'RawP', 'FDR_P'}));

disp('FDR-corrected similarity score difference (HC vs SCZ follow-up):');
disp(table(diseases', ttval2, pp2, fdr_pp2, 'VariableNames', {'Disease', 'Tstat', 'RawP', 'FDR_P'}));

% -------------------------------------------------------------------------
% Step 3: Statistical analysis - Longitudinal differences (optional)
% -------------------------------------------------------------------------
[tval_long, p_long] = calculate_longitudinal_statistics(similarity, group);

disp('Longitudinal similarity score differences (SCZ baseline vs follow-up):');
disp(table(diseases', tval_long, p_long, mafdr(p_long, 'BHFDR', true), ...
    'VariableNames', {'Disease', 'Tstat', 'RawP', 'FDR_P'}));

% -------------------------------------------------------------------------
% Step 4: Create boxplot for specified disease
% -------------------------------------------------------------------------
figure;

disease_idx = get_disease_index(target_disease);

boxplot_handle = boxplot(similarity(:, disease_idx), group, 'Colors', [0 0 0]);

set(gca, ...
    'XTickLabel', {'HC', 'SCZ(t1)', 'SCZ(t2)'}, ...  % Group labels
    'YLabel', get(gca, 'YLabel'), ...
    'YTick', [-0.5, 0, 0.5], ...                     % Fixed y-axis ticks
    'FontSize', 9, ...                               % Font size for axes
    'box', 'off', ...                                % Remove box around plot
    'YColor', 'k', 'XColor', 'k');                   % Keep axis lines black

ylabel('Similarity');  % Y-axis label

customize_boxplot(boxplot_handle, colorsRGB);

widthInches = 5 / 2.54;
heightInches = 4 / 2.54;
set(gcf, 'Units', 'Inches', 'Position', [0, 0, widthInches, heightInches]);

output_filename = strcat('../pictures/boxplot_', dataname, '_', target_disease, '.svg');
saveas(gcf, output_filename);


% -------------------------------------------------------------------------
% Helper function: Calculate group comparison statistics
% -------------------------------------------------------------------------
function [ttval1, pp1, ttval2, pp2] = calculate_group_statistics(similarity, group, num_diseases)
    ttval1 = zeros(num_diseases, 1);  % T-statistics (HC vs SCZ t1)
    pp1 = zeros(num_diseases, 1);     % P-values (HC vs SCZ t1)
    ttval2 = zeros(num_diseases, 1);  % T-statistics (HC vs SCZ t2)
    pp2 = zeros(num_diseases, 1);     % P-values (HC vs SCZ t2)
    
    for disease_idx = 1:num_diseases
        % Compare HC (0) vs SCZ baseline (1)
        [~, pp1(disease_idx), ~, stat] = ttest2(...
            similarity(group == 0, disease_idx), ...
            similarity(group == 1, disease_idx));
        ttval1(disease_idx) = stat.tstat;
        
        % Compare HC (0) vs SCZ follow-up (2)
        [~, pp2(disease_idx), ~, stat] = ttest2(...
            similarity(group == 0, disease_idx), ...
            similarity(group == 2, disease_idx));
        ttval2(disease_idx) = stat.tstat;
    end
end


% -------------------------------------------------------------------------
% Helper function: Calculate longitudinal comparison statistics
% -------------------------------------------------------------------------
function [tval_long, p_long] = calculate_longitudinal_statistics(similarity, group)
    num_diseases = size(similarity, 2);
    tval_long = zeros(num_diseases, 1);  % T-statistics (SCZ t1 vs t2)
    p_long = zeros(num_diseases, 1);     % P-values (SCZ t1 vs t2)
    
    % Extract longitudinal data (SCZ baseline vs follow-up)
    X1 = similarity(group == 1, :);  % SCZ at time 1
    X2 = similarity(group == 2, :);  % SCZ at time 2
    
    for disease_idx = 1:num_diseases
        [~, p_long(disease_idx), ~, stat] = ttest(X1(:, disease_idx), X2(:, disease_idx));
        tval_long(disease_idx) = stat.tstat;
    end
end


% -------------------------------------------------------------------------
% Helper function: Get column index for target disease
% -------------------------------------------------------------------------
function idx = get_disease_index(disease_name)
    switch lower(disease_name)
        case 'schizophrenia'
            idx = 8;
        case 'bipolar'
            idx = 4;
        case 'depression'
            idx = 5;
        case '22q'
            idx = 1;
        case 'ocd'
            idx = 7;
        otherwise
            error('Invalid disease name. Check disease list compatibility.');
    end
end


% -------------------------------------------------------------------------
% Helper function: Customize boxplot appearance
% -------------------------------------------------------------------------
function customize_boxplot(boxplot_handle, colors)
    % Customize box colors and transparency
    boxes = findobj(boxplot_handle, 'Tag', 'Box');
    for j = 1:length(boxes)
        patch(get(boxes(j), 'XData'), get(boxes(j), 'YData'), ...
            colors(j,:), 'FaceAlpha', 0.8, 'EdgeColor', colors(j,:));
    end
    
    % Customize median lines
    medians = findobj(boxplot_handle, 'Tag', 'Median');
    for j = 1:length(medians)
        medians(j).Color = colors(j,:);
    end
    
    % Customize whiskers
    whiskers = findobj(boxplot_handle, 'Tag', 'Whisker');
    set(whiskers, 'LineStyle', '-', 'Color', [0 0 0], 'Marker', 'none');
    
    % Customize outliers
    outliers = findobj(boxplot_handle, 'Tag', 'Outliers');
    set(outliers, 'Marker', 'o', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'w');
end
