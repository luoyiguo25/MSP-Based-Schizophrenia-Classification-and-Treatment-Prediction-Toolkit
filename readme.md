# MSP-Based Schizophrenia Classification and Treatment Prediction Toolkit

This project provides a comprehensive toolkit for analyzing brain imaging data and its relationship with psychiatric disorders, leveraging the ENIGMA dataset. It includes feature extraction, classification (SVM and other models), regression (Lasso), PLS analysis, ablation studies, and visualization scripts. The codebase supports both MATLAB and Python.

## Directory Structure

```
group_comparison_boxplot.m                # Group and longitudinal comparison boxplots
Lasso_CrossValidation.py                  # Lasso regression with nested cross-validation and visualization
construct_MSP/
    calc_similary.m                       # Calculate similarity (correlation) matrices
    extract_enigma_stat.m                 # Extract ENIGMA summary statistics
    extract_zscore.m                      # Extract and align z-score data
PLS/
    pls_result_analysis.m                 # PLS result analysis and visualization
    run_pls.m                             # Main PLS analysis with cross-validation
svm/
    brain_ablation_analysis.py            # SVM analysis with brain region ablation
    feature_ablation_analysis.py          # SVM analysis with feature ablation
    plot_brain_result.m                   # Visualize AUC changes after brain ablation
    plot_feature_results.py               # Visualize feature ablation results
    similarity_remove_brain_region.m      # Calculate similarity after removing each brain region
    svm.py                               # Main SVM classification script
```

## Main Features

- **Data Preprocessing & Feature Extraction**
  - Extract ENIGMA statistics and z-score data, align subject information.
  - Compute similarity matrices between subjects and disease templates.

- **Classification & Regression**
  - SVM and other models (logistic regression, decision tree, random forest, etc.) for classification, with metrics like AUC, accuracy, specificity, sensitivity.
  - Lasso regression with nested cross-validation for predicting clinical scores (e.g., PANSS).
  - PLS analysis for multivariate relationships between imaging and behavioral data.

- **Ablation Studies**
  - Brain region ablation: sequentially remove regions to assess impact on classification.
  - Feature ablation: sequentially remove disease features to assess impact.

- **Statistical Analysis & Visualization**
  - Boxplots for group and longitudinal comparisons.
  - Bar charts and brain atlas visualizations for ablation results.
  - Export results to CSV, SVG, PNG formats.

## Environment Requirements

- **MATLAB**: R2021a or newer.
- **Python**: 3.8 or newer, with `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `pandas`.

## Quick Start

1. **Prepare Data**
   - Place ENIGMA statistics and z-score files in `data/` and `centileBrain/` directories.
   - Run `construct_MSP/extract_enigma_stat.m` and `construct_MSP/extract_zscore.m` for preprocessing.

2. **Similarity Calculation**
   - Use `construct_MSP/calc_similary.m` or `svm/similarity_remove_brain_region.m` to compute similarity matrices.

3. **Classification/Regression**
   - SVM: Run `svm/svm.py` or ablation scripts.
   - Lasso: Run `Lasso_CrossValidation.py`.
   - PLS: Run `PLS/run_pls.m` and `PLS/pls_result_analysis.m`.

4. **Visualization**
   - Use `group_comparison_boxplot.m`, `svm/plot_feature_results.py`, `svm/plot_brain_result.m` for result visualization.

