import joblib
import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC


def load_data(dataset, dataname, time_point):
    """
    Load and preprocess data based on dataset type and time point
    
    Parameters:
        dataset (str): Name of the dataset
        dataname (str): Type of data ('SV', 'CT', or 'SV+CT')
        time_point (int): Time point to analyze (1 or 2)
    
    Returns:
        X (np.array): Feature matrix
        y (np.array): Label vector (0 for control, 1 for patient)
        disease (np.array): Array of disease names corresponding to features
    """
    # Load appropriate similarity data
    if dataname == 'SV+CT':
        data_sv = scipy.io.loadmat(f'similarity_{dataset}/similarity_SV.mat')
        data_ct = scipy.io.loadmat(f'similarity_{dataset}/similarity_CT.mat')
        similarity = np.concatenate((data_sv['similarity'], data_ct['similarity']), axis=1)
        data = data_sv  # Use SV data for metadata
    elif dataname in ['SV', 'CT']:
        data = scipy.io.loadmat(f'similarity_{dataset}/similarity_{dataname}.mat')
        similarity = data['similarity']
    else:
        raise ValueError(f"Invalid dataname: {dataname}. Supported types: 'SV', 'CT', 'SV+CT'")
    
    # Extract and process metadata
    group = data['group']
    disease = data['disease'].flatten()
    disease = np.array([str(item[0]) for item in disease])  # Convert to string array
    
    # Replace NaN values with 0
    similarity[np.isnan(similarity)] = 0
    
    # Select data for control (0) and specified time point (t)
    indices = (group == 0) | (group == time_point)
    X = similarity[indices.flatten(), :]
    y = group[indices.flatten()].ravel()  # Convert to 1D array
    
    # Standardize labels: 0 = control, 1 = patient
    y[y == 2] = 1
    
    return X, y, disease


def run_svm_classification(X, y, num_runs=100):
    """
    Run SVM classification with stratified k-fold cross-validation
    
    Parameters:
        X (np.array): Feature matrix
        y (np.array): Label vector
        num_runs (int): Number of independent runs
    
    Returns:
        dict: Dictionary containing performance metrics (mean and std)
    """
    # Initialize SVM model with RBF kernel
    model = SVC(kernel='rbf', probability=True)
    
    # Storage for performance metrics
    all_aucs = []
    all_accuracies = []
    all_specificities = []
    all_sensitivities = []
    
    # Fixed FPR values for ROC interpolation
    mean_fpr = np.linspace(0, 1, 100)
    
    for run in range(num_runs):
        # 5-fold stratified cross-validation with shuffling
        cv = StratifiedKFold(n_splits=5, shuffle=True)
        fold_aucs = []
        fold_accuracies = []
        fold_specificities = []
        fold_sensitivities = []
        
        for train_idx, test_idx in cv.split(X, y):
            # Train model
            model.fit(X[train_idx], y[train_idx])
            
            # Predictions and probabilities
            y_pred = model.predict(X[test_idx])
            y_proba = model.predict_proba(X[test_idx])[:, 1]
            
            # Calculate ROC and AUC
            fpr, tpr, _ = roc_curve(y[test_idx], y_proba)
            fold_aucs.append(auc(fpr, tpr))
            
            # Calculate accuracy
            fold_accuracies.append(accuracy_score(y[test_idx], y_pred))
            
            # Calculate specificity and sensitivity
            tn, fp, fn, tp = confusion_matrix(y[test_idx], y_pred).ravel()
            fold_specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
            fold_sensitivities.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        
        # Store average metrics for this run
        all_aucs.append(np.mean(fold_aucs))
        all_accuracies.append(np.mean(fold_accuracies))
        all_specificities.append(np.mean(fold_specificities))
        all_sensitivities.append(np.mean(fold_sensitivities))
    
    # Calculate overall statistics across all runs
    return {
        'AUC': (np.mean(all_aucs), np.std(all_aucs)),
        'Accuracy': (np.mean(all_accuracies), np.std(all_accuracies)),
        'Specificity': (np.mean(all_specificities), np.std(all_specificities)),
        'Sensitivity': (np.mean(all_sensitivities), np.std(all_sensitivities))
    }


def main():
    # Configuration
    time_point = 2
    dataset = 'ndyx'
    dataname = 'SV+CT'
    
    # Load and prepare data
    X, y, disease = load_data(dataset, dataname, time_point)
    num_features = X.shape[1]
    
    # Initialize results DataFrame
    results_df = pd.DataFrame(
        columns=['Feature Removed', 'AUC', 'Accuracy', 'Specificity', 'Sensitivity']
    )
    
    # Analyze each feature by removing it
    for feature_idx in range(num_features):
        # Determine feature type (SV or CT) and corresponding disease
        is_sv_feature = feature_idx < num_features // 2
        prefix = 'SV' if is_sv_feature else 'CT'
        disease_idx = feature_idx % (num_features // 2)
        feature_name = f'{prefix}_{disease[disease_idx]}'
        
        # Remove current feature
        X_reduced = np.delete(X, feature_idx, axis=1)
        
        # Run SVM classification
        print(f"Analyzing feature: {feature_name} (removed)")
        result = run_svm_classification(X_reduced, y, num_runs=100)
        
        # Store results
        results_df.loc[len(results_df)] = [
            feature_name,
            result['AUC'][0],
            result['Accuracy'][0],
            result['Specificity'][0],
            result['Sensitivity'][0]
        ]
    
    # Analyze with all features
    print("Analyzing with all features")
    result_full = run_svm_classification(X, y, num_runs=100)
    results_df.loc[len(results_df)] = [
        'All Features',
        result_full['AUC'][0],
        result_full['Accuracy'][0],
        result_full['Specificity'][0],
        result_full['Sensitivity'][0]
    ]
    
    # Save and display results
    output_file = f'analysis_features_contributions_t{time_point}.csv'
    results_df.to_csv(output_file, index=False)
    print("\nResults:")
    print(results_df)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
