import joblib
import scipy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix


def load_data(dataset, dataname, t):
    """
    Load and preprocess data from .mat files
    
    Parameters:
    dataset (str): Name of the dataset
    dataname (str): Type of data to load ('SV', 'CT', or 'SV+CT')
    t (int): Time point identifier (1 or 2)
    
    Returns:
    tuple: (X, y) where X is the feature matrix and y is the target vector
    """

    if dataname == 'SV+CT':
        data_sv = scipy.io.loadmat(f'similarity_{dataset}/similarity_SV.mat')
        data_ct = scipy.io.loadmat(f'similarity_{dataset}/similarity_CT.mat')
        similarity = np.concatenate((data_sv['similarity'], data_ct['similarity']), 1)
    elif dataname in ['SV', 'CT']:
        data = scipy.io.loadmat(f'similarity_{dataset}/similarity_{dataname}.mat')
        similarity = data['similarity']
    else:
        raise ValueError(f"Invalid dataname: {dataname}. Choose 'SV', 'CT', or 'SV+CT'")
    
    group = data['group'] if dataname != 'SV+CT' else data_sv['group']
    rate = data['rate'] if dataname != 'SV+CT' else data_sv['rate']

    similarity[np.isnan(similarity)] = 0

    indices = (group == 0) | (group == t)
    X = similarity[indices.flatten(), :]
    y = group[indices.flatten()]
    
    y = y.ravel()
    y[y == 2] = 1

    return (X, y)


def run_classification(model_name, X, y, time_label, dataname, num_runs=100):
    """
    Run classification with specified model using 5-fold stratified cross-validation
    repeated for a specified number of runs, and generate ROC curves.
    
    Parameters:
    model_name (str): Name of the classification model to use
    X (numpy.ndarray): Feature matrix
    y (numpy.ndarray): Target vector
    time_label (str): Label for time point (e.g., 'Baseline', 'Followup')
    dataname (str): Type of data being processed
    num_runs (int): Number of times to repeat cross-validation
    
    Returns:
    dict: Dictionary containing performance metrics (AUC, Accuracy, Specificity, Sensitivity)
          with their mean and standard deviation
    """
    if model_name == 'svm':
        model = SVC(kernel='rbf', probability=True)
    elif model_name == 'logistic':
        model = LogisticRegression(max_iter=1000)
    elif model_name == 'decision_tree':
        model = DecisionTreeClassifier()
    elif model_name == 'random_forest':
        model = RandomForestClassifier()
    elif model_name == 'bayes':
        model = GaussianNB()
    elif model_name == 'gradient_boosting':
        model = GradientBoostingClassifier()
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    all_aucs = []
    all_accuracies = []
    all_specificities = []
    all_sensitivities = []
    mean_tprs = []

    if dataname == 'SV+CT':
        plt.figure(figsize=(6.5/2.54, 6/2.54))
    else:
        plt.figure(figsize=(3/2.54, 2.5/2.54))

    mean_fpr = np.linspace(0, 1, 100)

    for run in range(num_runs):
        cv = StratifiedKFold(n_splits=5, shuffle=True)
        aucs = []
        accuracies = []
        specificities = []
        sensitivities = []
        tprs = []

        for train_idx, test_idx in cv.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            
            y_pred = model.predict(X[test_idx])
            probas_ = model.predict_proba(X[test_idx])

            fpr, tpr, thresholds = roc_curve(y[test_idx], probas_[:, 1])
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

            accuracy = accuracy_score(y[test_idx], y_pred)
            accuracies.append(accuracy)

            tn, fp, fn, tp = confusion_matrix(y[test_idx], y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificities.append(specificity)
            sensitivities.append(sensitivity)

        mean_tprs.append(np.mean(tprs, axis=0))
        all_aucs.append(np.mean(aucs))
        all_accuracies.append(np.mean(accuracies))
        all_specificities.append(np.mean(specificities))
        all_sensitivities.append(np.mean(sensitivities))

    mean_tpr = np.mean(mean_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(all_aucs)
    
    std_tpr = np.std(mean_tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    label_auc = f'AUC:{mean_auc:.2f}±{std_auc:.2f}'
    plt.plot(mean_fpr, mean_tpr, color='#5b81a6', lw=1.5, alpha=1, label=label_auc)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='#BFBFBF', alpha=0.3)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='gray', alpha=.5, label='Chance')

    plt.rcParams.update({
        'legend.fontsize': 6,
        'legend.handlelength': 2,
        'font.family': 'Arial'
    })

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    if dataname != 'SV+CT':
        plt.yticks([0, 0.5, 1])
    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=8)
    plt.xlabel('False Positive Rate', fontsize=8)
    plt.ylabel('True Positive Rate', fontsize=8)
    plt.legend(loc="lower right")
    
    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig(f'results/SVM/{time_label}_{dataname}.svg')
    plt.show()

    print(f"{time_label}:")
    print(f'Average AUC ({num_runs} runs) = {mean_auc:.2f} ± {std_auc:.2f}')
    print(f'Average Accuracy ({num_runs} runs): {np.mean(all_accuracies):.2f} ± {np.std(all_accuracies):.2f}')
    print(f'Average Specificity ({num_runs} runs): {np.mean(all_specificities):.2f} ± {np.std(all_specificities):.2f}')
    print(f'Average Sensitivity ({num_runs} runs): {np.mean(all_sensitivities):.2f} ± {np.std(all_sensitivities):.2f}')

    return {
        'AUC': (mean_auc, std_auc),
        'Accuracy': (np.mean(all_accuracies), np.std(all_accuracies)),
        'Specificity': (np.mean(all_specificities), np.std(all_specificities)),
        'Sensitivity': (np.mean(all_sensitivities), np.std(all_sensitivities))
    }


if __name__ == "__main__":
    dataname = 'CT'
    X, y = load_data('ndyx', dataname, 1)
    run_classification('svm', X, y, 'Baseline', dataname, 100)
    
    X, y = load_data('ndyx', dataname, 2)
    run_classification('svm', X, y, 'Followup', dataname, 100)

    dataname = 'SV'
    X, y = load_data('ndyx', dataname, 1)
    run_classification('svm', X, y, 'Baseline', dataname, 100)
    
    X, y = load_data('ndyx', dataname, 2)
    run_classification('svm', X, y, 'Followup', dataname, 100)

    dataname = 'SV+CT'
    X, y = load_data('ndyx', dataname, 1)
    run_classification('svm', X, y, 'Baseline', dataname, 100)
    
    X, y = load_data('ndyx', dataname, 2)
    run_classification('svm', X, y, 'Followup', dataname, 100)

    # Uncomment to run other classifiers
    # run_classification('logistic', X, y, 'Followup', dataname, 100)
    # run_classification('decision_tree', X, y, 'Followup', dataname, 100)
    # run_classification('random_forest', X, y, 'Followup', dataname, 100)
    # run_classification('bayes', X, y, 'Followup', dataname, 100)
    # run_classification('gradient_boosting', X, y, 'Followup', dataname, 100)
