from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix


# Configuration parameters
CONFIG = {
    "dataset_name": "ndyx",
    "data_type": "SV+CT",
    "time_point": 2,
    "num_runs": 100,
    "cv_folds": 5,
    "svm_kernel": "rbf",
    "result_csv_path": "analysis_brains_contributions_t{}.csv",
    "mat_file_dir": "similarity_{}"
}


def load_full_data(dataset: str, data_type: str, t: int) -> tuple[np.ndarray, np.ndarray]:
    """Load complete dataset without any brain region removed"""
    if data_type == "SV+CT":
        sv_data = scipy.io.loadmat(f"{CONFIG['mat_file_dir'].format(dataset)}/similarity_SV.mat")
        ct_data = scipy.io.loadmat(f"{CONFIG['mat_file_dir'].format(dataset)}/similarity_CT.mat")
        similarity = np.concatenate((sv_data["similarity"], ct_data["similarity"]), axis=1)
        group = sv_data["group"]
    else:
        data = scipy.io.loadmat(f"{CONFIG['mat_file_dir'].format(dataset)}/similarity_{data_type}.mat")
        similarity = data["similarity"]
        group = data["group"]

    # Handle missing values
    similarity[np.isnan(similarity)] = 0

    # Filter samples and prepare labels
    indices = (group == 0) | (group == t)
    X = similarity[indices.flatten(), :]
    y = group[indices.flatten()].ravel()
    y[y == 2] = 1  # Unify label format

    return X, y


def load_data_with_brain_removed(dataset: str, data_type: str, t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load dataset with individual brain region removal variants"""
    data = scipy.io.loadmat(
        f"{CONFIG['mat_file_dir'].format(dataset)}/similarity_removebrain_{data_type}.mat"
    )
    similarity = data["similarity"]
    group = data["group"]
    brain_raw = data["brain"].flatten()

    # Process brain region names
    brain_names = np.array([str(item[0]) for item in brain_raw])

    # Handle missing values
    similarity[np.isnan(similarity)] = 0

    # Filter samples and prepare labels
    indices = (group == 0) | (group == t)
    X = similarity[indices.flatten(), :, :]
    y = group[indices.flatten()].ravel()
    y[y == 2] = 1  # Unify label format

    return X, y, brain_names


def train_svm_with_evaluation(X: np.ndarray, y: np.ndarray) -> dict[str, tuple[float, float]]:
    """Train SVM model and evaluate multiple metrics"""
    svm_model = SVC(kernel=CONFIG["svm_kernel"], probability=True, random_state=42)
    
    # Storage for metrics across runs
    all_aucs = []
    all_accs = []
    all_specs = []
    all_sens = []

    for _ in range(CONFIG["num_runs"]):
        cv = StratifiedKFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=np.random.randint(1000))
        fold_aucs = []
        fold_accs = []
        fold_specs = []
        fold_sens = []

        for train_idx, test_idx in cv.split(X, y):
            # Prepare data splits
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Model training and prediction
            svm_model.fit(X_train, y_train)
            y_pred = svm_model.predict(X_test)
            y_proba = svm_model.predict_proba(X_test)[:, 1]

            # Calculate AUC
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fold_aucs.append(auc(fpr, tpr))

            # Calculate accuracy
            fold_accs.append(accuracy_score(y_test, y_pred))

            # Calculate specificity and sensitivity
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            fold_specs.append(tn / (tn + fp) if (tn + fp) != 0 else 0.0)
            fold_sens.append(tp / (tp + fn) if (tp + fn) != 0 else 0.0)

        # Store average metrics for this run
        all_aucs.append(np.mean(fold_aucs))
        all_accs.append(np.mean(fold_accs))
        all_specs.append(np.mean(fold_specs))
        all_sens.append(np.mean(fold_sens))

    # Calculate overall statistics
    return {
        "AUC": (np.mean(all_aucs), np.std(all_aucs)),
        "Accuracy": (np.mean(all_accs), np.std(all_accs)),
        "Specificity": (np.mean(all_specs), np.std(all_specs)),
        "Sensitivity": (np.mean(all_sens), np.std(all_sens))
    }


def main():
    # Load ablation data
    X_ablation, y_ablation, brain_names = load_data_with_brain_removed(
        dataset=CONFIG["dataset_name"],
        data_type=CONFIG["data_type"],
        t=CONFIG["time_point"]
    )
    
    # Load full data for baseline
    X_full, y_full = load_full_data(
        dataset=CONFIG["dataset_name"],
        data_type=CONFIG["data_type"],
        t=CONFIG["time_point"]
    )

    # Initialize results DataFrame
    results_df = pd.DataFrame(
        columns=["Brain Removed", "AUC", "Accuracy", "Specificity", "Sensitivity"]
    )

    # Evaluate each brain region removal
    for i, brain in enumerate(brain_names):
        print(f"Processing brain {i+1}/{len(brain_names)}: {brain}")
        X_current = X_ablation[:, :, i]
        metrics = train_svm_with_evaluation(X_current, y_ablation)
        
        results_df.loc[len(results_df)] = [
            brain,
            metrics["AUC"][0],
            metrics["Accuracy"][0],
            metrics["Specificity"][0],
            metrics["Sensitivity"][0]
        ]

    # Evaluate full model (all brains)
    print("Processing full model with all brains")
    baseline_metrics = train_svm_with_evaluation(X_full, y_full)
    results_df.loc[len(results_df)] = [
        "All Brains",
        baseline_metrics["AUC"][0],
        baseline_metrics["Accuracy"][0],
        baseline_metrics["Specificity"][0],
        baseline_metrics["Sensitivity"][0]
    ]

    # Save results
    csv_path = CONFIG["result_csv_path"].format(CONFIG["time_point"])
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    print("\nFinal Results:")
    print(results_df.round(4))


if __name__ == "__main__":
    main()
    