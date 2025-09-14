import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, t
from collections import Counter


# Configuration parameters: contains all data types, time points, and target settings
CONFIG = {
    "dataset": "ndyx",
    "data_types": ["SV", "CT", "SV+CT"],  # All data types
    "feature_type": "similarity",         # Type of feature
    "time_points": [1, 2],                # All time points
    "targets": {                          # All prediction targets with their lambda ranges
        "rate": {"SV": (40, 50), "CT": (10, 20), "SV+CT": (65, 75)},
        "t1panss": {"SV": (60, 70), "CT": (20, 30), "SV+CT": (30, 50)},
        "t2panss": {"SV": (50, 60), "CT": (1, 5), "SV+CT": (30, 50)},
        "Panss difference": {"SV": (50, 70), "CT": (30, 40), "SV+CT": (60, 80)}
    },
    "n_repeats": 100,      # Number of cross-validation repetitions
    "outer_folds": 5,      # Number of outer cross-validation folds
    "inner_folds": 5,      # Number of inner cross-validation folds
    "max_iter": 10000,     # Maximum iterations for Lasso
    "figure_size": (6.5/2.54, 5/2.54)  # Figure size in inches
}


def load_data(dataset: str, dataname: str, feature_type: str, t: int) -> tuple:
    """Load data: retrieve feature matrix and clinical metrics based on data type, feature type, and time point"""
    # Load combined SV+CT data
    if dataname == "SV+CT":
        data_sv = scipy.io.loadmat(f"similarity_{dataset}/{feature_type}_SV.mat")
        data_ct = scipy.io.loadmat(f"similarity_{dataset}/{feature_type}_CT.mat")
        X = np.concatenate((data_sv[feature_type], data_ct[feature_type]), axis=1)
        data = data_sv  # Share metadata
    else:  # Load individual SV or CT data
        data = scipy.io.loadmat(f"similarity_{dataset}/{feature_type}_{dataname}.mat")
        X = data[feature_type]
    
    # Extract clinical metrics
    group = data["group"]
    rate = data["rate"]
    t1panss = data["t1panss"]
    t2panss = data["t2panss"]
    diffpanss = data["diffpanss"]
    
    # Handle missing values
    X[np.isnan(X)] = 0
    
    # Filter samples and clinical metrics for specified time point
    indices = (group == t).flatten()
    indices_clinic = (group == 1).flatten()  # Clinical metrics are in group 1
    
    return (
        X[indices, :],                # Feature matrix
        group[indices].flatten(),     # Group labels
        rate[indices_clinic].flatten(),       # Rate metric
        t1panss[indices_clinic].flatten(),    # T1 PANSS metric
        t2panss[indices_clinic].flatten(),    # T2 PANSS metric
        diffpanss[indices_clinic].flatten()   # PANSS difference metric
    )


def nested_cv_lasso(X: np.ndarray, y: np.ndarray, yname: str, lambda_low: int, lambda_high: int) -> None:
    """Nested cross-validation for Lasso regression: with hyperparameter tuning and result visualization"""
    y = y.ravel()  # Ensure target variable is 1D array
    lambda_values = np.linspace(lambda_low/100, lambda_high/100, 100)  # Lambda search range
    
    # Initial lambda estimation
    scaler_init = StandardScaler()
    X_init_scaled = scaler_init.fit_transform(X)
    lasso_cv = LassoCV(alphas=np.logspace(-6, 6, 100), cv=5, max_iter=CONFIG["max_iter"])
    lasso_cv.fit(X_init_scaled, y)
    
    print(f"Target: {yname}")
    print(f"Initial best lambda estimate: {lasso_cv.alpha_:.6f}")
    print(f"Lambda search range: {lambda_low/100:.2f} to {lambda_high/100:.2f}\n")
    
    # Store results
    y_real_all = []
    y_pred_all = []
    best_lambdas = []
    
    # Nested cross-validation
    for _ in range(CONFIG["n_repeats"]):
        y_real_fold = []
        y_pred_fold = []
        
        # Outer cross-validation
        outer_cv = KFold(n_splits=CONFIG["outer_folds"], shuffle=True, random_state=np.random.randint(1000))
        
        for train_idx, test_idx in outer_cv.split(X):
            # Split into training and testing sets
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Feature standardization
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Inner cross-validation for lambda tuning
            inner_cv = KFold(n_splits=CONFIG["inner_folds"], shuffle=True, random_state=np.random.randint(1000))
            grid_search = GridSearchCV(
                estimator=Lasso(max_iter=CONFIG["max_iter"]),
                param_grid={"alpha": lambda_values},
                cv=inner_cv,
                scoring="neg_mean_squared_error"
            )
            grid_search.fit(X_train_scaled, y_train)
            
            # Save best lambda and make predictions
            best_lambda = grid_search.best_params_["alpha"]
            best_lambdas.append(best_lambda)
            
            model = Lasso(alpha=best_lambda, max_iter=CONFIG["max_iter"])
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            y_real_fold.extend(y_test)
            y_pred_fold.extend(y_pred)
        
        y_real_all.append(y_real_fold)
        y_pred_all.append(y_pred_fold)
    
    # Aggregate results
    y_real_all = np.array(y_real_all)
    y_pred_all = np.array(y_pred_all)
    y_real_mean = y_real_all.mean(axis=0)
    y_pred_mean = y_pred_all.mean(axis=0)
    
    # Calculate correlation
    r, p = pearsonr(y_real_mean, y_pred_mean)
    
    # Regression line and confidence interval
    slope, intercept = np.polyfit(y_real_mean, y_pred_mean, 1)
    y_fit = slope * y_real_mean + intercept
    residuals = y_pred_mean - y_fit
    n = len(y_real_mean)
    t_value = t.ppf(0.975, df=n-2)
    mean_x = np.mean(y_real_mean)
    standard_error = np.sqrt(np.sum(residuals**2)/(n-2) / np.sum((y_real_mean - mean_x)** 2))
    y_fit_upper = y_fit + t_value * standard_error
    y_fit_lower = y_fit - t_value * standard_error
    
    # Visualization
    plt.figure(figsize=CONFIG["figure_size"])
    plt.scatter(y_real_mean, y_pred_mean, edgecolors="#2e7bb8", s=4)
    plt.plot(y_real_mean, y_fit, color="#2e7bb8")
    plt.fill_between(y_real_mean, y_fit_lower, y_fit_upper, color="#2e7bb8", alpha=0.2)
    
    plt.xlabel(yname, fontsize=8)
    plt.ylabel(f"Predicted {yname}", fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    
    # Remove right and top spines
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    
    # Add correlation results
    plt.text(0.95, 0.05, f"r = {r:.2f}\np = {p:.2g}", 
             verticalalignment="bottom", horizontalalignment="right", 
             transform=plt.gca().transAxes, fontsize=8)
    
    plt.rcParams["svg.fonttype"] = "none"
    plt.tight_layout()
    plt.show()
    
    # Lambda statistics
    lambda_counter = Counter(best_lambdas)
    most_common_lambda, count = lambda_counter.most_common(1)[0]
    
    print(f"Pearson correlation coefficient: {r:.4f}, p-value: {p:.4g}")
    print(f"Most frequent lambda: {most_common_lambda:.6f} (count: {count}/{CONFIG['n_repeats']*CONFIG['outer_folds']})\n")


def run_all_analyses():
    """Run all combinations of analyses: iterate through data types, time points, and prediction targets"""
    # Iterate through all data types (SV, CT, SV+CT)
    for data_type in CONFIG["data_types"]:
        print(f"\n===== Data type: {data_type} =====")
        
        # Iterate through all time points (t=1 and t=2)
        for t in CONFIG["time_points"]:
            print(f"\n----- Time point: t={t} -----")
            
            # Load data for current combination
            X, _, rate, t1panss, t2panss, diffpanss = load_data(
                dataset=CONFIG["dataset"],
                dataname=data_type,
                feature_type=CONFIG["feature_type"],
                t=t
            )
            
            # Run Lasso regression for all prediction targets
            # 1. rate
            lambda_low, lambda_high = CONFIG["targets"]["rate"][data_type]
            nested_cv_lasso(X, rate, "rate", lambda_low, lambda_high)
            
            # 2. t1panss
            lambda_low, lambda_high = CONFIG["targets"]["t1panss"][data_type]
            nested_cv_lasso(X, t1panss, "t1panss", lambda_low, lambda_high)
            
            # 3. t2panss
            lambda_low, lambda_high = CONFIG["targets"]["t2panss"][data_type]
            nested_cv_lasso(X, t2panss, "t2panss", lambda_low, lambda_high)
            
            # 4. Panss difference
            lambda_low, lambda_high = CONFIG["targets"]["Panss difference"][data_type]
            nested_cv_lasso(X, diffpanss, "Panss difference", lambda_low, lambda_high)


if __name__ == "__main__":
    run_all_analyses()
    