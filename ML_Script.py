#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_ml_models.py

Models:
Classification: Logistic Regression (Elastic Net), SVM (RBF), Random Forest, XGBoost, Gaussian Process, ANN
Regression: Elastic Net, SVR, Random Forest, XGBoost, Gaussian Process, ANN

Outputs:
- Train/Test predictions
- Performance metrics (AUC/Accuracy or RMSE/R2)
- Hyperparameters
- Permutation importance
- SHAP values
"""
import os
import argparse
import numpy as np
import pandas as pd
import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score, roc_curve
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier, XGBRegressor
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.stats.outliers_influence import variance_inflation_factor
from collections import Counter
from sklearn.base import clone

from scipy import stats


# -------------------------------
# Argument Parsing
# -------------------------------
parser = argparse.ArgumentParser(description="All-in-one ML script")
parser.add_argument("--data_X", type=str, required=True, help="CSV of features")
parser.add_argument("--data_y", type=str, required=True, help="CSV of labels")
parser.add_argument("--task", type=str, choices=["classification","regression"], required=True)
parser.add_argument("--output_dir", type=str, default="results")
parser.add_argument("--random_state", type=int, default=42)
parser.add_argument("--test_size_list", nargs="+", type=float, default=[0.2,0.25,0.3,0.33,0.4,0.5])
parser.add_argument("--CV_Outer_Split",type=int, default=5)
parser.add_argument("--CV_Inner_Split",type=int, default=5)
parser.add_argument("--Corr_Threshold",type=float, default=0.9)
args = parser.parse_args()

# Create output directory if it does not exist
output_path = Path(args.output_dir)

try:
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory ready: {output_path.resolve()}")
except Exception as e:
    raise RuntimeError(f"Could not create output directory: {e}")

os.makedirs(args.output_dir, exist_ok=True)

class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=args.Corr_Threshold):
        self.threshold = threshold

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        corr = X_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.to_drop_ = [column for column in upper.columns if any(upper[column] > self.threshold)]
        self.features_kept_ = [c for c in X_df.columns if c not in self.to_drop_]
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        return X_df.drop(columns=self.to_drop_, errors="ignore")

# -------------------------------
# Load Data
# -------------------------------
X = pd.read_csv(args.data_X, index_col=0)
y = pd.read_csv(args.data_y, index_col=0).iloc[:,0]  # single column

# ---------------------------------
# Correlation Matrix Export
# ---------------------------------
corr_matrix = X.corr(method="spearman")
corr_matrix.to_csv(os.path.join(args.output_dir, "FULL_DATA_CORRELATION_MATRIX.csv"))

# ---------------------------------
# VIF Analysis
# ---------------------------------
def compute_vif(X):
    X_df = pd.DataFrame(X)
    vif_data = []
    for i in range(X_df.shape[1]):
        vif_data.append({
            "Feature": X_df.columns[i],
            "VIF": variance_inflation_factor(X_df.values, i)
        })
    return pd.DataFrame(vif_data)

vif_df = compute_vif(X)
vif_df.to_csv(os.path.join(args.output_dir, "VIF_ANALYSIS.csv"), index=False)

# -------------------------------
# Function: Train-test split optimization
# -------------------------------

def optimize_split(X, y, task, test_sizes, random_state):
    best_score = -np.inf
    best_split = None
    for ts in test_sizes:
        if task=="classification":
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=ts, stratify=y, random_state=random_state)
            score = roc_auc_score(y_te, np.full(len(y_te), y_tr.value_counts(normalize=True).idxmax()))
        else:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=ts, random_state=random_state)
            score = -mean_squared_error(y_te, np.full(len(y_te), y_tr.mean()))
        if score > best_score:
            best_score = score
            best_split = (X_tr, X_te, y_tr, y_te)
    return best_split

def bootstrap_roc_ci(y_true, y_prob, n_bootstraps=2000, seed=42):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    
    rng = np.random.RandomState(seed)
    aucs = []

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))

        # Skip invalid resamples
        if len(np.unique(y_true[indices])) < 2:
            continue

        auc = roc_auc_score(y_true[indices], y_prob[indices])
        aucs.append(auc)

    aucs = np.array(aucs)

    return (
        aucs.mean(),
        np.percentile(aucs, 2.5),
        np.percentile(aucs, 97.5),
        aucs
    )





def decision_curve_analysis(y_true, y_prob, model_name,
                            thresholds=np.linspace(0.01, 0.99, 99)):
    """
    Compute net benefit across probability thresholds.
    Returns a DataFrame with threshold, net benefit, treat-all, treat-none.
    """

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    N = len(y_true)
    prevalence = np.mean(y_true)

    rows = []

    for pt in thresholds:
        preds = (y_prob >= pt).astype(int)

        TP = np.sum((preds == 1) & (y_true == 1))
        FP = np.sum((preds == 1) & (y_true == 0))

        # Net Benefit formula
        net_benefit = (TP / N) - (FP / N) * (pt / (1 - pt))

        # Treat All strategy
        treat_all = prevalence - (1 - prevalence) * (pt / (1 - pt))

        # Treat None strategy
        treat_none = 0

        rows.append({
            "Model": model_name,
            "Threshold": pt,
            "Net_Benefit": net_benefit,
            "Treat_All": treat_all,
            "Treat_None": treat_none
        })

    return pd.DataFrame(rows)

def backward_feature_elimination(
    model_pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
    feature_importance,
    task="classification"
):
    """
    Iteratively removes lowest-importance features and retrains model.
    Returns performance table and best feature subset.
    """

    # Rank features highest → lowest
    ranked_features = (
        feature_importance
        .sort_values("Importance", ascending=False)["Feature"]
        .tolist()
    )

    results = []

    for k in range(len(ranked_features), 0, -1):

        selected_features = ranked_features[:k]

        X_tr_sub = X_train[selected_features]
        X_te_sub = X_test[selected_features]

        model = clone(model_pipeline)
        model.fit(X_tr_sub, y_train)

        if task == "classification":
            y_prob = model.predict_proba(X_te_sub)[:, 1]
            score = roc_auc_score(y_test, y_prob)

        else:
            y_pred = model.predict(X_te_sub)
            score = r2_score(y_test, y_pred)

        results.append({
            "Num_Features": k,
            "Score": score,
            "Features": selected_features
        })

    results_df = pd.DataFrame(results)

    # Identify best breakpoint
    best_row = results_df.loc[results_df["Score"].idxmax()]

    return results_df, best_row

# ---- DeLong Implementation ---- #

def compute_midrank(x):
    """Computes midranks."""
    sorted_idx = np.argsort(x)
    sorted_x = x[sorted_idx]
    N = len(x)
    midranks = np.zeros(N)

    i = 0
    while i < N:
        j = i
        while j < N and sorted_x[j] == sorted_x[i]:
            j += 1
        midrank = 0.5 * (i + j - 1)
        midranks[i:j] = midrank
        i = j

    out = np.empty(N)
    out[sorted_idx] = midranks + 1  # 1-based indexing
    return out


def fast_delong(predictions_sorted_transposed, label_1_count):
    """Fast DeLong implementation."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m

    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]

    k = predictions_sorted_transposed.shape[0]

    tx = np.array([compute_midrank(x) for x in positive_examples])
    ty = np.array([compute_midrank(x) for x in negative_examples])
    tz = np.array([compute_midrank(x) for x in predictions_sorted_transposed])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)

    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.cov(v01)
    sy = np.cov(v10)

    delong_cov = sx / m + sy / n

    return aucs, delong_cov


def delong_roc_test(y_true, y_pred1, y_pred2):
    """
    Returns p-value for difference in AUC between two models.
    """
    y_true = np.array(y_true)
    order = np.argsort(-y_true)
    y_true = y_true[order]
    y_pred1 = np.array(y_pred1)[order]
    y_pred2 = np.array(y_pred2)[order]

    label_1_count = int(np.sum(y_true))

    preds = np.vstack((y_pred1, y_pred2))
    aucs, delong_cov = fast_delong(preds, label_1_count)

    diff = aucs[0] - aucs[1]
    var = delong_cov[0, 0] + delong_cov[1, 1] - 2 * delong_cov[0, 1]
    z = diff / np.sqrt(var)
    p_value = 2 * stats.norm.sf(abs(z))

    return aucs[0], aucs[1], diff, p_value


X_train, X_test, y_train, y_test = optimize_split(X, y, args.task, args.test_size_list, args.random_state)
print(f"Optimal train-test split: {len(X_train)} train / {len(X_test)} test")



# -------------------------------
# Define models and parameter grids
# -------------------------------
models = {}
if args.task=="classification":
    models["LogisticRegression_EN"] = (
        Pipeline([("corr_filter", CorrelationFilter(threshold=args.Corr_Threshold)),
                  ("scaler", StandardScaler()), ("lr", LogisticRegression(penalty="elasticnet", solver="saga", max_iter=5000, random_state=args.random_state))]),
        {"lr__C":[0.01,0.1,1.0], "lr__l1_ratio":[0.2,0.5,0.8]}
    )
    models["SVM_RBF"] = (
        Pipeline([("corr_filter", CorrelationFilter(threshold=args.Corr_Threshold)),
            ("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", probability=True, random_state=args.random_state))]),
        {"svm__C":[0.1,1,10], "svm__gamma":["scale","auto"]}
    )
    models["RandomForest"] = (
        RandomForestClassifier(random_state=args.random_state),
        {"n_estimators":[100,500,1000,5000], "max_depth":[None,10,20], "min_samples_split":[2,5], "min_samples_leaf":[None,1,2], "max_features":["sqrt","log2"]}
    )
    models["XGBoost"] = (
        Pipeline([("corr_filter", CorrelationFilter(threshold=args.Corr_Threshold)),
                  ("xgb", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=args.random_state))]),
        {"xgb__n_estimators":[100,500,1000,5000], "xgb__max_depth":[3,6], "xgb__learning_rate":[0.01,0.1,0.3]}
    )
    models["GaussianProcess"] = (GaussianProcessClassifier(random_state=args.random_state), {})
    models["ANN"] = (
        Pipeline([("corr_filter", CorrelationFilter(threshold=args.Corr_Threshold)),
                  ("scaler", StandardScaler()), ("mlp", MLPClassifier(max_iter=2000, random_state=args.random_state))]),
        {"mlp__hidden_layer_sizes":[(10,),(20,), (50,), (10, 5),(20,5),(20,10)], "mlp__activation":["relu","tanh"], "mlp__alpha":[0.0001,0.001,0.01], "mlp__learning_rate_init":[0.001,0.01]}
    )
else:
    models["ElasticNet"] = (
        Pipeline([("corr_filter", CorrelationFilter(threshold=args.Corr_Threshold)),
                  ("scaler", StandardScaler()), ("en", ElasticNet(random_state=args.random_state))]),
        {"en__alpha":[0.01,0.1,1.0], "en__l1_ratio":[0.2,0.5,0.8]}
    )
    models["SVR"] = (
        Pipeline([("corr_filter", CorrelationFilter(threshold=args.Corr_Threshold)),
                  ("scaler", StandardScaler()), ("svr", SVR())]),
        {"svr__C":[0.1,1,10], "svr__gamma":["scale","auto"]}
    )
    models["RandomForest"] = (
        RandomForestRegressor(random_state=args.random_state),
        {"n_estimators":[100,500,1000,5000], "max_depth":[None,10,20], "min_samples_split":[2,5], "min_samples_leaf":[None,1,2], "max_features":["sqrt","log2"]}
    )
    models["XGBoost"] = (
        Pipeline([("corr_filter", CorrelationFilter(threshold=args.Corr_Threshold)),
                  ("xgb", XGBRegressor(random_state=args.random_state))]),
        {"xgb__n_estimators":[100,500,1000,5000], "xgb__max_depth":[3,6], "xgb__learning_rate":[0.01,0.1,0.3]}
    )
    models["GaussianProcess"] = (GaussianProcessRegressor(), {})
    models["ANN"] = (
        Pipeline([("corr_filter", CorrelationFilter(threshold=args.Corr_Threshold)),
                  ("scaler", StandardScaler()), ("mlp", MLPRegressor(max_iter=2000, random_state=args.random_state))]),
        {"mlp__hidden_layer_sizes":[(10,),(20,), (50,), (10, 5),(20,5),(20,10)], "mlp__activation":["relu","tanh"], "mlp__alpha":[0.0001,0.001,0.01], "mlp__learning_rate_init":[0.001,0.01]}
    )

# -------------------------------
# Nested CV + Training
# -------------------------------
cv_outer = StratifiedKFold(n_splits=args.CV_Outer_Split, shuffle=True, random_state=args.random_state) if args.task=="classification" else KFold(n_splits=5, shuffle=True, random_state=args.random_state)

# -------------------------------
# Collect all model performance
# -------------------------------
all_results = []
feature_tracking = []
shap_tracking = []
all_roc_rows = []
fitted_models_dict = {}
importance_dict = {}
model_predictions={}

for name, (model, param_grid) in models.items():
    print(f"\n--- Training {name} ---")
    
    cv_inner = StratifiedKFold(n_splits=args.CV_Inner_Split, shuffle=True, random_state=args.random_state) if args.task=="classification" else KFold(n_splits=3, shuffle=True, random_state=args.random_state)
    grid = GridSearchCV(model, param_grid, cv=cv_inner, scoring="roc_auc" if args.task=="classification" else "neg_root_mean_squared_error", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    fitted_models_dict[name] = grid.best_estimator_
    # Save best hyperparameters
    pd.DataFrame([grid.best_params_]).to_csv(os.path.join(args.output_dir,f"{name}_best_params.csv"), index=False)
    
    # Fit final model
    best_model.fit(X_train, y_train)
    if hasattr(best_model, "named_steps") and "corr_filter" in best_model.named_steps:
        kept = best_model.named_steps["corr_filter"].features_kept_
    else:
        kept = X.columns.tolist()
    
    feature_tracking.append({
        "Model": name,
        "Features_Kept": kept
    })
    
    # Predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    y_train_prob = best_model.predict_proba(X_train)[:,1] if args.task=="classification" else y_train_pred
    y_test_prob = best_model.predict_proba(X_test)[:,1] if args.task=="classification" else y_test_pred
    model_predictions[name+'_test']=y_test_prob
    model_predictions[name+'_train']=y_train_prob
    # Save predictions
    pd.DataFrame({"sample_index": X_train.index, "y_train": y_train, "y_pred_class": y_train_pred if args.task=="classification" else np.nan, "y_pred_prob": y_train_prob}).to_csv(os.path.join(args.output_dir,f"{name}_train_predictions.csv"), index=False)
    pd.DataFrame({"sample_index": X_test.index, "y_test": y_test, "y_pred_class": y_test_pred if args.task=="classification" else np.nan, "y_pred_prob": y_test_prob}).to_csv(os.path.join(args.output_dir,f"{name}_test_predictions.csv"), index=False)
    
    # -------------------------------
    # Performance (Train + Test)
    # -------------------------------
    if args.task=="classification":

        # ---- Train performance ----
        train_auc = roc_auc_score(y_train, y_train_prob)
        train_acc = accuracy_score(y_train, y_train_pred)

        # ---- Test performance ----
        test_auc = roc_auc_score(y_test, y_test_prob)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        # ROC Curve Export
        fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
        test_group=["test" for _ in fpr]
        tr_fpr, tr_tpr, tr_thresholds = roc_curve(y_train, y_train_prob)
        train_group=["train" for _ in tr_fpr]
        roc_df = pd.DataFrame({
            "Model": name,
            "FPR": fpr,
            "TPR": tpr,
            "Threshold": thresholds,
            "Group": test_group
        })
        train_roc_df = pd.DataFrame({
            "Model": name,
            "FPR": tr_fpr,
            "TPR": tr_tpr,
            "Threshold": tr_thresholds,
            "Group": train_group
        })
        
        roc_df.to_csv(os.path.join(args.output_dir, f"{name}_ROC.csv"), index=False)
        train_roc_df.to_csv(os.path.join(args.output_dir, f"{name}_Training_ROC.csv"), index=False)
        all_roc_rows.append(roc_df)
        all_roc_rows.append(train_roc_df)
        
        # Bootstrap CI
        auc_mean, ci_low, ci_high, auc_dist = bootstrap_roc_ci(y_test, y_test_prob)
        
        with open(os.path.join(args.output_dir, f"{name}_Test_AUC_CI.txt"), "w") as f:
            f.write(f"AUC: {auc_mean:.4f}\n")
            f.write(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]\n")
        
        train_auc_mean, train_ci_low, train_ci_high, train_auc_dist = bootstrap_roc_ci(y_train, y_train_prob)
        
        with open(os.path.join(args.output_dir, f"{name}_Train_AUC_CI.txt"), "w") as f:
            f.write(f"AUC: {train_auc_mean:.4f}\n")
            f.write(f"95% CI: [{train_ci_low:.4f}, {train_ci_high:.4f}]\n")

        print(f"{name} Train AUC: {train_auc:.4f}, Accuracy: {train_acc:.4f}")
        print(f"{name} Test  AUC: {test_auc:.4f}, Accuracy: {test_acc:.4f}")

        perf_dict = {
            "Model": name,
            "Train_AUC": train_auc,
            "Train_Accuracy": train_acc,
            "Test_AUC": test_auc,
            "Test_Accuracy": test_acc
        }
        # Decision Curve Analysis
        dca_df = decision_curve_analysis(y_test, y_test_prob, name)
        
        dca_df.to_csv(
            os.path.join(args.output_dir, f"{name}_DECISION_CURVE.csv"),
            index=False
        )

    else:

        # ---- Train performance ----
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)

        # ---- Test performance ----
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)

        print(f"{name} Train RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}")
        print(f"{name} Test  RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}")

        perf_dict = {
            "Model": name,
            "Train_RMSE": train_rmse,
            "Train_R2": train_r2,
            "Test_RMSE": test_rmse,
            "Test_R2": test_r2
        }

    # Save individual model performance
    pd.DataFrame([perf_dict]).to_csv(
        os.path.join(args.output_dir, f"{name}_performance.csv"),
        index=False
    )

    # Append to master results list
    all_results.append(perf_dict)
    
    # -------------------------------
    # Permutation Importance
    # -------------------------------
    perm = permutation_importance(best_model, X_test, y_test, scoring="roc_auc" if args.task=="classification" else "neg_root_mean_squared_error", n_repeats=30, random_state=args.random_state, n_jobs=-1)
    perm_df = pd.DataFrame({"feature": X.columns, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
    perm_df.to_csv(os.path.join(args.output_dir,f"{name}_permutation_importance.csv"), index=False)
    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": perm.importances_mean
    })
    
    importance_dict[name] = importance_df
        # -------------------------------
    # SHAP Values
    # -------------------------------
    try:
        background = shap.sample(X_train, 100, random_state=args.random_state)
        explainer = shap.Explainer(best_model.predict, background)
        shap_values = explainer(X_test)
        
        shap_df = pd.DataFrame(shap_values.values, columns=X.columns)
        shap_df["sample_index"] = X_test.index
        shap_df.to_csv(os.path.join(args.output_dir,f"{name}_shap_values.csv"), index=False)
        
        # Global importance
        shap_importance = np.abs(shap_values.values).mean(axis=0)
        pd.DataFrame({"feature": X.columns, "mean_abs_shap": shap_importance}).sort_values("mean_abs_shap", ascending=False).to_csv(os.path.join(args.output_dir,f"{name}_shap_importance.csv"), index=False)
        for i, feature in enumerate(X.columns):
            shap_tracking.append({
                "Model": name,
                "Feature": feature,
                "MeanAbsSHAP": shap_importance[i]
            })
    except:
        print(f"SHAP analysis not available for {name}")


# -------------------------------
# Save combined performance table
# -------------------------------
all_results_df = pd.DataFrame(all_results)

# -------------------------------
# Rank models automatically
# -------------------------------
if args.task == "classification":
    # Higher AUC is better
    all_results_df["Rank"] = all_results_df["Test_AUC"].rank(ascending=False, method="min")
    all_results_df = all_results_df.sort_values("Test_AUC", ascending=False)

    best_model_name = all_results_df.iloc[0]["Model"]
    print(f"\nBest model based on Test AUC: {best_model_name}")

else:
    # Lower RMSE is better
    all_results_df["Rank"] = all_results_df["Test_RMSE"].rank(ascending=True, method="min")
    all_results_df = all_results_df.sort_values("Test_RMSE", ascending=True)

    best_model_name = all_results_df.iloc[0]["Model"]
    print(f"\nBest model based on Test RMSE: {best_model_name}")

# Save ranked table
all_results_df.to_csv(
    os.path.join(args.output_dir, "ALL_MODEL_PERFORMANCE_RANKED.csv"),
    index=False
)

print("Saved ranked performance file: ALL_MODEL_PERFORMANCE_RANKED.csv")

# ---------------------------------
# Save Master ROC File
# ---------------------------------
if args.task == "classification":
    master_roc = pd.concat(all_roc_rows)
    master_roc.to_csv(os.path.join(args.output_dir, "ALL_MODELS_ROC_CURVES.csv"), index=False)

#-------------------------------------
# DeLong Test on AUCs
#-------------------------------------
from itertools import combinations
Delong_Stats=[]
if args.task == "classification":
    for (name1, preds1), (name2, preds2) in combinations(model_predictions.items(), 2):

        auc1, auc2, diff, p = delong_roc_test(
            y_test,
            preds1,
            preds2
        )
        Delong_Stats.append({
            "Model 1": name1,
            "Model 2": name2,
            "AUC 1": auc1,
            "AUC 2": auc2,
            "AUC Diff": diff,
            "p-value": p
            })
    Delong_Stats_Out=pd.DataFrame(Delong_Stats)
    Delong_Stats_Out.to_csv(os.path.join(args.output_dir, "ALL_MODELS_DeLong_Test.csv"), index=False)
    


# ---------------------------------
# Feature Stability
# ---------------------------------
stability_rows = []
for model_name in set([row["Model"] for row in feature_tracking]):
    model_features = [
        row["Features_Kept"]
        for row in feature_tracking
        if row["Model"] == model_name
    ]
    flat = [f for sublist in model_features for f in sublist]
    counts = Counter(flat)
    total = len(model_features)
    for feat, count in counts.items():
        stability_rows.append({
            "Model": model_name,
            "Feature": feat,
            "Selection_Frequency": count / total
        })

pd.DataFrame(stability_rows).to_csv(
    os.path.join(args.output_dir, "FEATURE_STABILITY.csv"),
    index=False
)

# ---------------------------------
# SHAP Stability
# ---------------------------------
if shap_tracking:
    shap_df = pd.DataFrame(shap_tracking)
    shap_df.to_csv(
        os.path.join(args.output_dir, "SHAP_FOLD_IMPORTANCE.csv"),
        index=False
    )
    
# Combine all DCA files
import glob

dca_files = glob.glob(os.path.join(args.output_dir, "*_DECISION_CURVE.csv"))
all_dca = pd.concat([pd.read_csv(f) for f in dca_files])
all_dca.to_csv(
    os.path.join(args.output_dir, "ALL_MODELS_DECISION_CURVE.csv"),
    index=False
)

# Assume you already computed permutation importance
# importance_df must have columns: Feature, Importance

best_model_pipeline = fitted_models_dict[best_model_name]
best_importance_df = importance_dict[best_model_name]

if args.task == "classification":
    for model_name in all_results_df["Model"]:
        results_df, best_subset = backward_feature_elimination(
            model_pipeline=fitted_models_dict[model_name],
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_importance=importance_dict[best_model_name],
            task=args.task
        )
        results_df.to_csv(
            os.path.join(args.output_dir, f"{model_name}_Break_Point_Analysis.csv"),
            index=False
        )
    
    
    

print("\nBest breakpoint model:")
print(best_subset)