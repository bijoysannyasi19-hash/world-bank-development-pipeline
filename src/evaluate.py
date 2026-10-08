import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import joblib

def run_evaluate():
    pro = os.path.join("data", "processed")
    fig = "figures"
    res = "results"
    mod = "models"
    df_te = pd.read_csv(os.path.join(pro, "test_data.csv"))
    df_tr = pd.read_csv(os.path.join(pro, "clustered.csv"))
    
    tgt = "SP.DYN.LE00.IN"
    df_tr = df_tr.dropna(subset=[tgt]).copy()
    tr_m = df_tr[tgt].mean()
    
    y_te = df_te[tgt].values
    y_dm = np.full_like(y_te, tr_m)
    
    f_m = joblib.load(os.path.join(mod, "final_model.joblib"))
    a_f = joblib.load(os.path.join(mod, "features_list.joblib"))
    X_te = df_te[a_f]
    
    y_pr = f_m.predict(X_te)
    
    rm = root_mean_squared_error(y_te, y_pr)
    ma = mean_absolute_error(y_te, y_pr)
    r2 = r2_score(y_te, y_pr)
    
    rm_d = root_mean_squared_error(y_te, y_dm)
    ma_d = mean_absolute_error(y_te, y_dm)
    r2_d = r2_score(y_te, y_dm)
    
    n_b = 1000
    b_r = []
    np.random.seed(42)
    for _ in range(n_b):
        idx = np.random.choice(len(y_te), len(y_te), replace=True)
        b_r.append(root_mean_squared_error(y_te[idx], y_pr[idx]))
    ci_l = np.percentile(b_r, 2.5)
    ci_u = np.percentile(b_r, 97.5)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_te, y_pr, alpha=0.7)
    plt.plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], "r--")
    plt.title("Predicted vs Actual Life Expectancy")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(fig, "13_predicted_vs_actual.png"), dpi=72)
    plt.close()
    
    rs = y_te - y_pr
    plt.figure(figsize=(8, 6))
    sns.histplot(rs, kde=True)
    plt.title("Residuals Distribution")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig, "14_residuals.png"), dpi=72)
    plt.close()
    
    pi = permutation_importance(f_m, X_te, y_te, n_repeats=10, random_state=42, scoring="neg_root_mean_squared_error")
    pi_m = pi.importances_mean
    pi_s = pi.importances_std
    idx_s = pi_m.argsort()
    
    plt.figure(figsize=(10, 6))
    plt.barh(np.array(a_f)[idx_s], pi_m[idx_s], xerr=pi_s[idx_s])
    plt.title("Permutation Feature Importance (Test Set)")
    plt.xlabel("Increase in RMSE when feature is shuffled")
    plt.tight_layout()
    plt.savefig(os.path.join(fig, "15_feature_importance.png"), dpi=72)
    plt.close()
    
    df_te["Prediction"] = y_pr
    df_te["Error"] = np.abs(rs)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_te, x="cluster", y="Error", palette="Set1")
    plt.title("Absolute Error by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Absolute Error (Years)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig, "16_error_by_cluster.png"), dpi=72)
    plt.close()
    
    w_p = df_te.sort_values(by="Error", ascending=False).head(5)[["name", "Error", tgt, "Prediction"]].to_dict(orient="records")
    
    m_d = {
        "RMSE": float(rm),
        "MAE": float(ma),
        "R2": float(r2),
        "RMSE_CI_95": [float(ci_l), float(ci_u)],
        "Dummy_RMSE": float(rm_d),
        "Dummy_MAE": float(ma_d),
        "Dummy_R2": float(r2_d),
        "Worst_Predicted": w_p
    }
    
    with open(os.path.join(res, "test_results.json"), "w") as f:
        json.dump(m_d, f, indent=4)
    print("Evaluation done")

if __name__ == "__main__":
    run_evaluate()
