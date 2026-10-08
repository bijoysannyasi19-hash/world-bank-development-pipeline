import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import joblib

def run_model():
    pro = os.path.join("data", "processed")
    fig = "figures"
    res = "results"
    mod = "models"
    os.makedirs(mod, exist_ok=True)
    df = pd.read_csv(os.path.join(pro, "clustered.csv"))
    
    tgt = "SP.DYN.LE00.IN"
    df = df.dropna(subset=[tgt]).copy()
    
    ex = [tgt, "SP.DYN.IMRT.IN", "SP.DYN.TFRT.IN", "id", "name", "region_name", "income_group", "cluster"]
    eng = ["health_spend_pc", "elec_urban_interaction"]
    
    a_f = [c for c in df.columns if c not in ex and df[c].dtype in [np.float64, np.int64]]
    b_f = [c for c in a_f if c not in eng]
    
    df_tr, df_te = train_test_split(df, test_size=0.2, random_state=42)
    df_te.to_csv(os.path.join(pro, "test_data.csv"), index=False)
    
    X_tr_b = df_tr[b_f]
    X_tr_a = df_tr[a_f]
    y_tr = df_tr[tgt]
    
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
    
    mdls = {
        "Dummy": DummyRegressor(strategy="mean"),
        "Ridge": Ridge(),
        "RF": RandomForestRegressor(random_state=42),
        "HGB": HistGradientBoostingRegressor(random_state=42)
    }
    
    cv_r = []
    
    for n, m in mdls.items():
        pl_b = Pipeline([("imp", SimpleImputer(strategy="mean")), ("sc", StandardScaler()), ("m", m)])
        s_b = cross_validate(pl_b, X_tr_b, y_tr, cv=cv, scoring="neg_root_mean_squared_error")
        cv_r.append({"Model": n, "Features": "Base", "RMSE": -s_b["test_score"].mean(), "STD": s_b["test_score"].std()})
        
        pl_a = Pipeline([("imp", SimpleImputer(strategy="mean")), ("sc", StandardScaler()), ("m", m)])
        s_a = cross_validate(pl_a, X_tr_a, y_tr, cv=cv, scoring="neg_root_mean_squared_error")
        cv_r.append({"Model": n, "Features": "Eng", "RMSE": -s_a["test_score"].mean(), "STD": s_a["test_score"].std()})
    
    df_cv = pd.DataFrame(cv_r)
    df_cv.to_csv(os.path.join(res, "cv_results.csv"), index=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_cv, x="Model", y="RMSE", hue="Features")
    plt.errorbar(x=np.arange(len(mdls)) - 0.2, y=df_cv[df_cv["Features"]=="Base"]["RMSE"], yerr=df_cv[df_cv["Features"]=="Base"]["STD"], fmt="none", c="black", capsize=5)
    plt.errorbar(x=np.arange(len(mdls)) + 0.2, y=df_cv[df_cv["Features"]=="Eng"]["RMSE"], yerr=df_cv[df_cv["Features"]=="Eng"]["STD"], fmt="none", c="black", capsize=5)
    plt.title("Model Comparison (RMSE)")
    plt.ylabel("RMSE (Years)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig, "11_model_comparison.png"), dpi=72)
    plt.close()
    
    pl_opt = Pipeline([("imp", SimpleImputer(strategy="mean")), ("sc", StandardScaler()), ("m", HistGradientBoostingRegressor(random_state=42))])
    p_g = {
        "m__learning_rate": [0.05, 0.1],
        "m__max_iter": [100, 200]
    }
    gs = GridSearchCV(pl_opt, p_g, cv=5, scoring="neg_root_mean_squared_error")
    gs.fit(X_tr_a, y_tr)
    
    t_r = pd.DataFrame(gs.cv_results_)
    plt.figure(figsize=(8, 6))
    bp = sns.barplot(x=[str(p) for p in t_r["params"]], y=-t_r["mean_test_score"])
    bp.set_xticklabels(bp.get_xticklabels(), rotation=45, ha="right")
    plt.title("Tuning Results (HGB)")
    plt.ylabel("RMSE (Years)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig, "12_tuning_results.png"), dpi=72)
    plt.close()
    
    f_m = gs.best_estimator_
    joblib.dump(f_m, os.path.join(mod, "final_model.joblib"))
    joblib.dump(a_f, os.path.join(mod, "features_list.joblib"))
    print("Modeling done")

if __name__ == "__main__":
    run_model()
