import os
import json
import pandas as pd
import numpy as np

def clean_data():
    raw = os.path.join("data", "raw")
    pro = os.path.join("data", "processed")
    res = "results"
    os.makedirs(pro, exist_ok=True)
    os.makedirs(res, exist_ok=True)
    dlog = []
    
    def log_dec(dec, opt, cho, rsn, evd):
        dlog.append({"decision": dec, "options": opt, "choice": cho, "reason": rsn, "evidence": evd})
    
    log_dec("Time window", "2019-2021, 2017-2019", "2017-2019", "Better coverage", "API checks")
    
    def get_val(x):
        return x["value"]
    
    mf = os.path.join(raw, "meta.json")
    with open(mf, "r", encoding="utf-8") as f:
        md = json.load(f)
    mdf = pd.DataFrame(md[1])
    mdf = mdf[mdf["region"].apply(get_val) != "Aggregates"].copy()
    mdf["region_name"] = mdf["region"].apply(get_val)
    mdf["income_group"] = mdf["incomeLevel"].apply(get_val)
    mdf = mdf[["id", "name", "region_name", "income_group"]]
    
    log_dec("Remove aggregates", "Keep or remove", "Remove", "Model needs real countries", f"{len(mdf)} remaining")
    
    inds = [
        "SP.DYN.LE00.IN", "NY.GDP.PCAP.CD", "SP.DYN.IMRT.IN", "SH.XPD.CHEX.GD.ZS",
        "EG.ELC.ACCS.ZS", "IT.NET.USER.ZS", "SP.URB.TOTL.IN.ZS", "SL.UEM.TOTL.ZS",
        "SP.POP.TOTL", "SH.H2O.BASW.ZS", "SP.DYN.TFRT.IN", "NY.GDP.MKTP.KD.ZG"
    ]
    
    dfm = mdf.copy()
    for ind in inds:
        fi = os.path.join(raw, f"{ind}.json")
        if not os.path.exists(fi):
            continue
        with open(fi, "r", encoding="utf-8") as f:
            jd = json.load(f)
        idf = pd.DataFrame(jd[1])
        idf["date"] = idf["date"].astype(str)
        idf = idf[idf["date"].isin(["2017", "2018", "2019"])]
        idf["value"] = pd.to_numeric(idf["value"], errors="coerce")
        idf["country_id"] = idf["countryiso3code"]
        idf = idf.groupby("country_id")["value"].mean().reset_index()
        idf.rename(columns={"value": ind}, inplace=True)
        r_b = len(dfm)
        dfm = dfm.merge(idf, left_on="id", right_on="country_id", how="left")
        dfm.drop(columns=["country_id"], inplace=True)
        r_a = len(dfm)
        print(f"Merged {ind}: {r_b} -> {r_a}")
    
    na_p = dfm.isna().mean()
    dk = na_p[na_p > 0.5].index.tolist()
    if dk:
        dfm.drop(columns=dk, inplace=True)
    log_dec("Drop bad indicators", "Keep all, drop >50%", "Drop >50%", "Too much missing", str(dk))
    
    na_c = dfm.isna().mean(axis=1)
    dfm = dfm[na_c <= 0.5].copy()
    log_dec("Drop bad countries", "Keep all, drop >50%", "Drop >50%", "Unreliable rows", f"{len(dfm)} left")
    
    if "NY.GDP.PCAP.CD" in dfm.columns:
        dfm["log_GDP_PC"] = np.log1p(dfm["NY.GDP.PCAP.CD"])
    if "SP.POP.TOTL" in dfm.columns:
        dfm["log_POP"] = np.log1p(dfm["SP.POP.TOTL"])
    
    log_dec("Log transforms", "None, log", "Log GDP and POP", "Skewed data", "EDA shows skew")
    
    dfm.to_csv(os.path.join(pro, "cleaned.csv"), index=False)
    pd.DataFrame(dlog).to_csv(os.path.join(res, "decision_log.csv"), index=False)
    print(f"Cleaned data: {dfm.shape}")

if __name__ == "__main__":
    clean_data()
