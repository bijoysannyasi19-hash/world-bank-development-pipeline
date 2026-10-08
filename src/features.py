import os
import pandas as pd

def run_features():
    pro = os.path.join("data", "processed")
    df = pd.read_csv(os.path.join(pro, "cleaned.csv"))
    
    if "NY.GDP.PCAP.CD" in df.columns and "SH.XPD.CHEX.GD.ZS" in df.columns:
        df["health_spend_pc"] = df["NY.GDP.PCAP.CD"] * (df["SH.XPD.CHEX.GD.ZS"] / 100.0)
    
    if "EG.ELC.ACCS.ZS" in df.columns and "SP.URB.TOTL.IN.ZS" in df.columns:
        df["elec_urban_interaction"] = df["EG.ELC.ACCS.ZS"] * df["SP.URB.TOTL.IN.ZS"] / 100.0
    
    df.to_csv(os.path.join(pro, "features.csv"), index=False)
    print(f"Features engineered: {df.shape}")

if __name__ == "__main__":
    run_features()
