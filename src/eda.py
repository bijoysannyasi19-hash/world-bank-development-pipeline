import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda():
    pro = os.path.join("data", "processed")
    fig = "figures"
    os.makedirs(fig, exist_ok=True)
    df = pd.read_csv(os.path.join(pro, "cleaned.csv"))
    
    plt.figure(figsize=(10, 6))
    na_c = df.isna().sum()
    na_c = na_c[na_c > 0].sort_values()
    sns.barplot(x=na_c.values, y=na_c.index, palette="viridis")
    plt.title("Missing Values by Feature")
    plt.xlabel("Count of Missing Values")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(fig, "02_missingness.png"), dpi=72)
    plt.close()
    
    nc = df.select_dtypes(include=["float64", "int64"]).columns
    nl = len(nc)
    fc = 4
    fr = (nl + fc - 1) // fc
    fig_d, ax_d = plt.subplots(fr, fc, figsize=(15, fr * 3))
    ax_d = ax_d.flatten()
    for i, c in enumerate(nc):
        sns.histplot(df[c].dropna(), kde=True, ax=ax_d[i], color="blue")
        ax_d[i].set_title(f"Distribution of {c}")
        ax_d[i].set_xlabel(c)
        ax_d[i].set_ylabel("Frequency")
    for j in range(i + 1, len(ax_d)):
        ax_d[j].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(fig, "03_distributions.png"), dpi=72)
    plt.close()
    
    plt.figure(figsize=(12, 10))
    cm = df[nc].corr()
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(fig, "04_correlation_heatmap.png"), dpi=72)
    plt.close()
    
    tgt = "SP.DYN.LE00.IN"
    if tgt in df.columns:
        pc = [c for c in nc if c != tgt and c not in ["SP.DYN.IMRT.IN", "SP.DYN.TFRT.IN", "id"]]
        if pc:
            top_p = cm[tgt].drop([tgt, "SP.DYN.IMRT.IN", "SP.DYN.TFRT.IN"], errors="ignore").abs().sort_values(ascending=False).head(4).index
            fig_s, ax_s = plt.subplots(2, 2, figsize=(12, 10))
            ax_s = ax_s.flatten()
            for i, p in enumerate(top_p):
                sns.scatterplot(data=df, x=p, y=tgt, hue="region_name", ax=ax_s[i], palette="tab10")
                ax_s[i].set_title(f"Life Expectancy vs {p}")
                ax_s[i].set_xlabel(p)
                ax_s[i].set_ylabel("Life Expectancy")
                if ax_s[i].get_legend():
                    ax_s[i].get_legend().remove()
            h, l = ax_s[0].get_legend_handles_labels()
            fig_s.legend(h, l, loc="lower center", ncol=4)
            plt.tight_layout(rect=[0, 0.05, 1, 1])
            plt.savefig(os.path.join(fig, "05_target_relationships.png"), dpi=72)
            plt.close()
            
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="income_group", y=tgt, palette="Set2")
        plt.title("Life Expectancy by Income Group")
        plt.xlabel("Income Group")
        plt.ylabel("Life Expectancy")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(fig, "06_group_comparisons.png"), dpi=72)
        plt.close()

if __name__ == "__main__":
    run_eda()
