import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage

def run_cluster():
    pro = os.path.join("data", "processed")
    fig = "figures"
    res = "results"
    df = pd.read_csv(os.path.join(pro, "features.csv"))
    
    tgt = "SP.DYN.LE00.IN"
    ex = [tgt, "SP.DYN.IMRT.IN", "SP.DYN.TFRT.IN", "id", "name", "region_name", "income_group"]
    f_c = [c for c in df.columns if c not in ex and df[c].dtype in [np.float64, np.int64]]
    
    X = df[f_c].copy()
    imp = SimpleImputer(strategy="mean")
    X_i = imp.fit_transform(X)
    sc = StandardScaler()
    X_s = sc.fit_transform(X_i)
    
    K = range(2, 11)
    sse = []
    sil = []
    ch = []
    db = []
    
    for k in K:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_s)
        sse.append(km.inertia_)
        sil.append(silhouette_score(X_s, km.labels_))
        ch.append(calinski_harabasz_score(X_s, km.labels_))
        db.append(davies_bouldin_score(X_s, km.labels_))
    
    fig_k, ax_k = plt.subplots(2, 2, figsize=(12, 10))
    ax_k[0, 0].plot(K, sse, marker="o")
    ax_k[0, 0].set_title("Elbow Method (Inertia)")
    ax_k[0, 1].plot(K, sil, marker="o", color="orange")
    ax_k[0, 1].set_title("Silhouette Score")
    ax_k[1, 0].plot(K, ch, marker="o", color="green")
    ax_k[1, 0].set_title("Calinski-Harabasz Index")
    ax_k[1, 1].plot(K, db, marker="o", color="red")
    ax_k[1, 1].set_title("Davies-Bouldin Index")
    for a in ax_k.flatten():
        a.set_xlabel("Number of clusters (k)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig, "07_k_selection.png"), dpi=200)
    plt.close()
    
    k_opt = 3
    km_f = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
    l_km = km_f.fit_predict(X_s)
    
    L = linkage(X_s, method="ward")
    plt.figure(figsize=(10, 6))
    dendrogram(L, truncate_mode="level", p=5)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Cluster Size")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(fig, "08_dendrogram.png"), dpi=200)
    plt.close()
    
    hc = AgglomerativeClustering(n_clusters=k_opt)
    l_hc = hc.fit_predict(X_s)
    ari = adjusted_rand_score(l_km, l_hc)
    
    pca = PCA(n_components=2)
    X_p = pca.fit_transform(X_s)
    ev = pca.explained_variance_ratio_
    
    df["cluster"] = l_km
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_p[:, 0], y=X_p[:, 1], hue=df["cluster"], palette="Set1", s=60)
    plt.title(f"PCA View (PC1: {ev[0]:.2f}, PC2: {ev[1]:.2f})")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(fig, "09_pca_clusters.png"), dpi=200)
    plt.close()
    
    cp = df.groupby("cluster")[f_c].mean()
    cp.to_csv(os.path.join(res, "cluster_profiles.csv"))
    
    sc_cp = StandardScaler()
    cp_s = pd.DataFrame(sc_cp.fit_transform(cp), index=cp.index, columns=cp.columns)
    plt.figure(figsize=(12, 6))
    sns.heatmap(cp_s, cmap="RdBu", center=0, annot=False)
    plt.title("Cluster Profile Heatmap (Scaled)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig, "10_cluster_profiles.png"), dpi=200)
    plt.close()
    
    df.to_csv(os.path.join(pro, "clustered.csv"), index=False)
    with open(os.path.join(res, "cluster_metrics.json"), "w") as f:
        import json
        json.dump({"k_opt": k_opt, "ari": ari}, f, indent=4)
    print("Clustering done")

if __name__ == "__main__":
    run_cluster()
