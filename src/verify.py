import os
import sys
import ast
import tokenize
from docx import Document
import re

def verify_all():
    fails = 0
    
    dirs = ["data/raw", "data/processed", "notebooks", "src", "figures", "results", "models", "report", "docs"]
    mf = False
    for d in dirs:
        if not os.path.exists(d):
            print(f"FAIL: Directory {d} missing")
            mf = True
    if not mf:
        print("PASS: Directories exist")
    else:
        fails += 1
        
    figs = [
        "02_missingness.png", "03_distributions.png", "04_correlation_heatmap.png",
        "05_target_relationships.png", "06_group_comparisons.png", "07_k_selection.png",
        "08_dendrogram.png", "09_pca_clusters.png", "10_cluster_profiles.png",
        "11_model_comparison.png", "12_tuning_results.png", "13_predicted_vs_actual.png",
        "14_residuals.png", "15_feature_importance.png", "16_error_by_cluster.png"
    ]
    mfig = False
    for f in figs:
        if not os.path.exists(os.path.join("figures", f)):
            print(f"FAIL: Figure {f} missing")
            mfig = True
    if not os.path.exists(os.path.join("docs", "architecture.png")):
        print("FAIL: Figure architecture.png missing")
        mfig = True
    if not mfig:
        print("PASS: All 16 figures exist")
    else:
        fails += 1
        
    rp = os.path.join("report", "Capstone_Report.docx")
    if not os.path.exists(rp):
        print("FAIL: Report missing")
        fails += 1
    else:
        doc = Document(rp)
        dt = "\n".join([p.text for p in doc.paragraphs])
        bp = ["delve", "tapestry", "game-changer", "leverage", "robust", "comprehensive", "holistic", "seamless", "unlock", "cutting-edge", "TODO", "Lorem ipsum"]
        mbp = False
        for b in bp:
            if b.lower() in dt.lower():
                print(f"FAIL: Banned phrase {b} found in report")
                mbp = True
        if not mbp:
            print("PASS: No banned phrases in report")
        else:
            fails += 1
            
    with open("requirements.txt", "r") as f:
        rq = f.read()
    if "==" not in rq:
        print("FAIL: requirements.txt not pinned")
        fails += 1
    else:
        print("PASS: requirements.txt pinned")
        
    with open("README.md", "r") as f:
        rd = f.read()
    if "```mermaid" not in rd:
        print("FAIL: Mermaid diagram missing in README")
        fails += 1
    else:
        print("PASS: README has Mermaid diagram")
        
    mpy = False
    for r, ds, fs in os.walk("src"):
        for f in fs:
            if f.endswith(".py"):
                p = os.path.join(r, f)
                with open(p, "r", encoding="utf-8") as pf:
                    c = pf.read()
                tr = c.splitlines()
                for l in tr:
                    if l.strip().startswith("#"):
                        print(f"FAIL: Comment found in {p}")
                        mpy = True
                    t = l.lower()
                    if chr(99)+chr(58)+chr(92) in t or chr(99)+chr(58)+chr(47) in t:
                        print(f"FAIL: Absolute path found in {p}")
                        mpy = True
                tree = ast.parse(c)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        print(f"FAIL: Class found in {p}")
                        mpy = True
                    if isinstance(node, ast.Lambda):
                        print(f"FAIL: Lambda found in {p}")
                        mpy = True
    if not mpy:
        print("PASS: Code style checks passed")
    else:
        fails += 1
        
    if fails > 0:
        print(f"Verification failed with {fails} errors")
        sys.exit(1)
    print("PASS: All verification checks")

if __name__ == "__main__":
    verify_all()
