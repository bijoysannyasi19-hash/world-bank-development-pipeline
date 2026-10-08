import os
import sys
import subprocess

def run_all():
    s_d = "src"
    s_f = [
        "get_data.py",
        "clean.py",
        "features.py",
        "eda.py",
        "cluster.py",
        "model.py",
        "evaluate.py",
        "plots.py",
        "build_report.py"
    ]
    for f in s_f:
        p = os.path.join(s_d, f)
        print(f"Running {p}")
        r = subprocess.run([sys.executable, p], capture_output=False)
        if r.returncode != 0:
            print(f"Error in {p}")
            sys.exit(1)
    
    print("Running verify.py")
    v_p = os.path.join(s_d, "verify.py")
    r = subprocess.run([sys.executable, v_p], capture_output=False)
    if r.returncode != 0:
        print("Verification failed")
        sys.exit(1)
    print("All tasks completed successfully")

if __name__ == "__main__":
    run_all()
