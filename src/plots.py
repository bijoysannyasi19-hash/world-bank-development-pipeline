import os
import matplotlib.pyplot as plt

def draw_arch():
    d = "docs"
    os.makedirs(d, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")
    bs = [
        (0.5, 0.9, "Public API"), (0.5, 0.8, "Raw Cache"),
        (0.5, 0.7, "Cleaning"), (0.5, 0.6, "Processed Data"),
        (0.5, 0.5, "EDA & Features"), (0.2, 0.4, "Clustering"),
        (0.8, 0.4, "Supervised"), (0.5, 0.3, "Error Analysis"),
        (0.5, 0.2, "Insights"), (0.5, 0.1, "Report")
    ]
    for x, y, t in bs:
        ax.text(x, y, t, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", fc="lightblue"))
    plt.savefig(os.path.join(d, "architecture.png"), dpi=200)
    plt.close()
    print("Architecture diagram generated")

if __name__ == "__main__":
    draw_arch()
