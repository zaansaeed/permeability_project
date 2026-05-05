import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def here():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

folder = here()
out_dir = os.path.join(folder, "dataset_distributions")
os.makedirs(out_dir, exist_ok=True)

# ── Load data ──
X = pd.read_csv(os.path.join(folder, "saved_model", "X.csv"))
y = pd.read_csv(os.path.join(folder, "saved_model", "y.csv")).squeeze()

N_POS = 6
LOGP = [f"Pos_{i}_logP" for i in range(1, N_POS + 1)]
IS_D = [f"Pos_{i}_is_D" for i in range(1, N_POS + 1)]
IS_NSUB = [f"Pos_{i}_is_NSub" for i in range(1, N_POS + 1)]

# ── 1) LogP distributions ──
fig, ax = plt.subplots(figsize=(10, 6))
data = [X[col].dropna().values for col in LOGP]
labels = [f"Pos {i}" for i in range(1, N_POS + 1)]
bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
                meanprops=dict(marker="D", markerfacecolor="red", markersize=5))
for patch, color in zip(bp["boxes"], plt.cm.tab10(np.arange(N_POS))):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel("LogP")
ax.set_title("LogP Distribution by Position")
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "logP_distributions.png"), dpi=200)
plt.close(fig)

# ── 2) Binary feature bar charts ──
fig, axes = plt.subplots(2, 6, figsize=(14, 5))
for ax, col in zip(axes[0], IS_D):
    counts = X[col].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color=["#4C72B0", "#DD8452"])
    ax.set_title(col, fontsize=8)
    ax.set_xticks([0, 1])
for ax, col in zip(axes[1], IS_NSUB):
    counts = X[col].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color=["#4C72B0", "#DD8452"])
    ax.set_title(col, fontsize=8)
    ax.set_xticks([0, 1])
fig.suptitle("Binary Feature Distributions (is_D / is_NSub)", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "binary_distributions.png"), dpi=200)
plt.close(fig)

# ── 3) Permeability distribution ──
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(y, bins=40, edgecolor="black", alpha=0.7)
ax.axvline(y.mean(), color="red", ls="--", label=f"μ={y.mean():.2f}")
ax.axvline(y.median(), color="green", ls="--", label=f"median={y.median():.2f}")
ax.set_xlabel("log Permeability")
ax.set_ylabel("Count")
ax.set_title("Permeability Distribution")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "permeability_distribution.png"), dpi=200)
plt.close(fig)

print(f"Saved all plots to {out_dir}")

import os
import pandas as pd
import matplotlib.pyplot as plt

monomers = pd.read_csv("/Users/zaan/Desktop/permeability_project/data/monomer_list_updated.csv")

counts = monomers["is_D"].value_counts().sort_index()
print(f"is_D = 0: {counts.get(0, 0)}")
print(f"is_D = 1: {counts.get(1, 0)}")

fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["is_D = 0", "is_D = 1"], [counts.get(0, 0), counts.get(1, 0)],
       color=["#4C72B0", "#DD8452"])
ax.set_ylabel("Count")
ax.set_title("is_D Distribution in Monomer List")
for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
    ax.text(i, v, str(v), ha="center", va="bottom")
fig.tight_layout()
plt.show()