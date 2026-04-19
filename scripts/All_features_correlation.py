import os
import numpy as np
import pandas as pd

def here():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

folder = here()
OUTPUT_DIR = os.path.join(folder, "All_features")

# ── Load full dataset ──
df = pd.read_csv(os.path.join(OUTPUT_DIR, "full_dataset_with_features.csv"))
meta_cols = ["Permeability", "Sequence", "ID"]
feature_cols = [c for c in df.columns if c not in meta_cols]

# ── Correlation matrix ──
corr = df[feature_cols].corr()

# ── Extract pairs with |r| > 0.95 (upper triangle only) ──
pairs = []
for i in range(len(feature_cols)):
    for j in range(i + 1, len(feature_cols)):
        r = corr.iloc[i, j]
        if abs(r) > 0.95:
            pairs.append({
                "feature_1": feature_cols[i],
                "feature_2": feature_cols[j],
                "pearson_r": round(r, 4),
            })

pairs_df = pd.DataFrame(pairs).sort_values("pearson_r", key=abs, ascending=False).reset_index(drop=True)

print(f"Found {len(pairs_df)} pairs with |r| > 0.95:\n")
print(pairs_df.to_string(index=False))

pairs_df.to_csv(os.path.join(OUTPUT_DIR, "highly_correlated_pairs.csv"), index=False)