"""
Cyclic Permutation Invariance Test
===================================
Tests whether the RF model's performance and feature importance rankings
are artifacts of the arbitrary linear encoding of the cyclic hexapeptide,
or reflect genuine positional chemistry.

For each of 6 cyclic rotations (shift = 0, 1, ..., 5):
  - Rotate the 18-feature vector (6 LogP, 6 chirality, 6 N-sub)
  - Retrain the RF with the same hyperparameters and split
  - Record R², RMSE, and per-position MDI importance

If importance "follows the chemistry" (i.e., rotates with the features),
the spatial interpretation is valid. If importance is stuck to column
indices regardless of which monomer sits there, it's an encoding artifact.

Usage:
  - Set DATA_DIR to wherever your CSVs live.
  - Run: python cyclic_permutation_test.py
"""

import os
import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MONOMER_CSV = os.path.join(DATA_DIR, "monomer_list_updated.csv")
PEPTIDE_CSV = os.path.join(DATA_DIR, "processed_peptides.csv")

# Output directory for results
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "positional_asymmetry")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_POSITIONS = 6
RANDOM_STATE = 42

# ─────────────────────────────────────────────
# 1) Load and build the unpermuted feature matrix
# ─────────────────────────────────────────────
monomer_df = pd.read_csv(MONOMER_CSV)
monomers_list = {}
for _, row in monomer_df.iterrows():
    symbol = row["Symbol"]
    logP = row.get("logP")
    is_D = row.get("is_D")
    is_NSub = row.get("is_NSub")
    if pd.isna(logP) or pd.isna(is_D) or pd.isna(is_NSub):
        continue
    monomers_list[symbol] = (float(logP), int(is_D), int(is_NSub))

peptides = pd.read_csv(PEPTIDE_CSV)
peptides["Sequence"] = peptides["Sequence"].apply(ast.literal_eval)

rows = []
for seq, permeability, pid in zip(peptides["Sequence"], peptides["Permeability"], peptides["ID"]):
    logP_vals, chiral_vals, nsub_vals = [], [], []
    bad = False
    for monomer in seq:
        if monomer not in monomers_list:
            bad = True
            break
        lp, d, ns = monomers_list[monomer]
        logP_vals.append(lp)
        chiral_vals.append(d)
        nsub_vals.append(ns)
    if bad:
        continue
    row = {"Permeability": float(permeability), "Sequence": "-".join(seq), "ID": pid}
    for i in range(N_POSITIONS):
        row[f"Pos_{i+1}_logP"] = logP_vals[i]
        row[f"Pos_{i+1}_is_D"] = chiral_vals[i]
        row[f"Pos_{i+1}_is_NSub"] = nsub_vals[i]
    rows.append(row)

df = pd.DataFrame(rows)

pos_cols = [c for c in df.columns if c.startswith("Pos_")]
df = df.groupby("Sequence", as_index=False).agg({
    "Permeability": "mean", "ID": "first", **{c: "first" for c in pos_cols}
}).reset_index(drop=True)

num_cols = ["Permeability"] + pos_cols
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=num_cols).reset_index(drop=True)
df = df[df["Permeability"] != -10.0].reset_index(drop=True)

feature_cols = list(pos_cols)
print(f"[Info] Dataset: {len(df)} peptides, {len(feature_cols)} features")


# ─────────────────────────────────────────────
# 2) Cyclic permutation function
# ─────────────────────────────────────────────
def cyclically_permute_features(dataframe, shift):
    """
    Rotate the feature data by `shift` positions.
    shift=0 is original. shift=1 means original position 2 -> column 1.
    """
    new_df = dataframe.copy()
    for feat_type in ["logP", "is_D", "is_NSub"]:
        orig_cols = [f"Pos_{i+1}_{feat_type}" for i in range(N_POSITIONS)]
        orig_values = dataframe[orig_cols].values
        shifted_values = np.roll(orig_values, -shift, axis=1)
        for i in range(N_POSITIONS):
            new_df[orig_cols[i]] = shifted_values[:, i]
    return new_df


# ─────────────────────────────────────────────
# 3) Hyperparameter grid (same as original)
# ─────────────────────────────────────────────
param_dist = {
    "n_estimators": np.linspace(200, 1000, num=9, dtype=int),
    "max_depth": [None] + list(np.arange(4, 26, 2)),
    "min_samples_split": [2, 4, 6, 8, 10, 20, 40],
    "min_samples_leaf": [1, 2, 3, 4, 5, 8, 10],
    "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
    "bootstrap": [True],
    "max_samples": [None, 0.6, 0.8, 1.0],
    "criterion": ["squared_error", "friedman_mse"],
}


# ─────────────────────────────────────────────
# 4) Run experiment for each cyclic shift
# ─────────────────────────────────────────────
all_results = []
all_importances = []

for shift in range(N_POSITIONS):
    print(f"\n{'='*50}")
    print(f"  SHIFT = {shift}  (original position {shift+1} -> column 1)")
    print(f"{'='*50}")

    df_shifted = cyclically_permute_features(df, shift)

    X = df_shifted[feature_cols].values
    y = df_shifted["Permeability"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        param_dist,
        n_iter=300,
        scoring="neg_mean_squared_error",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    best_rf = search.best_estimator_

    y_pred = best_rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    cv_r2 = cross_val_score(best_rf, X_train, y_train, cv=cv, scoring="r2")

    print(f"  Test R2 = {r2:.4f}, RMSE = {rmse:.4f}")
    print(f"  CV R2   = {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")

    # Feature importances in shifted labeling
    imp_shifted = pd.Series(best_rf.feature_importances_, index=feature_cols)

    # Map back to original positions
    imp_original = {}
    for feat_type in ["logP", "is_D", "is_NSub"]:
        for i in range(N_POSITIONS):
            shifted_col = f"Pos_{i+1}_{feat_type}"
            original_pos = (i + shift) % N_POSITIONS + 1
            original_col = f"Pos_{original_pos}_{feat_type}"
            imp_original[original_col] = imp_shifted[shifted_col]

    all_results.append({
        "shift": shift,
        "test_R2": r2,
        "test_RMSE": rmse,
        "cv_R2_mean": cv_r2.mean(),
        "cv_R2_std": cv_r2.std(),
    })

    imp_row = {"shift": shift}
    imp_row.update(imp_original)
    all_importances.append(imp_row)

    # ── Save per-shift details ──
    # ── Save per-shift details ──
    shift_detail = pd.DataFrame([{
        "shift": shift,
        "test_R2": r2,
        "test_RMSE": rmse,
        "cv_R2_mean": cv_r2.mean(),
        "cv_R2_std": cv_r2.std(),
        **{col: imp_shifted[col] for col in feature_cols},  # raw column-index importances
    }])
    ordered_cols = ["shift", "test_R2", "test_RMSE", "cv_R2_mean", "cv_R2_std"] + [
        f"Pos_{i+1}_{ft}"
        for ft in ["logP", "is_D", "is_NSub"]
        for i in range(N_POSITIONS)
    ]
    shift_detail[ordered_cols].to_csv(
        os.path.join(OUTPUT_DIR, f"shift_{shift}_results.csv"), index=False
    )


# ─────────────────────────────────────────────
# 5) Save aggregate results
# ─────────────────────────────────────────────
results_df = pd.DataFrame(all_results)
imp_df = pd.DataFrame(all_importances)

# Performance summary
results_df.to_csv(os.path.join(OUTPUT_DIR, "performance_summary.csv"), index=False)

# Full importance table (all 18 features, mapped to original positions)
all_feat_cols = (
    [f"Pos_{i+1}_logP" for i in range(N_POSITIONS)]
    + [f"Pos_{i+1}_is_D" for i in range(N_POSITIONS)]
    + [f"Pos_{i+1}_is_NSub" for i in range(N_POSITIONS)]
)
imp_df[["shift"] + all_feat_cols].to_csv(
    os.path.join(OUTPUT_DIR, "importance_all_features.csv"), index=False
)

# LogP-only importance table
logp_cols = [f"Pos_{i+1}_logP" for i in range(N_POSITIONS)]
imp_df[["shift"] + logp_cols].to_csv(
    os.path.join(OUTPUT_DIR, "importance_logP.csv"), index=False
)

# Coefficient of variation per original position
cv_rows = []
for col in all_feat_cols:
    vals = imp_df[col].values
    cv_val = vals.std() / vals.mean() if vals.mean() > 0 else float("inf")
    cv_rows.append({
        "feature": col,
        "mean_importance": vals.mean(),
        "std_importance": vals.std(),
        "CV": cv_val,
    })
cv_df = pd.DataFrame(cv_rows)
cv_df.to_csv(os.path.join(OUTPUT_DIR, "importance_cv_by_feature.csv"), index=False)


# ─────────────────────────────────────────────
# 6) Print summary (same as before)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  PERFORMANCE ACROSS CYCLIC PERMUTATIONS")
print("="*60)
print(results_df.to_string(index=False))
print(f"\n  R2 mean:   {results_df['test_R2'].mean():.4f} +/- {results_df['test_R2'].std():.4f}")
print(f"  RMSE mean: {results_df['test_RMSE'].mean():.4f} +/- {results_df['test_RMSE'].std():.4f}")

print("\n" + "="*60)
print("  LogP IMPORTANCE (mapped back to original positions)")
print("="*60)
print(imp_df[["shift"] + logp_cols].to_string(index=False, float_format="%.4f"))

print("\n  Coefficient of Variation (std/mean) per original position:")
for col in logp_cols:
    vals = imp_df[col].values
    cv_val = vals.std() / vals.mean() if vals.mean() > 0 else float("inf")
    print(f"    {col}: CV = {cv_val:.3f}  (mean = {vals.mean():.4f}, std = {vals.std():.4f})")

print("\n" + "="*60)
print("  FULL IMPORTANCE TABLE (all 18 features, original positions)")
print("="*60)
print(imp_df[["shift"] + all_feat_cols].to_string(index=False, float_format="%.4f"))

print(f"\n[Info] All results saved to: {OUTPUT_DIR}")
print("[Info] Done!")