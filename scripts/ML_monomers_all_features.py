import os
import ast
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
import joblib

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# -------------------------
# Utils
# -------------------------
def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def here() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -------------------------
# Per-monomer 2D RDKit descriptor extractor
# Returns a dict of {descriptor_name: value}
# -------------------------
MONOMER_DESCRIPTORS = [
    ("MolWt",        Descriptors.MolWt),
    ("LogP",         Descriptors.MolLogP),
    ("TPSA",         Descriptors.TPSA),
    ("HBD",          rdMolDescriptors.CalcNumHBD),       # H-bond donors
    ("HBA",          rdMolDescriptors.CalcNumHBA),       # H-bond acceptors
    ("RotBonds",     rdMolDescriptors.CalcNumRotatableBonds),
    ("RingCount",    rdMolDescriptors.CalcNumRings),
    ("AromaticRings",rdMolDescriptors.CalcNumAromaticRings),
    ("MaxPartCharge",Descriptors.MaxPartialCharge),
    ("MinPartCharge",Descriptors.MinPartialCharge),
    ("Chi0",         Descriptors.Chi0),                  # molecular connectivity
    ("Chi1",         Descriptors.Chi1),
]

DESC_NAMES = [name for name, _ in MONOMER_DESCRIPTORS]

def compute_monomer_descriptors(mol) -> dict:
    """Compute all 2D descriptors for a single monomer molecule."""
    result = {}
    for name, fn in MONOMER_DESCRIPTORS:
        try:
            result[name] = float(fn(mol))
        except Exception:
            result[name] = np.nan
    return result

# -------------------------
# 0) Paths & chdir
# -------------------------
os.chdir(here())
MONOMER_CSV = os.path.join(here(), "data", "monomer_list.csv")
PEPTIDE_CSV = os.path.join(here(), "data", "processed_peptides.csv")

# -------------------------
# 1) Load monomers → compute descriptors once
# -------------------------
monomer_df = load_data(MONOMER_CSV)

monomers_list = {}  # symbol -> {"mol": mol, "descs": {name: val}, "is_D": int}
for symbol, smile in zip(monomer_df["Symbol"], monomer_df["replaced_SMILES"]):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        continue
    descs = compute_monomer_descriptors(mol)
    is_D  = 1 if symbol.startswith('d') else 0
    monomers_list[symbol] = {"mol": mol, "descs": descs, "is_D": is_D}

if not monomers_list:
    raise ValueError("monomers_list is empty. Check monomer_list.csv content.")

print(f"[Info] Loaded {len(monomers_list)} monomers, each with {len(DESC_NAMES)} descriptors.")

# -------------------------
# 2) Load peptides
# -------------------------
peptides = load_data(PEPTIDE_CSV)
peptides["Sequence"] = peptides["Sequence"].apply(ast.literal_eval)

# -------------------------
# 3) Build feature table
#    For each peptide: per-position descriptors + is_D flag
# -------------------------
rows = []
for _, row in peptides.iterrows():
    seq          = row["Sequence"]
    permeability = float(row["Permeability"])
    pid          = row["ID"]

    if permeability == -10.0:
        continue

    entry = {"Permeability": permeability, "Sequence": "-".join(seq), "ID": pid}
    bad   = False

    for pos, monomer in enumerate(seq, start=1):
        if monomer not in monomers_list:
            bad = True
            break
        info = monomers_list[monomer]
        for desc_name, val in info["descs"].items():
            entry[f"Pos{pos}_{desc_name}"] = val
        entry[f"Pos{pos}_isD"] = info["is_D"]

    if bad:
        continue
    rows.append(entry)

df = pd.DataFrame(rows)

# Derive feature column list (everything except metadata)
meta_cols    = ["Permeability", "Sequence", "ID"]
feature_cols = [c for c in df.columns if c not in meta_cols]

# Coerce & drop NaN rows
df[feature_cols + ["Permeability"]] = df[feature_cols + ["Permeability"]].apply(
    pd.to_numeric, errors="coerce"
)
df = df.dropna(subset=feature_cols + ["Permeability"]).reset_index(drop=True)

print(f"[Info] Dataset shape:     {df.shape}")
print(f"[Info] Number of features: {len(feature_cols)}")
print(f"[Info] Features (first 10): {feature_cols[:10]}")

# -------------------------
# 4) Train/test split
# -------------------------
X = df[feature_cols].values
y = df["Permeability"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n[Info] Train: {len(X_train)}, Test: {len(X_test)}")

# -------------------------
# 5) Hyperparameter tuning
# -------------------------
rfr_pipeline = Pipeline([
    ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
])

param_dist = {
    "model__n_estimators":      np.linspace(200, 1000, num=9, dtype=int),
    "model__max_depth":         [None] + list(np.arange(4, 26, 2)),
    "model__min_samples_split": [2, 4, 6, 8, 10, 20, 40],
    "model__min_samples_leaf":  [1, 2, 3, 4, 5, 8, 10],
    "model__max_features":      ["sqrt", "log2", 0.3, 0.5, 0.7],
    "model__bootstrap":         [True, False],
    "model__max_samples":       [None, 0.6, 0.8, 1.0],
    "model__criterion":         ["squared_error", "friedman_mse"],
}

cv = KFold(n_splits=min(5, len(df)), shuffle=True, random_state=42)

rand_search = RandomizedSearchCV(
    rfr_pipeline, param_dist,
    n_iter=300, scoring="neg_mean_squared_error",
    cv=cv, random_state=42, n_jobs=-1, verbose=1
)

print("\n[Info] Starting hyperparameter tuning (300 iterations)...")
rand_search.fit(X_train, y_train)

best_pipeline = rand_search.best_estimator_
best_rf       = best_pipeline.named_steps["model"]
best_cv_rmse  = np.sqrt(-rand_search.best_score_)

print("\n=== Hyperparameter Tuning Results ===")
print("Best params:", rand_search.best_params_)
print(f"Best CV RMSE: {best_cv_rmse:.3f}")

# -------------------------
# 6) Test set evaluation
# -------------------------
y_pred = best_rf.predict(X_test)
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n=== Test Set Performance ===")
print(f"R²   = {r2:.3f}")
print(f"RMSE = {rmse:.3f}")

# -------------------------
# 7) Feature importances — grouped by position
# -------------------------
importances = pd.Series(best_rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 20 feature importances:")
print(importances.head(20))

# ── 7a) Full feature importance bar chart (horizontal, sorted) ──
n_show = min(60, len(importances))  # show top-N if many features
imp_top = importances.head(n_show)

fig_height = max(8, n_show * 0.28)
fig, ax = plt.subplots(figsize=(10, fig_height))
colors = plt.cm.tab10([int(c.split('_')[0].replace('Pos','')) % 10
                       for c in imp_top.index])
bars = ax.barh(imp_top.index[::-1], imp_top.values[::-1], color=colors[::-1])
ax.set_xlabel("Feature Importance (MDI)", fontsize=12)
ax.set_title(f"Top {n_show} Feature Importances\n(colored by position)", fontsize=13)
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
plt.tight_layout()
plt.savefig("feature_importances_full.png", dpi=200)
plt.close()
print("[Info] Saved: feature_importances_full.png")

# ── 7b) Per-position grouped importance chart ──
# Sum importances across each position
n_positions = len(list(peptides["Sequence"])[0])  # number of residues
pos_importance = {}
for pos in range(1, n_positions + 1):
    pos_cols = [c for c in feature_cols if c.startswith(f"Pos{pos}_")]
    pos_importance[f"Pos{pos}"] = importances[pos_cols].sum()

pos_imp_series = pd.Series(pos_importance).sort_values(ascending=False)
print("\nPer-position total importance:")
print(pos_imp_series)

fig, ax = plt.subplots(figsize=(8, 5))
colors_pos = plt.cm.tab10(np.arange(len(pos_imp_series)) % 10)
ax.bar(pos_imp_series.index, pos_imp_series.values, color=colors_pos, edgecolor='k')
ax.set_ylabel("Summed Feature Importance", fontsize=12)
ax.set_xlabel("Sequence Position", fontsize=12)
ax.set_title("Total Feature Importance per Sequence Position", fontsize=13)
plt.tight_layout()
plt.savefig("feature_importances_by_position.png", dpi=200)
plt.close()
print("[Info] Saved: feature_importances_by_position.png")

# ── 7c) Descriptor-type importance (summed across all positions) ──
all_desc_types = DESC_NAMES + ["isD"]
desc_importance = {}
for desc in all_desc_types:
    desc_cols = [c for c in feature_cols if c.endswith(f"_{desc}")]
    desc_importance[desc] = importances[desc_cols].sum() if desc_cols else 0.0

desc_imp_series = pd.Series(desc_importance).sort_values(ascending=False)
print("\nPer-descriptor-type total importance:")
print(desc_imp_series)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(desc_imp_series.index, desc_imp_series.values, edgecolor='k',
       color=plt.cm.Paired(np.linspace(0, 1, len(desc_imp_series))))
ax.set_ylabel("Summed Feature Importance", fontsize=12)
ax.set_xlabel("Descriptor Type", fontsize=12)
ax.set_title("Total Feature Importance per Descriptor Type\n(summed across all positions)", fontsize=13)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig("feature_importances_by_descriptor.png", dpi=200)
plt.close()
print("[Info] Saved: feature_importances_by_descriptor.png")

# ── 7d) Heatmap: position × descriptor ──
heat_data = np.zeros((n_positions, len(all_desc_types)))
for pi, pos in enumerate(range(1, n_positions + 1)):
    for di, desc in enumerate(all_desc_types):
        col = f"Pos{pos}_{desc}"
        heat_data[pi, di] = importances.get(col, 0.0)

fig, ax = plt.subplots(figsize=(13, 5))
im = ax.imshow(heat_data, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(len(all_desc_types)))
ax.set_xticklabels(all_desc_types, rotation=40, ha='right', fontsize=9)
ax.set_yticks(range(n_positions))
ax.set_yticklabels([f"Pos{i+1}" for i in range(n_positions)], fontsize=10)
ax.set_title("Feature Importance Heatmap: Position × Descriptor", fontsize=13)
plt.colorbar(im, ax=ax, label="Importance")
plt.tight_layout()
plt.savefig("feature_importances_heatmap.png", dpi=200)
plt.close()
print("[Info] Saved: feature_importances_heatmap.png")

# -------------------------
# 8) Predicted vs True
# -------------------------
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k', s=70)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, 'r--', lw=2, label='Ideal: y = x')
plt.xlabel("True Permeability", fontsize=13)
plt.ylabel("Predicted Permeability", fontsize=13)
plt.title(f"Predicted vs. True Permeability\nR² = {r2:.3f}, RMSE = {rmse:.3f}", fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("predicted_vs_true.png", dpi=200)
plt.close()
print("[Info] Saved: predicted_vs_true.png")

# -------------------------
# 9) Save model & data
# -------------------------
output_dir = "saved_model"
os.makedirs(output_dir, exist_ok=True)

pd.DataFrame({'feature': feature_cols}).to_csv(
    os.path.join(output_dir, "feature_names.csv"), index=False)
df.to_csv(os.path.join(output_dir, "full_dataset_with_features.csv"), index=False)
joblib.dump(best_pipeline, os.path.join(output_dir, "random_forest_model.joblib"))

print(f"\n[Info] Model saved to {output_dir}/random_forest_model.joblib")
print("[Info] Done!")