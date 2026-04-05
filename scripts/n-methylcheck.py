import os
import ast
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


# -------------------------
# Utils
# -------------------------
def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def here() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def is_n_methylated(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    pattern = Chem.MolFromSmarts("[NX3]([CH3])")
    return 1 if mol.HasSubstructMatch(pattern) else 0


# -------------------------
# 0) Paths
# -------------------------
os.chdir(here())
MONOMER_CSV = os.path.join(here(), "data/monomer_list.csv")
PEPTIDE_CSV = os.path.join(here(), "data/processed_peptides.csv")

# -------------------------
# 1) Load monomers → {symbol: (smiles, logP, is_nme)}
# -------------------------
monomer_df = load_data(MONOMER_CSV)

monomers_list = {}
for symbol, smile in zip(monomer_df["Symbol"], monomer_df["replaced_SMILES"]):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        continue
    logP = float(Descriptors.MolLogP(mol))
    nme = is_n_methylated(smile)
    monomers_list[symbol] = (smile, logP, nme)

if not monomers_list:
    raise ValueError("monomers_list is empty.")

# -------------------------
# 2) Load peptides & extract per-position features
# -------------------------
peptides = load_data(PEPTIDE_CSV)
peptides["Sequence"] = peptides["Sequence"].apply(ast.literal_eval)

LOGP_DICTIONARY = {}
for seq, permeability, pid in zip(peptides["Sequence"], peptides["Permeability"], peptides["ID"]):
    logP_values, chiral_tags, nme_tags = [], [], []
    bad = False
    for monomer in seq:
        if monomer not in monomers_list:
            bad = True
            break
        smile, logP, nme = monomers_list[monomer]
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            bad = True
            break
        logP_values.append(logP)
        chiral_tags.append(1 if monomer.startswith('d') else 0)
        nme_tags.append(nme)

    if bad:
        continue
    LOGP_DICTIONARY[tuple(seq)] = [logP_values, chiral_tags, nme_tags, float(permeability), pid]

if not LOGP_DICTIONARY:
    raise ValueError("LOGP_DICTIONARY is empty.")

# -------------------------
# 3) Build feature table (18 features)
# -------------------------
rows = []
for seq, (logP_values, chiral_tags, nme_tags, permeability, pid) in LOGP_DICTIONARY.items():
    row = {"Permeability": permeability, "Sequence": "-".join(seq), "ID": pid}
    for i, v in enumerate(logP_values, 1):
        row[f"Pos_{i}_logP"] = v
    for i, d in enumerate(chiral_tags, 1):
        row[f"Pos_{i}_is_D"] = d
    for i, n in enumerate(nme_tags, 1):
        row[f"Pos_{i}_is_Nme"] = n
    rows.append(row)

df = pd.DataFrame(rows)

num_cols = (
    ["Permeability"]
    + [f"Pos_{i}_logP" for i in range(1, 7)]
    + [f"Pos_{i}_is_D" for i in range(1, 7)]
    + [f"Pos_{i}_is_Nme" for i in range(1, 7)]
)

df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=num_cols).reset_index(drop=True)
df = df[df["Permeability"] != -10.0].reset_index(drop=True)

feature_cols = [c for c in num_cols if c != "Permeability"]

print(f"[Info] Final dataset shape: {df.shape}")
print(f"[Info] Number of features: {len(feature_cols)}")
print(f"[Info] Features: {feature_cols}")
print(df.head())

# -------------------------
# 4) Split
# -------------------------
X = df[feature_cols].values
y = df["Permeability"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n[Info] Train: {len(X_train)}, Test: {len(X_test)}")

# -------------------------
# 5) Hyperparameter tuning
# -------------------------
rfr_pipeline = Pipeline([("model", RandomForestRegressor(random_state=42, n_jobs=-1))])

param_dist = {
    "model__n_estimators": np.linspace(200, 1000, num=9, dtype=int),
    "model__max_depth": [None] + list(np.arange(4, 26, 2)),
    "model__min_samples_split": [2, 4, 6, 8, 10, 20, 40],
    "model__min_samples_leaf": [1, 2, 3, 4, 5, 8, 10],
    "model__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
    "model__bootstrap": [True, False],
    "model__max_samples": [None, 0.6, 0.8, 1.0],
    "model__criterion": ["squared_error", "friedman_mse"],
}

cv = KFold(n_splits=min(5, len(df)), shuffle=True, random_state=42)

rand_search = RandomizedSearchCV(
    rfr_pipeline, param_dist, n_iter=300,
    scoring="neg_mean_squared_error", cv=cv,
    random_state=42, n_jobs=-1, verbose=1,
)

print("\n[Info] Starting hyperparameter tuning...")
rand_search.fit(X_train, y_train)
best_rfrr = rand_search.best_estimator_
best_rfr = best_rfrr.named_steps["model"]
best_cv_rmse = np.sqrt(-rand_search.best_score_)

print("\n=== Tuning Results ===")
print("Best params:", rand_search.best_params_)
print(f"Best CV RMSE: {best_cv_rmse:.3f}")

# -------------------------
# 6) Test evaluation
# -------------------------
y_pred = best_rfr.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n=== Test Set ===")
print(f"R²:   {r2:.3f}")
print(f"RMSE: {rmse:.3f}")

# -------------------------
# 7) Feature importances
# -------------------------
importances = pd.Series(best_rfr.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature importances:")
print(importances)

plt.figure(figsize=(10, 6))
importances.plot(kind='barh')
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importances (18 features)')
plt.tight_layout()
plt.show()

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
plt.show()

print("[Info] Done!")