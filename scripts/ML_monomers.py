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
import joblib

import matplotlib.pyplot as plt

# -------------------------
# Utils
# -------------------------
def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def here() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -------------------------
# 0) Paths & chdir
# -------------------------
os.chdir(here())
MONOMER_CSV = os.path.join(here(), "data", "monomer_list.csv")
PEPTIDE_CSV = os.path.join(here(), "data", "processed_peptides.csv")

# -------------------------
# 1) Load monomer list and build {symbol: (smiles, logP)}
# -------------------------
monomer_df = load_data(MONOMER_CSV)

monomers_list = {}
for symbol, smile in zip(monomer_df["Symbol"], monomer_df["replaced_SMILES"]):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        continue
    logP = float(Descriptors.MolLogP(mol))
    monomers_list[symbol] = (smile, logP)

if not monomers_list:
    raise ValueError("monomers_list is empty.")

# -------------------------
# 2-3) Build feature table (average duplicate sequences)
# -------------------------
peptides = load_data(PEPTIDE_CSV)
peptides["Sequence"] = peptides["Sequence"].apply(ast.literal_eval)

rows = []
for seq, permeability, pid in zip(peptides["Sequence"], peptides["Permeability"], peptides["ID"]):
    logP_values = []
    chiral_tags = []
    bad = False
    for monomer in seq:
        if monomer not in monomers_list:
            bad = True
            break
        smile = monomers_list[monomer][0]
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            bad = True
            break
        logP_values.append(float(Descriptors.MolLogP(mol)))
        is_D = 1 if monomer.startswith('d') else 0
        chiral_tags.append(is_D)
    if bad:
        continue

    row = {"Permeability": float(permeability), "Sequence": "-".join(seq), "ID": pid}
    for i, v in enumerate(logP_values, start=1):
        row[f"Pos_{i}_logP"] = v
    for i, d in enumerate(chiral_tags, start=1):
        row[f"Pos_{i}_is_D"] = d
    rows.append(row)

df = pd.DataFrame(rows)

# Average permeability for duplicate sequences
pos_cols = [c for c in df.columns if c.startswith("Pos_")]
df = df.groupby("Sequence", as_index=False).agg({
    "Permeability": "mean",
    "ID": "first",
    **{c: "first" for c in pos_cols}
}).reset_index(drop=True)

num_cols = ["Permeability"] + \
           [f"Pos_{i}_logP" for i in range(1, 7)] + \
           [f"Pos_{i}_is_D" for i in range(1, 7)]

df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=num_cols).reset_index(drop=True)
df = df[df["Permeability"] != -10.0].reset_index(drop=True)
feature_cols = [c for c in num_cols if c != "Permeability"]

print(f"[Info] Final dataset shape: {df.shape}")
print(f"[Info] Features: {feature_cols}")

# -------------------------
# 4) Split data (track indices)
# -------------------------
X = df[feature_cols].values
y = df["Permeability"].values

X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
    X, y, df.index, test_size=0.2, random_state=42
)

print(f"[Info] Train: {len(X_train)}, Test: {len(X_test)}")

# -------------------------
# 5) Hyperparameter tuning
# -------------------------
rfr_pipeline = Pipeline([
    ("model", RandomForestRegressor(random_state=42, n_jobs=1))
])

param_dist = {
    "model__n_estimators": np.linspace(200, 1000, num=9, dtype=int),
    "model__max_depth": [None] + list(np.arange(4, 26, 2)),
    "model__min_samples_split": [2, 4, 6, 8, 10, 20, 40],
    "model__min_samples_leaf": [1, 2, 3, 4, 5, 8, 10],
    "model__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
    "model__bootstrap": [True],
    "model__max_samples": [None, 0.6, 0.8, 1.0],
    "model__criterion": ["squared_error", "friedman_mse"]
}

cv = KFold(n_splits=min(5, len(df)), shuffle=True, random_state=42)

rand_search = RandomizedSearchCV(
    rfr_pipeline, param_dist,
    n_iter=300, scoring="neg_mean_squared_error",
    cv=cv, random_state=42, n_jobs=1, verbose=1
)

print("\n[Info] Starting hyperparameter tuning...")
rand_search.fit(X_train, y_train)
best_pipeline = rand_search.best_estimator_
best_rfr = best_pipeline.named_steps["model"]
best_cv_rmse = np.sqrt(-rand_search.best_score_)

# CV stats for best hyperparameter config
cv_results = rand_search.cv_results_
best_idx = rand_search.best_index_

mean_cv_rmse = np.sqrt(-cv_results["mean_test_score"][best_idx])
std_cv_rmse = np.sqrt(cv_results["std_test_score"][best_idx])

# R² requires a separate cross_val_score call
from sklearn.model_selection import cross_val_score
cv_r2_scores = cross_val_score(best_pipeline, X_train, y_train, cv=cv, scoring="r2")
mean_cv_r2 = cv_r2_scores.mean()
std_cv_r2 = cv_r2_scores.std()

print("\n=== Best Params ===")
print(rand_search.best_params_)
print(f"CV RMSE: {best_cv_rmse:.4f}")
print(f"CV RMSE: {mean_cv_rmse:.4f} ± {std_cv_rmse:.5f}")
print(f"CV R²:   {mean_cv_r2:.4f} ± {std_cv_r2:.4f}")
# -------------------------
# 6) Evaluate & save results
# -------------------------
output_dir = "saved_model"
os.makedirs(output_dir, exist_ok=True)

y_pred = best_rfr.predict(X_test)
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n=== Test Set ===")
print(f"R² = {r2:.4f}, RMSE = {rmse:.4f}")

# Predictions CSV
pred_df = pd.DataFrame({
    "ID": df.loc[test_idx, "ID"].values,
    "Sequence": df.loc[test_idx, "Sequence"].values,
    "True_Permeability": y_test,
    "Predicted_Permeability": y_pred,
    "Residual": y_test - y_pred
})
pred_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

# Model metrics CSV
results = pd.DataFrame([{
    "R2": r2,
    "RMSE": rmse,
    "CV_RMSE_mean": mean_cv_rmse,
    "CV_RMSE_std": std_cv_rmse,
    "CV_R2_mean": mean_cv_r2,
    "CV_R2_std": std_cv_r2,
    "n_samples": len(df),
    "n_features": len(feature_cols),
    **rand_search.best_params_
}])
results.to_csv(os.path.join(output_dir, "model_results.csv"), index=False)

# -------------------------
# 7) Feature importances
# -------------------------
importances = pd.Series(best_rfr.feature_importances_, index=feature_cols).sort_values(ascending=False)
importances.to_csv(os.path.join(output_dir, "feature_importances.csv"), header=["importance"])

plt.figure(figsize=(10, 6))
importances.plot(kind='barh')
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importances')
plt.tight_layout()
plt.savefig('feature_importances.png', dpi=200)
plt.close()

# -------------------------
# 8) Predicted vs True plot
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
plt.savefig('predicted_vs_true.png', dpi=200)
plt.close()

# -------------------------
# 9) Save model & data
# -------------------------
df.to_csv(os.path.join(output_dir, "full_dataset_with_features.csv"), index=False)
joblib.dump(best_pipeline, os.path.join(output_dir, "random_forest_model.joblib"))

print(f"\n[Info] All outputs saved to {output_dir}/")
print("[Info] Done!")