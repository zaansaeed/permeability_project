import os
import ast
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors

from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
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
    # Directory of this script
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def save_bar_chart(series: pd.Series, title: str, ylabel: str, outfile: str):
    plt.figure()
    series.plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()

# -------------------------
# 0) Paths & chdir
# -------------------------
os.chdir(here())
print(here())
MONOMER_CSV = os.path.join(here() + "/data/monomer_list.csv")
PEPTIDE_CSV = os.path.join(here() + "/data/processed_peptides.csv")

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
    raise ValueError("monomers_list is empty. Check monomer_list.csv content.")

# -------------------------
# 2) Load peptides & compute per-position logP + stereochemistry
# -------------------------
peptides = load_data(PEPTIDE_CSV)
peptides["Sequence"] = peptides["Sequence"].apply(ast.literal_eval)


LOGP_DICTIONARY = {}
for seq, permeability, pid in zip(peptides["Sequence"], peptides["Permeability"], peptides["ID"]):
    logP_values = []
    chiral_tags = []  # NEW: store D/L for each position
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
        
        # NEW: Extract stereochemistry (1 if D-amino acid, 0 if L)
        # Assuming 'd' prefix indicates D-amino acid in your monomer symbols
        is_D = 1 if monomer.startswith('d') else 0
        chiral_tags.append(is_D)
    
    if bad:
        continue

    LOGP_DICTIONARY[tuple(seq)] = [logP_values, chiral_tags, float(permeability), pid]

if not LOGP_DICTIONARY:
    raise ValueError("LOGP_DICTIONARY is empty. Check sequences/monomer symbols match your monomer list.")

# -------------------------
# 3) Build tidy feature table with stereochemistry
# -------------------------
rows = []
for seq, (logP_values, chiral_tags, permeability, pid) in LOGP_DICTIONARY.items():
    row = {"Permeability": permeability, "Sequence": "-".join(seq), "ID": pid}
    
    # Add LogP features
    for i, v in enumerate(logP_values, start=1):
        row[f"Pos_{i}_logP"] = v
    
    # NEW: Add stereochemistry features
    for i, is_D in enumerate(chiral_tags, start=1):
        row[f"Pos_{i}_is_D"] = is_D
    
    rows.append(row)

df = pd.DataFrame(rows)

# Update numeric columns to include both LogP and stereochemistry
num_cols = ["Permeability"] + \
           [f"Pos_{i}_logP" for i in range(1, 7)] + \
           [f"Pos_{i}_is_D" for i in range(1, 7)]

df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=num_cols).reset_index(drop=True)

# ✅ Exclude invalid permeability rows (e.g., sentinel -10)
df = df[df["Permeability"] != -10.0].reset_index(drop=True)
feature_cols = [c for c in num_cols if c != "Permeability"]



print(f"[Info] Final dataset shape: {df.shape}")
print(f"[Info] Number of features: {len(feature_cols)}")
print(f"[Info] Features: {feature_cols}")
print("\nFirst few rows:")
print(df.head())

# -------------------------
# 4) Split data
# -------------------------
X = df[feature_cols].values
y = df["Permeability"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n[Info] Training set size: {len(X_train)}")
print(f"[Info] Test set size: {len(X_test)}")

# -------------------------
# 5) Hyperparameter tuning (RandomizedSearchCV)
# -------------------------
rfr_pipeline = Pipeline([
    ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
])

param_dist = {
    "model__n_estimators": np.linspace(200, 1000, num=9, dtype=int),
    "model__max_depth": [None] + list(np.arange(4, 26, 2)),
    "model__min_samples_split": [2, 4, 6, 8, 10, 20, 40],
    "model__min_samples_leaf": [1, 2, 3, 4, 5, 8, 10],
    "model__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
    "model__bootstrap": [True, False],
    "model__max_samples": [None, 0.6, 0.8, 1.0],
    "model__criterion": ["squared_error", "friedman_mse"]
}

cv = KFold(n_splits=min(5, len(df)), shuffle=True, random_state=42)

rand_search = RandomizedSearchCV(
    estimator=rfr_pipeline,
    param_distributions=param_dist,
    n_iter=300,
    scoring="neg_mean_squared_error",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\n[Info] Starting hyperparameter tuning...")
rand_search.fit(X_train, y_train)
best_rfrr = rand_search.best_estimator_
best_rfr = best_rfrr.named_steps["model"]
best_cv_rmse = np.sqrt(-rand_search.best_score_)

print("\n=== Hyperparameter Tuning Results ===")
print("Best params:", rand_search.best_params_)
print(f"Best CV RMSE (neg MSE scoring): {best_cv_rmse:.3f}")

# -------------------------
# 6) Evaluate tuned model on test set
# -------------------------
y_pred = best_rfr.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n=== Tuned Random Forest (Test Set) ===")
print(f"R² (test):  {r2:.3f}")
print(f"RMSE (test): {rmse:.3f}")

# -------------------------
# 7) Feature importances
# -------------------------
importances = pd.Series(best_rfr.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature importances (descending):")
print(importances)

# Save feature importance plot
plt.figure(figsize=(10, 6))
importances.plot(kind='barh')
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importances')
plt.tight_layout()
plt.savefig('feature_importances.png', dpi=200)
plt.close()
print("\n[Info] Feature importance plot saved as 'feature_importances.png'")



# -------------------------
# 9) Predicted vs True chart (1:1 line)
# -------------------------
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k', s=70)

# 1:1 perfect prediction line
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
print("[Info] Prediction plot saved as 'predicted_vs_true.png'")

# -------------------------
# 10) Save model and data
# -------------------------
output_dir = "saved_model"
os.makedirs(output_dir, exist_ok=True)

# Save feature names
feature_names_df = pd.DataFrame({'feature': feature_cols})
feature_names_df.to_csv(os.path.join(output_dir, "feature_names.csv"), index=False)

# Save full dataset with features
df.to_csv(os.path.join(output_dir, "full_dataset_with_features.csv"), index=False)

# Save train/test splits
pd.DataFrame(X, columns=feature_cols).to_csv(os.path.join(output_dir, "X.csv"), index=False)
pd.DataFrame(y, columns=["Permeability"]).to_csv(os.path.join(output_dir, "y.csv"), index=False)

# Save trained Random Forest model
model_path = os.path.join(output_dir, "random_forest_model.joblib")
joblib.dump(best_rfrr, model_path)

print(f"\n=== Saved Outputs ===")
print(f"X saved to: {os.path.join(output_dir, 'X.csv')}")
print(f"y saved to: {os.path.join(output_dir, 'y.csv')}")
print(f"Feature names saved to: {os.path.join(output_dir, 'feature_names.csv')}")
print(f"Full dataset saved to: {os.path.join(output_dir, 'full_dataset_with_features.csv')}")
print(f"Model saved to: {model_path}")

print("\n[Info] Done!")