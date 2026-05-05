import os
import pandas as pd
import joblib

def here():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

folder = here()

# ── Load model and monomer data ──
rf_obj = joblib.load(os.path.join(folder, "saved_model", "random_forest_model.joblib"))
model = rf_obj.named_steps["model"]
feature_cols = list(pd.read_csv(os.path.join(folder, "saved_model", "X.csv")).columns)

monomer_df = pd.read_csv(os.path.join(folder, "data", "monomer_list_updated.csv"))
monomers = {}
for _, row in monomer_df.iterrows():
    s = row["Symbol"]
    lp, d, ns = row.get("logP"), row.get("is_D"), row.get("is_NSub")
    if pd.isna(lp) or pd.isna(d) or pd.isna(ns):
        continue
    monomers[s] = (float(lp), int(d), int(ns))

# ── Define the sequence ──
sequence = ["Lys(Tfa)", "P", "medl-", "Mono71", "P", "Ala(tBu)"]

# ── Build feature vector ──
row = {}
for i, mon in enumerate(sequence, start=1):
    if mon not in monomers:
        raise ValueError(f"Monomer '{mon}' not found in monomer list!")
    lp, d, ns = monomers[mon]
    row[f"Pos_{i}_logP"] = lp
    row[f"Pos_{i}_is_D"] = d
    row[f"Pos_{i}_is_NSub"] = ns

X_new = pd.DataFrame([row])[feature_cols]

print("Feature vector:")
print(X_new.to_string(index=False))

# ── Predict ──
pred = model.predict(X_new.values)[0]
print(f"\nSequence: {' - '.join(sequence)}")
print(f"Predicted log permeability: {pred:.3f}")