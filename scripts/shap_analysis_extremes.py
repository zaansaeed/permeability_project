import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


def here():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


folder = here()
OUTPUT_DIR = os.path.join(folder, "shap_analysis_of_extremes")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load model and data ──
X = pd.read_csv(os.path.join(folder, "saved_model", "X.csv"))
rf_obj = joblib.load(os.path.join(folder, "saved_model", "random_forest_model.joblib"))
model = rf_obj.named_steps["model"]
feature_cols = list(X.columns)

# ── Load full dataset ──
full_df = pd.read_csv(os.path.join(folder, "saved_model", "full_dataset_with_features.csv"))

# ── Find candidates ──
high_p1 = full_df["Pos_1_logP"] >= 1.5
red_mask = high_p1 & (full_df["Pos_2_logP"] >= 0.5) & (full_df["Pos_6_logP"] >= 0.5)
green_mask = high_p1 & (full_df["Pos_2_logP"] <= -0.3) & (full_df["Pos_6_logP"] <= -0.3)

red_row = full_df[red_mask].sort_values("Permeability", ascending=True).iloc[0]
green_row = full_df[green_mask].sort_values("Permeability", ascending=False).iloc[0]

print(f"RED:   {red_row['Sequence']}, Perm = {red_row['Permeability']:.2f}")
print(f"GREEN: {green_row['Sequence']}, Perm = {green_row['Permeability']:.2f}")

# ── SHAP ──
red_features = red_row[feature_cols].values.astype(float).reshape(1, -1)
green_features = green_row[feature_cols].values.astype(float).reshape(1, -1)

explainer = shap.TreeExplainer(model, X.values)
shap_red = explainer(red_features)
shap_green = explainer(green_features)
shap_red.feature_names = feature_cols
shap_green.feature_names = feature_cols

# ── Save candidate info ──
for label, row in [("red", red_row), ("green", green_row)]:
    pred = model.predict(row[feature_cols].values.astype(float).reshape(1, -1))[0]
    info = pd.DataFrame([{
        "Sequence": row["Sequence"],
        "True_Permeability": row["Permeability"],
        "Predicted_Permeability": pred,
        **{f: row[f] for f in feature_cols},
    }])
    info.to_csv(os.path.join(OUTPUT_DIR, f"{label}_candidate.csv"), index=False)
    print(f"  {label.upper()} predicted: {pred:.2f}, true: {row['Permeability']:.2f}")

# ── Waterfall plots ──
shap.plots.waterfall(shap_red[0], max_display=18, show=False)
plt.title(f"RED (low perm): {red_row['Sequence']}\nPerm = {red_row['Permeability']:.2f}")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_waterfall_red.png"), dpi=200, bbox_inches="tight")
plt.show()

shap.plots.waterfall(shap_green[0], max_display=18, show=False)
plt.title(f"GREEN (high perm): {green_row['Sequence']}\nPerm = {green_row['Permeability']:.2f}")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_waterfall_green.png"), dpi=200, bbox_inches="tight")
plt.show()

# ── Save SHAP values ──
for label, sv in [("red", shap_red), ("green", shap_green)]:
    shap_df = pd.DataFrame([{
        "feature": f,
        "feature_value": sv[0].data[i],
        "shap_value": sv[0].values[i],
    } for i, f in enumerate(feature_cols)])
    shap_df["base_value"] = sv[0].base_values
    shap_df.to_csv(os.path.join(OUTPUT_DIR, f"{label}_shap_values.csv"), index=False)