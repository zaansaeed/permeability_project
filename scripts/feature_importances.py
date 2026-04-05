import os
import joblib
import pandas as pd

# ─────────────────────────────────────────────
# Paths — adjust if your saved_model folder is elsewhere
# ─────────────────────────────────────────────
MODEL_PATH = "saved_model/random_forest_model.joblib"
FEATURES_PATH = "saved_model/feature_names.csv"

# ─────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────
rf_obj = joblib.load(MODEL_PATH)
model = rf_obj.named_steps["model"] if hasattr(rf_obj, "named_steps") else rf_obj

feature_names = pd.read_csv(FEATURES_PATH)["feature"].tolist()

# ─────────────────────────────────────────────
# Feature importances
# ─────────────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=feature_names)
importances = importances.sort_values(ascending=False)

print("Feature Importances (MDI)\n")
print(f"{'Feature':<20} {'Importance':>12} {'% Total':>10}")
print("-" * 44)

total = importances.sum()
for feat, imp in importances.items():
    print(f"{feat:<20} {imp:>12.4f} {imp/total*100:>9.2f}%")

print("-" * 44)
print(f"{'Total':<20} {total:>12.4f} {100.0:>9.2f}%")

# Top 3 cumulative
top3 = importances.head(3)
print(f"\nTop 3 features account for {top3.sum()/total*100:.2f}% of total importance")