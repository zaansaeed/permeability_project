import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

def here() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load data
X = pd.read_csv(here() + '/saved_model/X.csv')
rf_model = joblib.load(here() + '/saved_model/random_forest_model.joblib')
rf_model = rf_model['model']

# ============================================
# COMPUTE SHAP VALUES
# ============================================
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X)

# If regression → shap_values is (n_samples, n_features)
# If classification → take class 1 (or adjust as needed)
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# ============================================
# BEESWARM PLOT
# ============================================
plt.figure()

shap.summary_plot(
    shap_values,
    X,
    plot_type="dot",   # THIS = beeswarm
    show=False
)

plt.title("SHAP Beeswarm Plot")
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=300)
plt.close()

print("✓ Saved: shap_beeswarm.png")