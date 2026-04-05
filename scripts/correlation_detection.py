# inverse_design.py
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

def here() -> str:
    # Directory of this script
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# === 0) Paths (same pattern as your shap_analysis.py) ===
os.chdir(os.path.dirname(os.path.abspath(__file__)))
folder = here()

X_PATH     = folder + "/models_and_training_data/X.csv"
Y_PATH     = folder + "/models_and_training_data/y.csv"  # not strictly required, but available
MONOMER_LIST_CSV = folder + "/data/monomer_logP.csv"


X = pd.read_csv(X_PATH)
# y = pd.read_csv(Y_PATH)  # optional

FEATURES = list(X.columns)  # e.g., 12 columns: 6 LogP + 6 chirality

# Assume first 6 features are LogP, last 6 are chirality
N_LOGP = 6
N_CHIRAL = 6
LOGP_FEATURES = FEATURES[:N_LOGP]
CHIRALITY_FEATURES = FEATURES[N_LOGP:N_LOGP + N_CHIRAL]

import seaborn as sns
import matplotlib.pyplot as plt

# Correlation matrix for logP features only
logp_df = X[LOGP_FEATURES]

# Heatmap
sns.heatmap(logp_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Pairwise LogP Correlations")
plt.tight_layout()
plt.savefig("pairwise_correlations.png", dpi=600, bbox_inches="tight")
plt.show()
