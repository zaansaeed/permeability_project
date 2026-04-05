import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import os

def here() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

plt.style.use('default')
sns.set_palette("husl")

# Load data
X = pd.read_csv(here() + '/models_and_training_data/X.csv')
y = pd.read_csv(here() + '/models_and_training_data/y.csv')
rf_model = joblib.load(here() + '/models_and_training_data/random_forest_model.joblib')
rf_model = rf_model['model']

print(f"Data shape: X={X.shape}, y={y.shape}")
print(f"Features: {list(X.columns)}")

# ============================================
# COMPUTE SHAP VALUES
# ============================================
print("\nComputing SHAP values...")
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X)

# ============================================
# IDENTIFY LOGP FEATURES
# ============================================
logp_features = [col for col in X.columns if 'logp' in col.lower()]
print(f"\nFound {len(logp_features)} LogP features: {logp_features}")

# ============================================
# CALCULATE GLOBAL LOGP RANGE (for standardization)
# ============================================
all_logp_values = X[logp_features].values.flatten()
global_logp_min = np.min(all_logp_values)
global_logp_max = np.max(all_logp_values)

# Add some padding (5% on each side)
logp_range = global_logp_max - global_logp_min
logp_padding = logp_range * 0.05
standardized_xlim = (global_logp_min - logp_padding, global_logp_max + logp_padding)

print(f"\nStandardized x-axis range for all LogP features:")
print(f"  Min: {standardized_xlim[0]:.3f}")
print(f"  Max: {standardized_xlim[1]:.3f}")

# ============================================
# CALCULATE GLOBAL SHAP RANGE (for standardization)
# ============================================
logp_indices = [list(X.columns).index(feat) for feat in logp_features]
all_logp_shap = shap_values[:, logp_indices].flatten()
global_shap_min = np.min(all_logp_shap)
global_shap_max = np.max(all_logp_shap)

# Add padding
shap_range = global_shap_max - global_shap_min
shap_padding = shap_range * 0.05
standardized_ylim = (global_shap_min - shap_padding, global_shap_max + shap_padding)

print(f"\nStandardized y-axis range for all SHAP values:")
print(f"  Min: {standardized_ylim[0]:.3f}")
print(f"  Max: {standardized_ylim[1]:.3f}")

# ============================================
# GENERATE PLOTS
# ============================================
logp_combinations = list(combinations(logp_features, 2))
output_dir = '/Users/zaan/Desktop/permeability_project/statistical_plots'
os.makedirs(output_dir, exist_ok=True)

print(f"\n{'=' * 70}")
print("GENERATING STANDARDIZED SHAP DEPENDENCE PLOTS")
print("=" * 70)

# Individual plots
for idx, (feature1, feature2) in enumerate(logp_combinations, 1):
    print(f"[{idx}/{len(logp_combinations)}] {feature1} vs {feature2}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    shap.dependence_plot(
        ind=feature1,
        shap_values=shap_values,
        features=X,
        interaction_index=feature2,
        show=False,
        ax=ax
    )
    
    # APPLY STANDARDIZED AXES
    ax.set_xlim(standardized_xlim)
    ax.set_ylim(standardized_ylim)
    
    ax.set_title(
        f'SHAP Dependence: {feature1} (colored by {feature2})',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel(f'{feature1} Value', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'SHAP value for {feature1}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    filename = f"shap_dependence_{feature1}_vs_{feature2}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {filename}")

# ============================================
# COMPOSITE PLOT WITH STANDARDIZED AXES
# ============================================
print(f"\n{'=' * 70}")
print("CREATING COMPOSITE PLOT (STANDARDIZED AXES)")
print("=" * 70)

n_plots = len(logp_combinations)
n_cols = 3
n_rows = (n_plots + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 5))
axes = axes.flatten()

for idx, (feature1, feature2) in enumerate(logp_combinations):
    print(f"  Subplot {idx + 1}/{n_plots}: {feature1} vs {feature2}")
    
    shap.dependence_plot(
        ind=feature1,
        shap_values=shap_values,
        features=X,
        interaction_index=feature2,
        show=False,
        ax=axes[idx]
    )
    
    # APPLY STANDARDIZED AXES
    axes[idx].set_xlim(standardized_xlim)
    axes[idx].set_ylim(standardized_ylim)
    
    axes[idx].set_title(f'{feature1} × {feature2}', fontsize=10, fontweight='bold')
    axes[idx].set_xlabel(f'{feature1}', fontsize=9)
    axes[idx].set_ylabel(f'SHAP({feature1})', fontsize=9)
    axes[idx].grid(True, alpha=0.2, linestyle='--')

# Hide unused subplots
for idx in range(n_plots, len(axes)):
    axes[idx].axis('off')

fig.suptitle(
    'SHAP Dependence Plots: All LogP Combinations (Standardized Axes)',
    fontsize=16,
    fontweight='bold',
    y=0.995
)

plt.tight_layout()

composite_path = os.path.join(output_dir, 'shap_dependence_all_logp_combinations_standardized.png')
plt.savefig(composite_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ Saved: shap_dependence_all_logp_combinations_standardized.png")

# ============================================
# SUMMARY
# ============================================
print(f"\n{'=' * 70}")
print("ANALYSIS COMPLETE!")
print("=" * 70)
print(f"\nAll plots use standardized axes:")
print(f"  X-axis (LogP values): [{standardized_xlim[0]:.2f}, {standardized_xlim[1]:.2f}]")
print(f"  Y-axis (SHAP values): [{standardized_ylim[0]:.2f}, {standardized_ylim[1]:.2f}]")
print(f"\n✓ Generated {len(logp_combinations)} plots")