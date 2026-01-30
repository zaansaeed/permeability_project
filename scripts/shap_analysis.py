import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import os

def here() -> str:
    # Directory of this script
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

# Load data
# Load the data
X = pd.read_csv(here() + '/models_and_training_data/X.csv')
y = pd.read_csv(here() + '/models_and_training_data/y.csv')

# Load your trained RF model
rf_model = joblib.load(here() + '/models_and_training_data/random_forest_model.joblib')

# Handle both dict and direct model formats
rf_model = rf_model['model']

print(f"Data shape: X={X.shape}, y={y.shape}")
print(f"Features: {list(X.columns)}")
print(f"Model type: {type(rf_model)}")

# ============================================
# COMPUTE SHAP VALUES
# ============================================
print("\nComputing SHAP values...")

# Create SHAP explainer for tree-based model
explainer = shap.TreeExplainer(rf_model)

# Calculate SHAP values for all samples
shap_values = explainer.shap_values(X)

print(f"SHAP values shape: {shap_values.shape}")

# ============================================
# GENERATE LOGP DEPENDENCE PLOTS
# ============================================
print("\n" + "=" * 70)
print("GENERATING SHAP DEPENDENCE PLOTS FOR LOGP COMBINATIONS")
print("=" * 70)

# Identify LogP features (case-insensitive matching)
logp_features = []
for col in X.columns:
    col_lower = col.lower()
    if 'logp' in col_lower:
        logp_features.append(col)

print(f"\nFound {len(logp_features)} LogP features:")
for feat in logp_features:
    print(f"  - {feat}")

# Generate all unique combinations of LogP features (15 combinations for 6 features)
logp_combinations = list(combinations(logp_features, 2))

print(f"\nGenerating {len(logp_combinations)} unique LogP feature pairs...")
print("=" * 70)

# Create output directory if it doesn't exist
output_dir = here() + '/statistical_plots'
os.makedirs(output_dir, exist_ok=True)

# Generate a plot for each combination
for idx, (feature1, feature2) in enumerate(logp_combinations, 1):
    print(f"\n[{idx}/{len(logp_combinations)}] Creating plot: {feature1} vs {feature2}")
    
    # Create individual plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create dependence plot with feature2 as the interaction feature
    shap.dependence_plot(
        ind=feature1,
        shap_values=shap_values,
        features=X,
        interaction_index=feature2,
        show=False,
        ax=ax
    )
    
    # Enhance the plot
    ax.set_title(
        f'SHAP Dependence: {feature1} (colored by {feature2})',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel(f'{feature1} Value', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'SHAP value for {feature1}', fontsize=12, fontweight='bold')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Create filename (sanitize feature names for filesystem)
    filename = f"shap_dependence_{feature1}_vs_{feature2}.png"
    filename = filename.replace('/', '_').replace('\\', '_')  # Handle any special characters
    filepath = os.path.join(output_dir, filename)
    
    # Save the plot
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {filename}")

# ============================================
# CREATE SUMMARY TABLE
# ============================================
print("\n" + "=" * 70)
print("CREATING SUMMARY TABLE")
print("=" * 70)

# Create a summary of all combinations
summary_data = []
for feature1, feature2 in logp_combinations:
    # Get indices for these features
    feat1_idx = list(X.columns).index(feature1)
    feat2_idx = list(X.columns).index(feature2)
    
    # Calculate correlation between the two features
    correlation = X[feature1].corr(X[feature2])
    
    # Calculate mean absolute SHAP values for each feature
    mean_abs_shap1 = np.abs(shap_values[:, feat1_idx]).mean()
    mean_abs_shap2 = np.abs(shap_values[:, feat2_idx]).mean()
    
    summary_data.append({
        'Feature_1': feature1,
        'Feature_2': feature2,
        'Feature_1_Mean_Abs_SHAP': mean_abs_shap1,
        'Feature_2_Mean_Abs_SHAP': mean_abs_shap2,
        'Feature_Correlation': correlation,
        'Plot_Filename': f"shap_dependence_{feature1}_vs_{feature2}.png"
    })

summary_df = pd.DataFrame(summary_data)

# Save summary table
summary_path = os.path.join(output_dir, 'logp_combinations_summary.csv')
summary_df.to_csv(summary_path, index=False)
print(f"\n✓ Saved summary table: logp_combinations_summary.csv")

print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(summary_df.to_string(index=False))

# ============================================
# CREATE COMPOSITE PLOT (ALL 15 IN ONE FIGURE)
# ============================================
print("\n" + "=" * 70)
print("CREATING COMPOSITE PLOT (ALL 15 COMBINATIONS)")
print("=" * 70)

# Calculate grid dimensions (5 rows x 3 columns for 15 plots)
n_plots = len(logp_combinations)
n_cols = 3
n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 5))
axes = axes.flatten()

for idx, (feature1, feature2) in enumerate(logp_combinations):
    print(f"  Adding subplot {idx + 1}/{n_plots}: {feature1} vs {feature2}")
    
    # Create dependence plot in the subplot
    shap.dependence_plot(
        ind=feature1,
        shap_values=shap_values,
        features=X,
        interaction_index=feature2,
        show=False,
        ax=axes[idx]
    )
    
    # Enhance subplot
    axes[idx].set_title(
        f'{feature1} × {feature2}',
        fontsize=10,
        fontweight='bold'
    )
    axes[idx].set_xlabel(f'{feature1}', fontsize=9)
    axes[idx].set_ylabel(f'SHAP({feature1})', fontsize=9)
    axes[idx].grid(True, alpha=0.2, linestyle='--')

# Hide any unused subplots
for idx in range(n_plots, len(axes)):
    axes[idx].axis('off')

# Add main title
fig.suptitle(
    'SHAP Dependence Plots: All LogP Feature Combinations',
    fontsize=16,
    fontweight='bold',
    y=0.995
)

plt.tight_layout()

# Save composite plot
composite_path = os.path.join(output_dir, 'shap_dependence_all_logp_combinations.png')
plt.savefig(composite_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ Saved composite plot: shap_dependence_all_logp_combinations.png")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)

print(f"\nGenerated {len(logp_combinations)} individual SHAP dependence plots")
print(f"Output directory: {output_dir}")

print("\nFiles created:")
print(f"  - {len(logp_combinations)} individual PNG files (one per LogP combination)")
print(f"  - 1 composite plot with all combinations")
print(f"  - 1 summary CSV file")

print("\nLogP Feature Combinations:")
for i, (f1, f2) in enumerate(logp_combinations, 1):
    print(f"  {i:2d}. {f1:15s} × {f2:15s}")

print("\n" + "=" * 70)
