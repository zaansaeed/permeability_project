import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def here() -> str:
    # Directory of this script
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load the data
X = pd.read_csv(here() + '/models_and_training_data/X.csv')
y = pd.read_csv(here() + '/models_and_training_data/y.csv')

# Combine data
data = X.copy()
data['permeability'] = y.iloc[:, 0].values

print("="*70)
print("OPTIMAL FEATURE RANGES FOR HIGH PERMEABILITY")
print("="*70)

# Define high permeability as top 25%
high_perm_threshold = data['permeability'].quantile(0.75)
low_perm_threshold = data['permeability'].quantile(0.25)

print(f"\nHigh permeability threshold (top 25%): {high_perm_threshold:.3f}")
print(f"Low permeability threshold (bottom 25%): {low_perm_threshold:.3f}")

# Split data
high_perm = data[data['permeability'] >= high_perm_threshold]
low_perm = data[data['permeability'] <= low_perm_threshold]

print("\n" + "-"*70)
print("OPTIMAL RANGES FOR HIGH PERMEABILITY (Top 25%)")
print("-"*70)

for col in X.columns:
    high_mean = high_perm[col].mean()
    high_std = high_perm[col].std()
    high_median = high_perm[col].median()
    high_min = high_perm[col].min()
    high_max = high_perm[col].max()
    
    low_mean = low_perm[col].mean()
    
    print(f"\n{col}:")
    print(f"  High perm range: [{high_min:.3f}, {high_max:.3f}]")
    print(f"  High perm mean: {high_mean:.3f} ± {high_std:.3f}")
    print(f"  High perm median: {high_median:.3f}")
    print(f"  Low perm mean: {low_mean:.3f}")
    print(f"  Difference: {high_mean - low_mean:+.3f}")

# Create visualization comparing high vs low permeability
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()

for idx, col in enumerate(X.columns):
    ax = axes[idx]
    
    # Box plots for high vs low permeability
    bp_data = [low_perm[col], high_perm[col]]
    bp = ax.boxplot(bp_data, labels=['Low\nPerm\n(Bottom 25%)', 'High\nPerm\n(Top 25%)'],
                    patch_artist=True, widths=0.6)
    
    # Color the boxes
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightgreen')
    
    ax.set_title(col, fontsize=11, fontweight='bold')
    ax.set_ylabel('Feature Value', fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    
    # Add mean markers
    means = [low_perm[col].mean(), high_perm[col].mean()]
    ax.plot([1, 2], means, 'ro', markersize=8, label='Mean')

plt.suptitle('Feature Value Distributions: Low vs High Permeability', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('optimal_feature_ranges.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a heatmap showing optimal ranges
print("\n" + "="*70)
print("CREATING OPTIMAL RANGE HEATMAP")
print("="*70)

# Calculate mean values for different permeability bins
n_bins = 5
data['perm_bin'] = pd.qcut(data['permeability'], q=n_bins, 
                            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

# Create heatmap data
heatmap_data = data.groupby('perm_bin')[X.columns].mean()

plt.figure(figsize=(14, 6))
sns.heatmap(heatmap_data.T, annot=True, fmt='.2f', cmap='RdYlGn', 
            center=0, linewidths=0.5, cbar_kws={"label": "Mean Feature Value"})
plt.title('Mean Feature Values Across Permeability Bins', fontsize=14, fontweight='bold')
plt.xlabel('Permeability Category', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('feature_values_by_permeability_bins.png', dpi=300, bbox_inches='tight')
plt.show()

# Identify optimal setpoints for each feature
print("\n" + "="*70)
print("RECOMMENDED SETPOINTS FOR HIGH PERMEABILITY")
print("="*70)

for col in X.columns:
    high_mean = high_perm[col].mean()
    
    if 'is_D' in col:
        # For binary features
        pct = high_mean * 100
        recommendation = "D-amino acid" if high_mean > 0.5 else "L-amino acid"
        print(f"\n{col}:")
        print(f"  Recommendation: {recommendation} ({pct:.1f}% D in high perm peptides)")
    else:
        # For continuous features (logP)
        optimal_range = (high_perm[col].quantile(0.25), high_perm[col].quantile(0.75))
        print(f"\n{col}:")
        print(f"  Optimal range (IQR): [{optimal_range[0]:.3f}, {optimal_range[1]:.3f}]")
        print(f"  Target value: {high_mean:.3f}")