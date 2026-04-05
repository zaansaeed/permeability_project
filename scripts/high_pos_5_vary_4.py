import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def here() -> str:
    # Directory of this script
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

# ============================================
# LOAD DATA AND MODEL
# ============================================
print("=" * 70)
print("LOADING DATA AND MODEL")
print("=" * 70)

# Load the data
X = pd.read_csv(here() + '/models_and_training_data/X.csv')
y = pd.read_csv(here() + '/models_and_training_data/y.csv')

# Load your trained RF model
rf_model = joblib.load(here() + '/models_and_training_data/random_forest_model.joblib')

# Handle both dict and direct model formats
if isinstance(rf_model, dict):
    rf_model = rf_model['model']

print(f"Data shape: X={X.shape}, y={y.shape}")
print(f"Features: {list(X.columns)}")
print(f"Model type: {type(rf_model)}")

# Identify LogP and is_D features
logp_features = [col for col in X.columns if 'logp' in col.lower()]
is_d_features = [col for col in X.columns if 'is_d' in col.lower()]

print(f"\nFound {len(logp_features)} LogP features: {logp_features}")
print(f"Found {len(is_d_features)} is_D features: {is_d_features}")

# ============================================
# GENERATE SYNTHETIC DATA & TEST MODEL
# ============================================
print("\n" + "=" * 70)
print("GENERATING SYNTHETIC DATA FOR POS_4 SWEEP")
print("=" * 70)

# Calculate mean and std for each LogP feature from original data
logp_params = {}
for feature in logp_features:
    logp_params[feature] = {
        'mean': X[feature].mean(),
        'std': X[feature].std(),
        'min': X[feature].min(),
        'max': X[feature].max()
    }

print("\nDistribution parameters:")
for feat, params in logp_params.items():
    print(f"  {feat}: μ={params['mean']:.3f}, σ={params['std']:.3f}")

# Generate synthetic data
n_pos4_steps = 30  # Number of Pos_4 values from high to low
n_samples_per_step = 200  # Random samples at each Pos_4 value

# Pos_4: sweep from high to low
pos4_values = np.linspace(
    logp_params['Pos_4_logP']['max'], 
    logp_params['Pos_4_logP']['min'], 
    n_pos4_steps
)

# Pos_5: keep high (use mean + 1 std as "high")
pos5_high = logp_params['Pos_5_logP']['mean'] + logp_params['Pos_5_logP']['std']

print(f"\nSynthetic data generation:")
print(f"  - Pos_4: {n_pos4_steps} steps from {pos4_values[0]:.3f} to {pos4_values[-1]:.3f}")
print(f"  - Pos_5: fixed at {pos5_high:.3f} (high value)")
print(f"  - Pos 1,2,3,6 logP: sampled from Gaussian distributions")
print(f"  - Pos 1-6 is_D: random binary (0 or 1)")
print(f"  - Total samples: {n_pos4_steps * n_samples_per_step:,}")

# Generate data
synthetic_data = []

for pos4_val in pos4_values:
    for _ in range(n_samples_per_step):
        sample = {}
        
        # Positions 1, 2, 3, 6: random sampling from Gaussian for logP
        for pos in ['Pos_1_logP', 'Pos_2_logP', 'Pos_3_logP', 'Pos_6_logP']:
            val = np.random.normal(
                logp_params[pos]['mean'], 
                logp_params[pos]['std']
            )
            # Clip to original data range
            val = np.clip(val, logp_params[pos]['min'], logp_params[pos]['max'])
            sample[pos] = val
        
        # Position 4: systematic sweep
        sample['Pos_4_logP'] = pos4_val
        
        # Position 5: keep high
        sample['Pos_5_logP'] = pos5_high
        
        # All positions is_D: random binary (0 or 1)
        for is_d_feat in is_d_features:
            sample[is_d_feat] = np.random.choice([0, 1])
        
        synthetic_data.append(sample)

X_synthetic = pd.DataFrame(synthetic_data)

print(f"\n✓ Generated synthetic data: {X_synthetic.shape}")
print(f"\nSample of is_D features distribution:")
for is_d_feat in is_d_features:
    prop_d = X_synthetic[is_d_feat].mean()
    print(f"  {is_d_feat}: {prop_d:.2%} D-stereochemistry")

# ============================================
# PREDICT ON SYNTHETIC DATA
# ============================================
print("\n" + "=" * 70)
print("RUNNING MODEL PREDICTIONS")
print("=" * 70)

# Make predictions
y_pred = rf_model.predict(X_synthetic)

print(f"\nPrediction statistics:")
print(f"  Mean: {y_pred.mean():.4f}")
print(f"  Std:  {y_pred.std():.4f}")
print(f"  Min:  {y_pred.min():.4f}")
print(f"  Max:  {y_pred.max():.4f}")

# Add predictions to dataframe
X_synthetic['Predicted_Permeability'] = y_pred

# Calculate mean prediction for each Pos_4 value
pos4_summary = X_synthetic.groupby('Pos_4_logP').agg({
    'Predicted_Permeability': ['mean', 'std', 'min', 'max']
}).reset_index()
pos4_summary.columns = ['Pos_4_logP', 'Mean_Pred', 'Std_Pred', 'Min_Pred', 'Max_Pred']

# ============================================
# VISUALIZE RESULTS
# ============================================
print("\n" + "=" * 70)
print("CREATING VISUALIZATIONS")
print("=" * 70)

# Plot 1: Predicted permeability vs Pos_4
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Scatter plot with all points
axes[0].scatter(
    X_synthetic['Pos_4_logP'], 
    X_synthetic['Predicted_Permeability'],
    alpha=0.3, s=10, color='steelblue'
)
axes[0].plot(
    pos4_summary['Pos_4_logP'], 
    pos4_summary['Mean_Pred'],
    color='red', linewidth=2, label='Mean prediction'
)
axes[0].fill_between(
    pos4_summary['Pos_4_logP'],
    pos4_summary['Mean_Pred'] - pos4_summary['Std_Pred'],
    pos4_summary['Mean_Pred'] + pos4_summary['Std_Pred'],
    alpha=0.3, color='red', label='±1 SD'
)
axes[0].set_xlabel('Pos_4_logP', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Predicted Permeability', fontsize=12, fontweight='bold')
axes[0].set_title('Effect of Pos_4_logP on Predicted Permeability\n(Pos_5 held high, Pos 1,2,3,6 randomized, is_D random)', 
                  fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Mean prediction trend
axes[1].plot(
    pos4_summary['Pos_4_logP'], 
    pos4_summary['Mean_Pred'],
    color='darkgreen', linewidth=3, marker='o', markersize=4
)
axes[1].set_xlabel('Pos_4_logP', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Mean Predicted Permeability', fontsize=12, fontweight='bold')
axes[1].set_title('Mean Prediction Trend', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("SYNTHETIC DATA ANALYSIS COMPLETE!")
print("=" * 70)
print(f"\nPos_4 Summary (first 10 rows):")
print(pos4_summary.head(10).to_string(index=False))