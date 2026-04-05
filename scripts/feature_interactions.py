import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

def here() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================
# LOAD DATA AND MODEL
# ============================================
print("=" * 70)
print("3-WAY ANALYSIS: Pos_1_logP × Pos_2_logP × Pos_6_logP")
print("=" * 70)

X = pd.read_csv(here() + "/models_and_training_data/X.csv")
y = pd.read_csv(here() + "/models_and_training_data/y.csv")  # optional; unused below

rf_obj = joblib.load(here() + "/models_and_training_data/random_forest_model.joblib")
# support either {"model": ...} or a pipeline/model directly
model = rf_obj["model"] if isinstance(rf_obj, dict) and "model" in rf_obj else rf_obj

features = ["Pos_1_logP", "Pos_2_logP", "Pos_6_logP"]

# ============================================
# CONDITIONAL EFFECT ANALYSIS (ON-MANIFOLD)
# ============================================
print("\n" + "=" * 70)
print("CONDITIONAL EFFECT ANALYSIS (ON-MANIFOLD)")
print("=" * 70)

RANDOM_STATE = 0
N_SWEEP = 40          # points along the sweep curve
N_SAMPLES = 30        # real rows to average per context (reduce if dataset is tiny)
MIN_GROUP_N = 8       # skip contexts smaller than this

def make_mask(series: pd.Series, label: str) -> pd.Series:
    med = series.median()
    return (series <= med) if label == "Low" else (series > med)

def safe_group_sweep_range(df: pd.DataFrame, col: str, n_points: int):
    # Sweep only within observed range inside that context group
    lo = df[col].min()
    hi = df[col].max()
    if np.isclose(lo, hi):
        # Degenerate range: return constant line
        return np.array([lo] * n_points)
    return np.linspace(lo, hi, n_points)

def predict_mean_over_real_rows(rows: pd.DataFrame, swept_feature: str, f_val: float) -> float:
    rows_mod = rows.copy()
    rows_mod[swept_feature] = f_val
    return float(np.mean(model.predict(rows_mod)))

for swept_feature in features:
    ctx = [f for f in features if f != swept_feature]
    F_ctx1, F_ctx2 = ctx

    print(f"\n{'─' * 70}")
    print(f"Sweeping: {swept_feature} | Context: ({F_ctx1}, {F_ctx2})")
    print(f"{'─' * 70}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    cond_results = []

    conditions = [("Low", "Low"), ("Low", "High"), ("High", "Low"), ("High", "High")]

    for idx, (s1, s2) in enumerate(conditions):
        mask = make_mask(X[F_ctx1], s1) & make_mask(X[F_ctx2], s2)
        group = X.loc[mask]

        n_group = len(group)
        if n_group < MIN_GROUP_N:
            axes[idx].axis("off")
            axes[idx].set_title(f"{F_ctx1}={s1}, {F_ctx2}={s2}\nN={n_group} (skipped)", fontweight="bold")
            cond_results.append({
                "condition": f"{F_ctx1}={s1}, {F_ctx2}={s2}",
                "slope": np.nan, "r2": np.nan, "n": n_group
            })
            continue

        # sample real rows (on-manifold)
        rows = group.sample(n=min(N_SAMPLES, n_group), random_state=RANDOM_STATE)

        # sweep only within this group's observed range
        f_sweep = safe_group_sweep_range(group, swept_feature, N_SWEEP)

        preds = np.array([predict_mean_over_real_rows(rows, swept_feature, f_val) for f_val in f_sweep])

        # slope summary (linear fit) + also report r2
        slope, intercept, r_val, _, _ = linregress(f_sweep, preds)

        cond_results.append({
            "condition": f"{F_ctx1}={s1}, {F_ctx2}={s2}",
            "slope": float(slope),
            "r2": float(r_val**2),
            "n": int(n_group),
            "x_min": float(f_sweep.min()),
            "x_max": float(f_sweep.max()),
        })

        axes[idx].plot(f_sweep, preds, linewidth=2.5)
        axes[idx].set_title(
            f"{F_ctx1}={s1}, {F_ctx2}={s2} (N={n_group})\n"
            f"Slope={slope:+.4f}, R²={r_val**2:.3f}",
            fontweight="bold",
            fontsize=11,
        )
        axes[idx].set_xlabel(swept_feature, fontweight="bold")
        axes[idx].set_ylabel("Predicted Permeability", fontweight="bold")
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle(
        f"{swept_feature} Effect in Different ({F_ctx1}, {F_ctx2}) Contexts\n"
        f"(averaged over real samples; context-specific sweep range)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    outname = f"conditional_effects_onmanifold_{swept_feature}.png"
    plt.savefig(outname, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✓ Saved {outname}")

    # Print table
    print("\nSlope comparison (higher = more positive effect):")
    sortable = [r for r in cond_results if not np.isnan(r["slope"])]
    for r in sorted(sortable, key=lambda x: x["slope"], reverse=True):
        print(
            f"  {r['condition']:30s} | Slope: {r['slope']:+.4f} | R²={r['r2']:.3f} | "
            f"N={r['n']} | sweep=[{r['x_min']:.2f},{r['x_max']:.2f}]"
        )

    if sortable:
        slope_range = max(r["slope"] for r in sortable) - min(r["slope"] for r in sortable)
        print(f"\nSlope range (across valid contexts): {slope_range:.4f}")
        if slope_range > 0.1:
            print(f"→ {swept_feature} effect is HIGHLY context-dependent")
        elif slope_range > 0.05:
            print(f"→ {swept_feature} has moderate context-dependence")
        else:
            print(f"→ {swept_feature} effect is relatively stable")
    else:
        print("\nNo valid contexts (all groups too small).")

print(f"\n{'=' * 70}")
print("✓ Analysis complete.")
print("=" * 70)