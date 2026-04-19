import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt


def here() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
folder = here()

X      = pd.read_csv(folder + "/saved_model/X.csv")
rf_obj = joblib.load(folder + "/saved_model/random_forest_model.joblib")
model  = rf_obj["model"] if isinstance(rf_obj, dict) and "model" in rf_obj else rf_obj

FEATURES = list(X.columns)


# ─────────────────────────────────────────────
# 3-WAY PDP  (true marginalization)
# ─────────────────────────────────────────────

def plot_3way_pdp(feat1, feat2, feat3, n_grid=25, n_slices=6, n_sample=500):
    f3_vals = np.linspace(X[feat3].min(), X[feat3].max(), n_slices)
    f1_grid = np.linspace(X[feat1].min(), X[feat1].max(), n_grid)
    f2_grid = np.linspace(X[feat2].min(), X[feat2].max(), n_grid)
    F1, F2  = np.meshgrid(f1_grid, f2_grid)
    n_cells = n_grid ** 2

    X_sample = X.sample(min(n_sample, len(X)), random_state=42).reset_index(drop=True)

    all_preds = []
    for f3_val in f3_vals:
        grid_preds = np.zeros(n_cells)

        for _, row in X_sample.iterrows():
            Xm = pd.DataFrame(
                np.tile(row.values, (n_cells, 1)),
                columns=FEATURES
            )
            Xm[feat1] = F1.ravel()
            Xm[feat2] = F2.ravel()
            Xm[feat3] = f3_val
            grid_preds += model.predict(Xm)

        all_preds.append((grid_preds / len(X_sample)).reshape(n_grid, n_grid))

    # Color scale from min/max predicted values across all slices in this PDP
    pdp_vmin = min(p.min() for p in all_preds)
    pdp_vmax = max(p.max() for p in all_preds)

    # ── plot with global color scale ──
    n_cols = 3
    n_rows = (n_slices + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for idx, (f3_val, preds) in enumerate(zip(f3_vals, all_preds)):
        ct = axes[idx].contourf(
            f1_grid, f2_grid, preds,
            levels=20, cmap="RdYlGn", vmin=pdp_vmin, vmax=pdp_vmax
        )
        axes[idx].scatter(
            X[feat1], X[feat2],
            c='#404040', s=5, alpha=0.3, zorder=5
        )
        axes[idx].set_title(f"{feat3} = {f3_val:.2f}", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel(feat1, fontweight="bold")
        axes[idx].set_ylabel(feat2, fontweight="bold")

    for idx in range(n_slices, len(axes)):
        axes[idx].axis("off")

    # Single shared colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(vmin=pdp_vmin, vmax=pdp_vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Predicted Permeability")

    plt.suptitle(
        f"3-Way PDP: {feat1} × {feat2} | {feat3} slices\n"
        f"(other features marginalized over {len(X_sample)} samples)",
        fontsize=13, fontweight="bold"
    )

    fname = f"3way_PDP_{feat1}_{feat2}_{feat3}.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"✓ Saved → {fname}")
    plt.show()


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

plot_3way_pdp('Pos_2_logP', 'Pos_1_logP', 'Pos_6_logP')
plot_3way_pdp('Pos_1_logP', 'Pos_6_logP', 'Pos_2_logP')
plot_3way_pdp('Pos_2_logP', 'Pos_6_logP', 'Pos_1_logP')