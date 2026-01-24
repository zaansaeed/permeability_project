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

MODEL_PATH = folder + "/models_and_training_data/random_forest_model.joblib"
X_PATH     = folder + "/models_and_training_data/X.csv"
Y_PATH     = folder + "/models_and_training_data/y.csv"  # not strictly required, but available
MONOMER_LIST_CSV = folder + "/data/monomer_logP.csv"

# === 1) Load model and data ===
pipe = joblib.load(MODEL_PATH)
# If your saved object is a Pipeline with a 'model' step, extract it; otherwise use the object directly
model = getattr(pipe, "named_steps", {}).get("model", pipe)

X = pd.read_csv(X_PATH)
# y = pd.read_csv(Y_PATH)  # optional

FEATURES = list(X.columns)  # e.g., 12 columns: 6 LogP + 6 chirality

# Assume first 6 features are LogP, last 6 are chirality
N_LOGP = 6
N_CHIRAL = 6
LOGP_FEATURES = FEATURES[:N_LOGP]
CHIRALITY_FEATURES = FEATURES[N_LOGP:N_LOGP + N_CHIRAL]

print(f"[Info] LogP features: {LOGP_FEATURES}")
print(f"[Info] Chirality features: {CHIRALITY_FEATURES}")


def target_region(
    target,
    constraints=None,           # {"Pos_1_logP": 1.25} or {"Pos_3_logP": (0.0, 2.0)}
    eps=0.05,                   # tolerance on |y_pred - target|
    n_samples=20000,            # how many candidates to try
    jitter_frac=0.5,           # add small noise to broaden coverage
    include_shap=False          # set True if you want SHAP overlay (needs TreeExplainer)
):
    # Create random number generator
    rng = np.random.default_rng(0)
    
    # Create base dataset, using sampling with replacement if n_samples > len(X)
    base = X.sample(n=max(len(X), n_samples), replace=len(X) < n_samples, random_state=0).copy()

    # === SPLIT HANDLING: LogP gets Gaussian jitter, Chirality gets random 0/1 ===
    
    # 1) Handle LogP features: add Gaussian jitter
    logp_values = base[LOGP_FEATURES].values
    logp_std = X[LOGP_FEATURES].std().replace(0, 1e-9).values
    logp_jitter = rng.normal(0, 1, size=(len(base), N_LOGP)) * (jitter_frac * logp_std)
    jittered_logp = logp_values + logp_jitter
    
    # 2) Handle Chirality features: generate random 0s and 1s
    random_chirality = rng.integers(0, 2, size=(len(base), N_CHIRAL))
    
    # 3) Combine into candidate dataframe
    cand = pd.DataFrame(
        np.hstack([jittered_logp, random_chirality]),
        columns=FEATURES
    )

    # Enforce constraints
    if constraints:
        for raw_key, v in constraints.items():
            col = raw_key
            if col is None or col not in cand.columns:
                print(f"[target_region] Skipping constraint for '{raw_key}' — no matching feature in FEATURES={FEATURES}.")
                continue
            
            
            if isinstance(v, (tuple, list)):
                lo, hi = float(v[0]), float(v[1])
                cand[col] = cand[col].clip(lo, hi)
            else:
                cand[col] = float(v)

    # Keep candidates within robust bounds (1st–99th pct) for LogP features only
    ql, qh = X[LOGP_FEATURES].quantile(0.01), X[LOGP_FEATURES].quantile(0.99)
    for c in LOGP_FEATURES:
        if c not in constraints: # only clip if not already constrained
            cand[c] = cand[c].clip(float(ql[c]), float(qh[c]))

    # Ensure chirality features are valid binary values (0 or 1)
    for c in CHIRALITY_FEATURES:
        cand[c] = cand[c].round().clip(0, 1).astype(int)

    # Predict and filter to those near the target
    try:
        yhat = pipe.predict(cand[FEATURES])
    except Exception:
        yhat = model.predict(cand[FEATURES])
    yhat = np.asarray(yhat).ravel()

    mask = np.abs(yhat - target) <= eps  # only keep candidates within eps of target
    hit = cand.loc[mask].copy()
    hit["y_pred"] = yhat[mask]

    if hit.empty:
        return None, {"msg": f"No candidates within ±{eps} of target. Try increasing eps, n_samples, or jitter_frac."}

    monomers = pd.read_csv(MONOMER_LIST_CSV)
    
    # Summarize feasible ranges
    def preview_list(lst, k=8):
        lst = list(lst)
        return lst if len(lst) <= k else lst[:k] + [f"...(+{len(lst)-k} more)"]

    # Summary for LogP features only (chirality is random)
    summary = pd.DataFrame({
        "min": hit[LOGP_FEATURES].min(),
        "q25": hit[LOGP_FEATURES].quantile(0.25),
        "median": hit[LOGP_FEATURES].median(),
        "q75": hit[LOGP_FEATURES].quantile(0.75),
        "max": hit[LOGP_FEATURES].max(),
        "possible_monomers": [
            preview_list(
                monomers.loc[
                    (monomers["logP"] >= hit[c].min()) &
                    (monomers["logP"] <= hit[c].max()),
                    "Symbol"
                ].dropna().drop_duplicates().tolist(),
                k=3
            )
            for c in LOGP_FEATURES
        ]
    })

    out = {
        "n_candidates": int(len(cand)),
        "n_hits": int(len(hit)),
        "hit_rate": float(len(hit)/len(cand)),
        "target": float(target),
        "eps": float(eps)
    }

    # Optional: SHAP on the hit set to explain which features "made" the target
    shap_summary = None
    if include_shap:
        try:
            import shap
            tree_model = getattr(pipe, "named_steps", {}).get("model", model)
            explainer = shap.TreeExplainer(tree_model)
            phi = explainer.shap_values(hit[FEATURES].values)
            shap_summary = {
                "mean_abs_shap": dict(zip(FEATURES, np.mean(np.abs(phi), axis=0).tolist()))
            }
        except Exception as e:
            shap_summary = {"error": f"SHAP failed: {e}"}

    examples = hit.sample(min(10, len(hit)), random_state=1)
    examples = examples[FEATURES + ["y_pred"]]

    return {"feasible_ranges": summary, "examples": examples}, {**out, "shap": shap_summary}


# === 7) Example usage ===
if __name__ == "__main__":
    TARGET = -5  # Target Permeability value
    
    # Use full feature names for constraints (assuming features are named like Pos_1_logP, Pos_2_logP, etc.)
    CONSTRAINTS = {
        LOGP_FEATURES[1]: (1.0,1.2),      # Second LogP feature must be between 1.0-1.2
        LOGP_FEATURES[2]: (0, 4.0),     # Third LogP feature can be from 2.0-2.3
    }
    
    y = pd.read_csv(Y_PATH).squeeze()

    res, meta = target_region(TARGET, constraints=CONSTRAINTS, eps=0.06, n_samples=30000, include_shap=True)
    
    if res is None:
        print(meta["msg"])
    else:
        print("=== Feasible ranges (hits only) - LogP features ===")
        print(res["feasible_ranges"].to_string())
        print("\n=== Example designs (first 10 sampled hits) ===")
        print(res["examples"].head(10).to_string(index=False))
        print("\nMETA:", meta)