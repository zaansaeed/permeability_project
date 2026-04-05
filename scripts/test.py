import joblib
import numpy as np
import pandas as pd

# Load model
pipe = joblib.load("models_and_training_data/random_forest_model.joblib")

# Example: 6 logP values, 6 chirality flags
example = pd.DataFrame([{
    "Pos_1_logP": 0.44,   # e.g. Leu-like
    "Pos_2_logP": 0.44,  # e.g. Pro-like
    "Pos_3_logP": 0.44,   # e.g. Ala-like
    "Pos_4_logP": 0.44,   # e.g. Ile-like
    "Pos_5_logP": -0.17,  # e.g. Gly-like
    "Pos_6_logP": 0.346,   # e.g. Val-like
    "Pos_1_is_D": 0,
    "Pos_2_is_D": 1,
    "Pos_3_is_D": 0,
    "Pos_4_is_D": 1,
    "Pos_5_is_D": 0,
    "Pos_6_is_D": 1,
}])

pred = pipe.predict(example)
print(f"Predicted permeability: {pred[0]:.3f}")