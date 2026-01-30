ML_Monomers.py - generates the random forest regressor model on the existing dataset in the "data" folder:

permeability_project/data/processed_peptides.csv

ID	Sequence	Permeability	SMILES
2	['dL', 'dL', 'L', 'dL', 'P', 'Y']	-6.2	[H]N[C@H](CC(C)C)C(=O)N[C@H](CC(C)C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@H](CC(C)C)C(=O)N1CCC[C@H]1C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)OC
...
...
...

Separates data into 80% training set, 20% test set

Important results (example)

=== Hyperparameter Tuning Results ===
Best params: {'model__n_estimators': np.int64(200), 'model__min_samples_split': 2, 'model__min_samples_leaf': 1, 'model__max_samples': 1.0, 'model__max_features': 0.7, 'model__max_depth': np.int64(18), 'model__criterion': 'friedman_mse', 'model__bootstrap': True}
Best CV RMSE (neg MSE scoring): 0.410

=== Tuned Random Forest (Test Set) ===
R² (test):  0.779
RMSE (test): 0.359

Feature importances (descending):
Pos_1_logP    0.244927
Pos_2_logP    0.215052
Pos_6_logP    0.208976
Pos_5_logP    0.121205
Pos_3_logP    0.070260
Pos_4_logP    0.035926
Pos_1_is_D    0.027034
Pos_4_is_D    0.020356
Pos_3_is_D    0.020329
Pos_5_is_D    0.019055
Pos_2_is_D    0.015152
Pos_6_is_D    0.001727

=== Saved Outputs ===
X saved to: saved_model/X.csv // all the features (e.g., six logP, and six stereochemistry values
y saved to: saved_model/y.csv // the target value (e.g., experimental permeability)
Feature names saved to: saved_model/feature_names.csv 
Full dataset saved to: saved_model/full_dataset_with_features.csv 
Model saved to: saved_model/random_forest_model.joblib
