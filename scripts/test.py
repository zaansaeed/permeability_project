import joblib
import numpy as np
import pandas as pd




# Load model

from rdkit import Chem

from rdkit.Chem import Descriptors, rdMolDescriptors





mol = Chem.MolFromSmiles("O=C(O)[C@@H]1CCCN1")
logP = Descriptors.MolLogP(mol)

print(logP)