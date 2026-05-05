import os
import re
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

df = pd.read_csv(os.path.join(DATA_DIR, "monomer_list.csv"))

# Filter to backbone monomers only
df = df[df["Monomer_Type"] == "Backbone"].reset_index(drop=True)

# Chirality: 1 = D, -1 = L, 0 = neither
d_symbol_pattern = re.compile(r"d[A-Z]")

def check_chirality(row):
    iupac = str(row.get("IUPAC_Name", ""))
    symbol = str(row.get("Symbol", ""))
    is_D = ("2R" in iupac) or bool(d_symbol_pattern.search(symbol))
    is_L = "2S" in iupac
    if is_L:
        return -1
    if is_D:
        return 1
    return 0

df["chirality"] = df.apply(check_chirality, axis=1)

# is_NSub: 0 if "2-amino" in IUPAC_Name OR "O->S" in Symbol, else 1
def check_nsub(row):
    iupac = str(row.get("IUPAC_Name", ""))
    symbol = str(row.get("Symbol", ""))
    if "2-amino" in iupac or "O->S" in symbol:
        return 0
    return 1

df["is_NSub"] = df.apply(check_nsub, axis=1)

def compute_logP(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return float(Descriptors.MolLogP(mol))

df["logP"] = df["replaced_SMILES"].apply(compute_logP)

out_path = os.path.join(DATA_DIR, "monomer_list_updated.csv")
df.to_csv(out_path, index=False)

print(f"Saved to {out_path}")
print(f"Total backbone monomers: {len(df)}")
print(f"D-amino acids (1):       {(df['chirality'] == 1).sum()}")
print(f"L-amino acids (-1):      {(df['chirality'] == -1).sum()}")
print(f"Neither (0):             {(df['chirality'] == 0).sum()}")
print(f"N-substituted:           {df['is_NSub'].sum()}")
print(f"logP computed:           {df['logP'].notna().sum()}, missing: {df['logP'].isna().sum()}")