import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

df = pd.read_csv(os.path.join(DATA_DIR, "monomer_list.csv"))

# is_D: check for "D-" in IUPAC_Condensed
# If IUPAC_Condensed is missing or "N.D", fall back to Symbol starting with "d"
def check_is_D(row):
    iupac = str(row.get("IUPAC_Condensed", ""))
    if iupac in ("", "nan", "N.D"):
        return 1 if str(row["Symbol"]).startswith("d") else 0
    return 1 if "D-" in iupac else 0

df["is_D"] = df.apply(check_is_D, axis=1)

# is_NSub: anything that does NOT match canonical alpha-amino acid backbone
# [NX3H2]-[CX4]-[CX3](=O) matches free NH2 on alpha-carbon
# If it doesn't match -> N is substituted (methylated, proline ring, etc.)
canonical_smarts = Chem.MolFromSmarts("[NX3H2]-[CX4]-[CX3](=O)")

def check_nsub(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return 0 if mol.HasSubstructMatch(canonical_smarts) else 1

df["is_NSub"] = df["replaced_SMILES"].apply(check_nsub)

def compute_logP(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return float(Descriptors.MolLogP(mol))

df["logP"] = df["replaced_SMILES"].apply(compute_logP)

out_path = os.path.join(DATA_DIR, "monomer_list_updated.csv")
df.to_csv(out_path, index=False)

print(f"Saved to {out_path}")
print(f"Total: {len(df)}")
print(f"D-amino acids:  {df['is_D'].sum()}")
print(f"N-substituted:  {df['is_NSub'].sum()} (of {df['is_NSub'].notna().sum()} valid)")
print(f"logP computed:  {df['logP'].notna().sum()}, missing: {df['logP'].isna().sum()}")