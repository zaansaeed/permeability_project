# methylated_logP_stats.py
import pandas as pd
import numpy as np

CSV_PATH = "/Users/zaan/Desktop/permeability_project/data/monomer_list.csv"

df = pd.read_csv(CSV_PATH)

# Filter for N-methylated monomers (IUPAC_Condensed starts with "Me-")
methylated = df[df["IUPAC_Condensed"].str.startswith("Me-", na=False)].copy()

print(f"Total monomers: {len(df)}")
print(f"N-methylated monomers: {len(methylated)}")
print()

# logP stats
logp = methylated["MolLogP"]

stats = {
    "min": logp.min(),
    "q25": logp.quantile(0.25),
    "median": logp.median(),
    "mean": logp.mean(),
    "q75": logp.quantile(0.75),
    "max": logp.max(),
    "std": logp.std(),
}

print("=== LogP Distribution of N-Methylated Monomers ===")
for k, v in stats.items():
    print(f"  {k:>7s}: {v:.4f}")