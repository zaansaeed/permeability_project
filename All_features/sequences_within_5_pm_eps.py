import pandas as pd

# Load the data
df = pd.read_csv('/Users/zaan/Desktop/permeability_project/All_features/full_dataset_with_features.csv')

# Filter rows where Permeability is between -5.2 and -4.8 (inclusive)
filtered = df[(df['Permeability'] >= -5.2) & (df['Permeability'] <= -4.8)]

# Save to new CSV
output_path = '/Users/zaan/Desktop/permeability_project/All_features/filtered_permeability_dataset.csv'
filtered.to_csv(output_path, index=False)

print(f"Original rows: {len(df)}")
print(f"Filtered rows: {len(filtered)}")
print(f"Saved to: {output_path}")