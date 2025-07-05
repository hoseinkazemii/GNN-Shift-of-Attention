import pandas as pd
import torch

# Parameters
params = {
    "num_samples": 29,
    "dist_threshold": 80,
    "LOOKAHEAD_SECONDS": 15.0,
}

# Load the labeled dataset
csv_path = f"combined_labeled_data_{params['num_samples']}_participants.csv"
df = pd.read_csv(csv_path)

# Filter df_critical
df_critical = df[
    (df["Min Distance to Powerline (Grabbable Object)"] < params["dist_threshold"]) &
    (df["LoadingStarted"] == 1)
].copy()

# Determine the correct label column name
label_col = f"label_powerline_future_{params['LOOKAHEAD_SECONDS']}_seconds"

# Split into VR and Tobii (PC) based on 'source_file'
df_critical_vr = df_critical[df_critical["source_file"].str.startswith("VREyeTracking")]
df_critical_pc = df_critical[df_critical["source_file"].str.startswith("TobiiEyeTracking")]

# Print label distributions
print("=== VR Label Distribution (df_critical only) ===")
print(df_critical_vr[label_col].value_counts().sort_index())
print()

print("=== PC (Tobii) Label Distribution (df_critical only) ===")
print(df_critical_pc[label_col].value_counts().sort_index())
