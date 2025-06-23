import pandas as pd
import os
import glob

def extract_timestamp(filename: str) -> str:
    return filename.split("_")[-1].replace(".csv", "")

def find_loading_start_time(load_file_path: str) -> float:
    df_load = pd.read_csv(load_file_path)
    df_load_started = df_load[df_load["LoadingStarted"] == 1]
    if df_load_started.empty:
        return None
    return df_load_started["Time"].values[0]

def generate_powerline_future_labels_variable_timestep(df, powerline_name="PowerLine1", 
                                                       lookahead_seconds=1.5, min_fixation_duration=0.1):
    df = df.reset_index(drop=True).copy()
    labels = []
    name_series = df["Name"].astype(str).values
    time_series = df["Timeframe"].values
    total_rows = len(df)

    for i in range(total_rows):
        current_time = time_series[i]
        label = 0
        fixation_start = None

        for j in range(i + 1, total_rows):
            time_diff = time_series[j] - current_time
            if time_diff > lookahead_seconds:
                break

            if name_series[j] == powerline_name:
                if fixation_start is None:
                    fixation_start = time_series[j]
                fixation_duration = time_series[j] - fixation_start

                if fixation_duration >= min_fixation_duration:
                    label = 1
                    break
            else:
                fixation_start = None

        labels.append(label)

    df["label_powerline_future_1p5s"] = labels
    return df

data_dir = "./Data/"
vr_files = sorted(glob.glob(os.path.join(data_dir, "VREyeTracking_PowerLineScenario_*.csv")))
tobii_files = sorted(glob.glob(os.path.join(data_dir, "TobiiEyeTracking_PowerLineScenario_*.csv")))
all_files = vr_files + tobii_files
# sample_files = all_files[:5]
sample_files = all_files

LOOKAHEAD_SECONDS = 1.5
MIN_FIXATION_DURATION = 0.1

labeled_dfs = []
for eye_file in sample_files:
    df_eye = pd.read_csv(eye_file)
    df_labeled = generate_powerline_future_labels_variable_timestep(
        df_eye,
        powerline_name="PowerLine1",
        lookahead_seconds=LOOKAHEAD_SECONDS,
        min_fixation_duration=MIN_FIXATION_DURATION
    )
    
    timestamp = extract_timestamp(eye_file)
    load_file = os.path.join(data_dir, f"LoadData_PowerLineScenario_{timestamp}.csv")
    
    if os.path.exists(load_file):
        load_start_time = find_loading_start_time(load_file)
        if load_start_time is not None:
            df_labeled["LoadingStarted"] = (df_labeled["Timeframe"] >= load_start_time).astype(int)
        else:
            print(f"[WARNING] No loading start found for: {eye_file}")
            df_labeled["LoadingStarted"] = 0
    else:
        print(f"[WARNING] No LoadData file found for: {eye_file}")
        df_labeled["LoadingStarted"] = 0

    df_labeled["source_file"] = os.path.basename(eye_file)
    labeled_dfs.append(df_labeled)

df_combined = pd.concat(labeled_dfs, ignore_index=True)

print(df_combined.head())
df_combined.to_csv("combined_labeled_data.csv", index=False)
