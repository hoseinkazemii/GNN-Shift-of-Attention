import torch
import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed
from tqdm import tqdm


def build_rnn_features(df_t, object_names):
    features = []
    for obj in object_names:
        obj_df = df_t[df_t["Name"] == obj]

        # Default values if object is not fixated
        dist = 0
        dx = 0
        dy = 0
        dz = 0
        gaze_dir_x = 0
        gaze_dir_y = 0
        gaze_dir_z = 0
        gaze_pos_x = 0
        gaze_pos_y = 0
        gaze_pos_z = 0

        if not obj_df.empty:
            dist = obj_df["Min Distance to Powerline (Hit Object)"].iloc[0]
            dx = obj_df["Powerline and Grabbable X Distance"].iloc[0]
            dy = obj_df["Powerline and Grabbable Y Distance"].iloc[0]
            dz = obj_df["Powerline and Grabbable Z Distance"].iloc[0]
            gaze_dir_x = obj_df["Gaze Direction X"].iloc[0]
            gaze_dir_y = obj_df["Gaze Direction Y"].iloc[0]
            gaze_dir_z = obj_df["Gaze Direction Z"].iloc[0]
            gaze_pos_x = obj_df["Gaze Position X"].iloc[0]
            gaze_pos_y = obj_df["Gaze Position Y"].iloc[0]
            gaze_pos_z = obj_df["Gaze Position Z"].iloc[0]

        features.append([
            dist, dx, dy, dz,
        ])

    tensor = torch.tensor(features, dtype=torch.float).flatten()
    return tensor


def create_single_rnn_sequence(i, timeframes, window_size, lookahead_steps, df_critical, object_names):
    window_times = timeframes[i:i + window_size]
    label_times = timeframes[i + window_size:i + window_size + lookahead_steps]

    sequence = []
    for t in window_times:
        df_t = df_critical[df_critical["Timeframe"] == t]
        x = build_rnn_features(df_t, object_names)
        sequence.append(x)

    sequence = torch.stack(sequence)

    df_future = df_critical[df_critical["Timeframe"].isin(label_times)]
    label = int((df_future["Name"] == "PowerLine1").any())

    window_df = df_critical[df_critical["Timeframe"].isin(window_times)]
    source_file = window_df["source_file"].mode().iloc[0]

    return sequence, label, source_file


def build_rnn_dataset(output_path, n_jobs=-1, **params):
    CRITICAL_DIST_THRESHOLD = params.get("dist_threshold")
    lookahead_seconds = params.get("LOOKAHEAD_SECONDS")
    window_size = params.get("window_size")
    FIX_MIN_SEC = params.get("MIN_FIXATION_DURATION")

    print("Starting RNN dataset construction (parallel)...")
    
    df = pd.read_csv("./processed_data/combined_labeled_data.csv")

    df_critical = df[
        (df["Min Distance to Powerline (Grabbable Object)"] < CRITICAL_DIST_THRESHOLD) &
        (df["LoadingStarted"] == 1)
    ].copy()

    label_counts = df_critical[f"label_powerline_future_{lookahead_seconds}_seconds"].value_counts().sort_index()
    label_percentages = df_critical[f"label_powerline_future_{lookahead_seconds}_seconds"].value_counts(normalize=True).sort_index() * 100
    print("\nLabel Distribution in df_critical:")
    for label, count in label_counts.items():
        percent = label_percentages[label]
        print(f"Label {label}: {count} samples ({percent:.2f}%)")

    fixation_duration = df_critical.groupby("Name")["Timeframe"].count() * 0.02
    fixated_objects = fixation_duration[fixation_duration >= FIX_MIN_SEC].index.tolist()

    print(f"Using {len(fixated_objects)} fixated objects.")

    object_names = fixated_objects
    df_critical = df_critical[df_critical["Name"].isin(object_names)].copy()
    timeframes = sorted(df_critical["Timeframe"].unique())

    delta_t = np.diff(timeframes).mean()
    lookahead_steps = int(lookahead_seconds / delta_t) # How many timeframes to look ahead for checking the label

    total_sequences = len(timeframes) - window_size - lookahead_steps
    print(f"Using {len(object_names)} objects, {len(timeframes)} timeframes, Δt = {delta_t:.4f}s")
    print(f"Building {total_sequences:,} sequences...")

    results = Parallel(n_jobs=n_jobs)(
        delayed(create_single_rnn_sequence)(i, timeframes, window_size, lookahead_steps, df_critical, object_names)
        for i in tqdm(range(total_sequences), desc="Building RNN sequences")
    )

    sequences, labels, source_file_names = zip(*results)

    torch.save((sequences, labels, source_file_names),
               os.path.join(output_path, "rnn_dataset_{}_{}.pt".format(CRITICAL_DIST_THRESHOLD, lookahead_steps)))

    print(f"Saved {len(sequences)} RNN sequences to {output_path}/rnn_dataset_{CRITICAL_DIST_THRESHOLD}_{lookahead_steps}.pt")
