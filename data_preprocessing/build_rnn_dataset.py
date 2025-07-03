import torch
import pandas as pd
import numpy as np
import os
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


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


def create_single_rnn_sequence(i, window_size, df_critical, object_names, source_file, lookahead_seconds):
    timeframes = df_critical["Timeframe"].unique().tolist()
    window_times = timeframes[i:i + window_size + 1]
    t_start = window_times[-1]

    sequence = []
    for t in window_times:
        df_t = df_critical[df_critical["Timeframe"] == t]
        x = build_rnn_features(df_t, object_names)
        sequence.append(x)

    sequence = torch.stack(sequence)

    label_col = f"label_powerline_future_{lookahead_seconds}_seconds"
    label = int(df_critical[df_critical["Timeframe"] == t_start][label_col].iloc[0])

    return sequence, label, source_file


def filter_fixation_episodes(df, min_fix_duration):
    df = df.sort_values(["source_file", "Timeframe"]).copy()
    df["prev_name"] = df["Name"].shift(1)
    df["same_as_prev"] = df["Name"] == df["prev_name"]
    df["episode_id"] = (~df["same_as_prev"]).cumsum()

    fixation_groups = (
        df.groupby(["source_file", "episode_id", "Name"])
        .agg({"Timeframe": "count"})
        .reset_index()
    )
    fixation_groups["duration"] = fixation_groups["Timeframe"] * 0.02

    valid_episodes = fixation_groups[
        fixation_groups["duration"] >= min_fix_duration
    ][["source_file", "episode_id"]]

    df = df.merge(valid_episodes, on=["source_file", "episode_id"], how="inner")
    return df.drop(columns=["prev_name", "same_as_prev"])


def build_rnn_dataset(output_path, n_jobs=-1, **params):
    CRITICAL_DIST_THRESHOLD = params.get("dist_threshold")
    lookahead_seconds = params.get("LOOKAHEAD_SECONDS")
    window_size = params.get("window_size")
    FIX_MIN_SEC = params.get("MIN_FIXATION_DURATION")
    num_samples = params.get("num_samples")

    print("Starting RNN dataset construction (parallel)...")

    df = pd.read_csv("./processed_data/combined_labeled_data_{}_participants.csv".format(int(num_samples)))

    df_critical = df[
        (df["Min Distance to Powerline (Grabbable Object)"] < CRITICAL_DIST_THRESHOLD) &
        (df["LoadingStarted"] == 1)
    ].copy()

    print(f"df_critical['label_powerline_future_{lookahead_seconds}_seconds'] == 1] = ", len(df_critical[df_critical[f'label_powerline_future_{lookahead_seconds}_seconds'] == 1]))
    print("**" * 50)
    print(f"df_critical['label_powerline_future_{lookahead_seconds}_seconds'] == 0] = ", len(df_critical[df_critical[f'label_powerline_future_{lookahead_seconds}_seconds'] == 0]))
    print("**" * 50)
    print(f"df_critical.shape", df_critical.shape)


    df_fixated_objects = filter_fixation_episodes(df_critical, FIX_MIN_SEC)

    fixated_objects = df_fixated_objects["Name"].unique().tolist()
    object_names = [obj for obj in fixated_objects if obj != "PowerLine1"]
    print(f"Using {len(object_names)} objects for feature extraction (excluding PowerLine1).")

    feature_cols = [
        "Min Distance to Powerline (Hit Object)",
        "Powerline and Grabbable X Distance",
        "Powerline and Grabbable Y Distance",
        "Powerline and Grabbable Z Distance",
        "Gaze Direction X", "Gaze Direction Y", "Gaze Direction Z",
        "Gaze Position X", "Gaze Position Y", "Gaze Position Z"
    ]
    scaler = StandardScaler()
    df_critical[feature_cols] = scaler.fit_transform(df_critical[feature_cols])
    joblib.dump(scaler, os.path.join(output_path, f"scaler_rnn_{CRITICAL_DIST_THRESHOLD}_{lookahead_seconds}.joblib"))

    all_results = []
    grouped = df_critical.groupby("source_file")

    for participant_id, df_participant in grouped:
        df_participant = df_participant.sort_values("Timeframe").reset_index(drop=True)
        timeframes = df_participant["Timeframe"].unique().tolist()

        if len(timeframes) <= window_size:
            continue

        print(f"Processing participant: {participant_id} with {len(timeframes)} frames")

        valid_start_indices = []
        for i in range(len(timeframes) - window_size):
            t_start = timeframes[i]
            t_lookahead_target = t_start + lookahead_seconds

            future_candidates = [t for t in timeframes[i + window_size:] if t >= t_lookahead_target]
            if not future_candidates:
                continue
            valid_start_indices.append((i, future_candidates[0]))

        print(f"Valid sequences for {participant_id}: {len(valid_start_indices)}")

        results = Parallel(n_jobs=n_jobs)(
            delayed(create_single_rnn_sequence)(
                i, window_size, df_participant, object_names,
                participant_id, lookahead_seconds
            )
            for (i, t_future) in tqdm(valid_start_indices, desc=f"Building sequences for {participant_id}")
        )
        all_results.extend(results)

    sequences, labels, source_file_names = zip(*all_results)
    sequence_tensor = torch.stack(sequences)

    labels_tensor = torch.tensor(labels)
    print("\nLabel distribution in saved dataset:")
    print("Positive:", (labels_tensor == 1).sum().item())
    print("Negative:", (labels_tensor == 0).sum().item())

    torch.save((sequence_tensor, labels, source_file_names),
               os.path.join(output_path, f"rnn_dataset_{CRITICAL_DIST_THRESHOLD}_{lookahead_seconds}_{FIX_MIN_SEC}_{int(num_samples)}_participants.pt"))

    print(f"\nSaved {len(sequences)} sequences across {len(grouped)} participants.")
