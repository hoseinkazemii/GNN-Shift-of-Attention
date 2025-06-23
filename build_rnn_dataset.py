import torch
import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed
from tqdm import tqdm

LOOKAHEAD_SECONDS = 1.5
CRITICAL_DIST_THRESHOLD = 50
FIX_MIN_SEC = 0.1  # ≥0.1 s fixation
WINDOW_SIZE = 10   # window length (#frames)

df = pd.read_csv("combined_labeled_data.csv")

df_critical = df[
    (df["Min Distance to Powerline (Grabbable Object)"] < CRITICAL_DIST_THRESHOLD) &
    (df["LoadingStarted"] == 1)
].copy()

print(f"Original size: {len(df)}")
print(f"Critical size (distance + loading): {len(df_critical)}")
label_counts = df_critical["label_powerline_future_1p5s"].value_counts().sort_index()
label_percentages = df_critical["label_powerline_future_1p5s"].value_counts(normalize=True).sort_index() * 100
print("\nLabel Distribution in df_critical:")
for label, count in label_counts.items():
    percent = label_percentages[label]
    print(f"Label {label}: {count} samples ({percent:.2f}%)")

fixation_duration = df_critical.groupby("Name")["Timeframe"].count() * 0.02
fixated_objects = fixation_duration[fixation_duration >= FIX_MIN_SEC].index.tolist()

print(f"Using {len(fixated_objects)} fixated objects.")


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
            gaze_dir_x, gaze_dir_y, gaze_dir_z,
            gaze_pos_x, gaze_pos_y, gaze_pos_z
        ])

    tensor = torch.tensor(features, dtype=torch.float).flatten()  # shape [N_OBJ*10]
    return tensor


def create_single_rnn_sequence(i, timeframes, window_size, lookahead_steps, df_critical, object_names):
    window_times = timeframes[i:i + window_size]
    label_times = timeframes[i + window_size:i + window_size + lookahead_steps]

    sequence = []
    for t in window_times:
        df_t = df_critical[df_critical["Timeframe"] == t]
        x = build_rnn_features(df_t, object_names)
        sequence.append(x)

    sequence = torch.stack(sequence)  # [window_size, N_OBJ*10]

    df_future = df_critical[df_critical["Timeframe"].isin(label_times)]
    label = int((df_future["Name"] == "PowerLine1").any())

    window_df = df_critical[df_critical["Timeframe"].isin(window_times)]
    source_file = window_df["source_file"].mode().iloc[0]

    return sequence, label, source_file


def build_rnn_dataset(df_critical, output_path, window_size, lookahead_seconds, n_jobs=-1):
    print("Starting RNN dataset construction (parallel)...")
    os.makedirs(output_path, exist_ok=True)

    object_names = fixated_objects
    df_critical = df_critical[df_critical["Name"].isin(object_names)].copy()
    timeframes = sorted(df_critical["Timeframe"].unique())

    delta_t = np.diff(timeframes).mean()
    lookahead_steps = int(lookahead_seconds / delta_t)

    total_sequences = len(timeframes) - window_size - lookahead_steps
    print(f"Using {len(object_names)} objects, {len(timeframes)} timeframes, Δt = {delta_t:.4f}s")
    print(f"Building {total_sequences:,} sequences...")

    results = Parallel(n_jobs=n_jobs)(
        delayed(create_single_rnn_sequence)(i, timeframes, window_size, lookahead_steps, df_critical, object_names)
        for i in tqdm(range(total_sequences), desc="Building RNN sequences")
    )

    sequences, labels, source_file_names = zip(*results)

    torch.save((sequences, labels, object_names, source_file_names),
               os.path.join(output_path, "rnn_dataset.pt"))

    print(f"Saved {len(sequences)} RNN sequences to {output_path}/rnn_dataset.pt")


build_rnn_dataset(
    df_critical=df_critical,
    output_path="./rnn_data",
    window_size=WINDOW_SIZE,
    lookahead_seconds=LOOKAHEAD_SECONDS
)



# import torch
# import pandas as pd, numpy as np
# from joblib import Parallel, delayed
# from tqdm import tqdm

# LOOKAHEAD_SECONDS = 1.5
# CRIT_DIST         = 50
# FIX_MIN_SEC       = 0.1      # ≥0.1 s total fixation to be kept
# WIN               = 10       # window length (#frames)

# df_all = pd.read_csv("combined_labeled_data.csv")

# df_crit = df_all[
#     (df_all["Min Distance to Powerline (Grabbable Object)"] < CRIT_DIST) &
#     (df_all["LoadingStarted"] == 1)
# ].copy()

# fix_dur  = df_crit.groupby("Name")["Timeframe"].count() * 0.02
# OBJECTS  = fix_dur[fix_dur >= FIX_MIN_SEC].index.tolist()
# N_OBJ, F = len(OBJECTS), 4
# df_crit = df_crit[df_crit["Name"].isin(OBJECTS)].copy()

# def node_features_timestep(df_t):
#     feats = []
#     for obj in OBJECTS:
#         row = df_t[df_t["Name"] == obj]
#         if row.empty:
#             feats.append([0, 0, 0, 0])
#         else:
#             feats.append([
#                 row["Min Distance to Powerline (Hit Object)"].iloc[0],
#                 row["Powerline and Grabbable X Distance"].iloc[0],
#                 row["Powerline and Grabbable Y Distance"].iloc[0],
#                 row["Powerline and Grabbable Z Distance"].iloc[0]
#             ])
#     return torch.tensor(feats, dtype=torch.float).flatten()  # [N_OBJ*F]

# dt        = np.diff(sorted(df_crit["Timeframe"].unique())).mean()
# LA_STEPS  = int(LOOKAHEAD_SECONDS / dt)

# def one_sequence(i, times):
#     win_times = times[i : i + WIN]
#     lbl_times = times[i + WIN : i + WIN + LA_STEPS]

#     x_seq = torch.stack([
#         node_features_timestep(df_crit[df_crit["Timeframe"] == t])
#         for t in win_times
#     ])                                           # shape [WIN, N_OBJ*F]

#     future_df = df_crit[df_crit["Timeframe"].isin(lbl_times)]
#     y = int((future_df["Name"] == "PowerLine1").any())

#     src_file = (df_crit[df_crit["Timeframe"].isin(win_times)]
#                 ["source_file"].mode().iloc[0])
#     return x_seq, y, src_file

# times_all  = sorted(df_crit["Timeframe"].unique())
# SEQ_TOTAL  = len(times_all) - WIN - LA_STEPS
# print(f"Building {SEQ_TOTAL:,} sequences (Δt = {dt:.4f} s)…")

# seqs, labels, srcs = zip(*Parallel(n_jobs=-1)(
#     delayed(one_sequence)(i, times_all) for i in tqdm(range(SEQ_TOTAL))
# ))

# torch.save((seqs, labels, srcs), "rnn_dataset.pt")
