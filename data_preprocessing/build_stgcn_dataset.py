import numpy as np
import os
import torch
from joblib import Parallel, delayed
from tqdm import tqdm


def build_node_features(df_t, object_names):
    """
    Given a dataframe for a single timestep, returns node feature tensor in the order of object_names.
    """
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

    tensor = torch.tensor(features, dtype=torch.float)
    return tensor


def create_single_sequence(i, timeframes, window_size, lookahead_steps, df_critical, object_names, edge_index):
    window_times = timeframes[i:i + window_size]
    label_times = timeframes[i + window_size:i + window_size + lookahead_steps]

    frame_tensors = []
    for t in window_times:
        df_t = df_critical[df_critical["Timeframe"] == t]
        frame_tensors.append(build_node_features(df_t, object_names))
    seq_tensor = torch.stack(frame_tensors)

    df_future = df_critical[df_critical["Timeframe"].isin(label_times)]
    label = int((df_future["Name"] == "PowerLine1").any())

    window_df = df_critical[df_critical["Timeframe"].isin(window_times)]
    source_file = window_df["source_file"].mode().iloc[0]

    return seq_tensor, label, source_file 


def nx_to_edge_index(G, object_names):
    name_to_index = {name: idx for idx, name in enumerate(object_names)}
    edge_list = []
    for u, v in G.edges():
        if u in name_to_index and v in name_to_index:
            edge_list.append([name_to_index[u], name_to_index[v]])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    print(f"edge_index: shape = {edge_index.shape}")
    return edge_index


def build_stgcn_dataset(df_critical, G, output_path="./processed_data", n_jobs=-1, **params):
    window_size = params.get("window_size") 
    lookahead_seconds = params.get("LOOKAHEAD_SECONDS")
    CRITICAL_DIST_THRESHOLD = params.get("dist_threshold")
    num_samples = params.get("num_samples")
    FIX_MIN_SEC = params.get("MIN_FIXATION_DURATION")
    

    print("Starting STGCN dataset construction (parallel)...")
    os.makedirs(output_path, exist_ok=True)

    object_names = list(G.nodes)
    edge_index = nx_to_edge_index(G, object_names)

    df_critical = df_critical[df_critical["Name"].isin(object_names)].copy()
    timeframes = sorted(df_critical["Timeframe"].unique())
    delta_t = np.diff(timeframes).mean()
    lookahead_steps = int(lookahead_seconds / delta_t)
    print(f"Using {len(object_names)} nodes, {len(timeframes)} timeframes, Δt = {delta_t:.4f}s")

    total_sequences = len(timeframes) - window_size - lookahead_steps

    results = Parallel(n_jobs=n_jobs)(
        delayed(create_single_sequence)(i, timeframes, window_size, lookahead_steps, df_critical, object_names, edge_index)
        for i in tqdm(range(total_sequences), desc="Building sequences")
    )

    sequences, labels, source_file_names = zip(*results)

    torch.save((sequences, labels, object_names, edge_index, source_file_names),
               os.path.join(output_path, f"stgcn_dataset_{CRITICAL_DIST_THRESHOLD}_{lookahead_seconds}_{FIX_MIN_SEC}_{int(num_samples)}_participants.pt"))

    print(f"Saved {len(sequences)} sequences to {output_path}/stgcn_dataset_{CRITICAL_DIST_THRESHOLD}_{lookahead_seconds}_{FIX_MIN_SEC}_{int(num_samples)}_participants.pt")
