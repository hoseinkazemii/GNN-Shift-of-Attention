import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import os
import time
import torch
from torch_geometric.data import Data
from joblib import Parallel, delayed
from tqdm import tqdm

df = pd.read_csv("combined_labeled_data.csv")

WINDOW_SIZE = 10
LOOKAHEAD_SECONDS = 1.5
CRITICAL_DIST_THRESHOLD = 50
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

fixation_duration = df_critical.groupby("Name")["Timeframe"].count() * 0.02  # 0.02s sampling interval
fixated_objects = fixation_duration[fixation_duration >= 0.1].index.tolist()

object_positions = df_critical[df_critical["Name"].isin(fixated_objects)].groupby("Name")[["Object X", "Object Y", "Object Z"]].mean()

# Build distance matrix
position_matrix = object_positions.values
object_names = object_positions.index.tolist()
dist_matrix = pairwise_distances(position_matrix)

dist_df = pd.DataFrame(dist_matrix, index=object_names, columns=object_names)

TOP_K = 3 # Number of nearest neighbors to consider for each node
G = nx.Graph()

for obj in object_names:
    G.add_node(obj)

for i, obj in enumerate(object_names):
    dists = dist_matrix[i]
    nearest_indices = np.argsort(dists)[1:TOP_K + 1]  # skip self
    for j in nearest_indices:
        neighbor = object_names[j]
        G.add_edge(obj, neighbor)

# # Visualize the graph
# plt.figure(figsize=(18, 12))
# pos = nx.spring_layout(G, seed=42, k=0.5)  # `k` controls spacing between nodes
# nx.draw_networkx_nodes(G, pos, node_color="lightgreen", node_size=700, alpha=0.9)
# nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.6)
# nx.draw_networkx_labels(
#     G,
#     pos,
#     font_size=9,
#     font_family="sans-serif",
#     bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2", alpha=0.8)
# )

# plt.title("Static Graph — Fixated Objects (≥0.1s), Loading Started, Distance < 50", fontsize=14)
# plt.axis("off")
# plt.tight_layout()
# plt.savefig("./plots/static_graph.png", dpi=600)
# plt.show()


def build_node_features(df_t, object_names):
    """
    Given a dataframe for a single timestep, returns node feature tensor in the order of object_names.
    """
    start_time = time.time()
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

    tensor = torch.tensor(features, dtype=torch.float)
    return tensor


def create_single_sequence(i, timeframes, window_size, lookahead_steps, df_critical, object_names, edge_index):
    window_times = timeframes[i:i + window_size]
    label_times = timeframes[i + window_size:i + window_size + lookahead_steps]

    graphs = []
    for t in window_times:
        df_t = df_critical[df_critical["Timeframe"] == t]
        x = build_node_features(df_t, object_names)
        data = Data(x=x, edge_index=edge_index)
        graphs.append(data)

    df_future = df_critical[df_critical["Timeframe"].isin(label_times)]
    label = int((df_future["Name"] == "PowerLine1").any())

    window_df = df_critical[df_critical["Timeframe"].isin(window_times)]
    source_file = window_df["source_file"].mode().iloc[0]

    return graphs, label, source_file

def create_graph_data(features, edge_index):
    return Data(x=features, edge_index=edge_index)


def nx_to_edge_index(G, object_names):
    name_to_index = {name: idx for idx, name in enumerate(object_names)}
    edge_list = []
    for u, v in G.edges():
        if u in name_to_index and v in name_to_index:
            edge_list.append([name_to_index[u], name_to_index[v]])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    print(f"edge_index: shape = {edge_index.shape}")
    return edge_index


def build_stgcn_dataset(df_critical, G, output_path, window_size, lookahead_seconds, n_jobs=-1):
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
               os.path.join(output_path, "stgcn_dataset.pt"))

    print(f"Saved {len(sequences)} sequences to {output_path}/stgcn_dataset.pt")


source_file_series = df_critical["source_file"].values

build_stgcn_dataset(
    df_critical=df_critical,
    G=G,
    output_path="./stgcn_data_no_fixated",
    window_size=WINDOW_SIZE,
    lookahead_seconds=LOOKAHEAD_SECONDS
)
