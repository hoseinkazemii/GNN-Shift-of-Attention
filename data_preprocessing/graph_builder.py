import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics import pairwise_distances


def build_graph(**params):
    plot_graph = params.get("plot_graph") 
    top_k = params.get("top_k")
    dist_threshold = params.get("dist_threshold")
    lookahead_seconds = params.get("LOOKAHEAD_SECONDS")


    df = pd.read_csv("./processed_data/combined_labeled_data.csv")

    CRITICAL_DIST_THRESHOLD = dist_threshold
    df_critical = df[
        (df["Min Distance to Powerline (Grabbable Object)"] < CRITICAL_DIST_THRESHOLD) &
        (df["LoadingStarted"] == 1)
    ].copy()

    print(f"Original size: {len(df)}")
    print(f"Critical size (distance + loading): {len(df_critical)}")
    
    label_counts = df_critical[f"label_powerline_future_{lookahead_seconds}_seconds"].value_counts().sort_index()
    label_percentages = df_critical[f"label_powerline_future_{lookahead_seconds}_seconds"].value_counts(normalize=True).sort_index() * 100
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

    TOP_K = top_k
    G = nx.Graph()

    for obj in object_names:
        G.add_node(obj)

    for i, obj in enumerate(object_names):
        dists = dist_matrix[i]
        nearest_indices = np.argsort(dists)[1:TOP_K + 1]  # skip self
        for j in nearest_indices:
            neighbor = object_names[j]
            G.add_edge(obj, neighbor)

    if plot_graph:
        plot_graph(G)

    return G, object_names, df_critical
