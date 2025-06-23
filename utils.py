# #### spliting the dataset into train and test based on 20% of the data for testing and 80% for training

# import torch
# from torch.utils.data import Dataset
# from torch_geometric.data import Data

# class STGCNDataset(Dataset):
#     def __init__(self, data_path):
#         super().__init__()
#         self.sequences, self.labels, self.object_names, self.edge_index = torch.load(data_path)

#         # Determine number of nodes & features
#         self.num_nodes = self.sequences[0][0].num_nodes
#         self.num_node_features = self.sequences[0][0].num_node_features

#     def __len__(self):
#         return len(self.sequences)

#     def __getitem__(self, idx):
#         graph_seq = self.sequences[idx]  # list of Data objects
#         label = self.labels[idx]

#         # Stack node features over time: [T, N, F]
#         x = torch.stack([g.x for g in graph_seq])  # [T, N, F]

#         return x, torch.tensor(label, dtype=torch.float)


import torch
import matplotlib.pyplot as plt
import networkx as nx

# Load the dataset
sequences, labels, object_names, edge_index, source_files = torch.load("stgcn_data_no_fixated/stgcn_dataset.pt")
print(f"Loaded {len(sequences)} sequences, {len(labels)} labels, {len(object_names)} object names")
print(f"Edge index shape: {edge_index.shape}")
print(f"Object names: {object_names}")
print(f"Sample label: {labels[0]}")
print(f"Sample sequence length: {len(sequences[0])}")
print(f"Sample sequence shape: {sequences[0][0].x.shape}")
print(f"Sample sequence: {sequences[0]}")
raise ValueError
# Pick one sequence (e.g., sample 0)
sample_idx = 0
graph_sequence = sequences[sample_idx]

# Helper: convert edge_index to a NetworkX graph
def edge_index_to_nx(edge_index, object_names):
    G = nx.Graph()
    for i in range(edge_index.size(1)):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        G.add_edge(object_names[src], object_names[dst])
    return G

# Build the static spatial graph
G_static = edge_index_to_nx(edge_index, object_names)

# Visualize a few consecutive graphs
num_graphs_to_plot = 5
plt.figure(figsize=(15, 3 * num_graphs_to_plot))

for idx in range(num_graphs_to_plot):
    plt.subplot(num_graphs_to_plot, 1, idx + 1)
    G_t = G_static.copy()
    x_t = graph_sequence[idx].x

    # Node color based on total (dx + dy + dz) magnitude for that node
    if x_t.shape[1] >= 3:
        feature_color = x_t[:, :3].abs().sum(dim=1).numpy()
    else:
        feature_color = x_t[:, 0].numpy()

    pos = nx.spring_layout(G_t, seed=42)  # consistent layout
    nodes = nx.draw_networkx_nodes(G_t, pos, node_color=feature_color, cmap='viridis', node_size=400)
    nx.draw_networkx_edges(G_t, pos, alpha=0.5)
    nx.draw_networkx_labels(G_t, pos, font_size=8)
    plt.title(f"Graph at timestep {idx}")
    plt.colorbar(nodes)
    plt.axis('off')

plt.tight_layout()
plt.show()










