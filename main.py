import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv


# Define the folder path where the CSV files are stored
folder_path = './PowerLine Scenario Data-20241031/Backup'  # Update this path to your CSV folder

def read_data(folder_path):
    # Initialize an empty list to store dataframes
    dataframes = []
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a CSV file and starts with the specific prefix
        if filename.startswith("VREyeTracking_PowerLineScenario") and filename.endswith(".csv"):
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            # Read the CSV file into a dataframe
            df = pd.read_csv(file_path)
            # Append the dataframe to the list
            dataframes.append(df)

    dataframes = pd.concat(dataframes, ignore_index=True)

    return dataframes

# Concatenate all dataframes into a single dataframe
eye_df = read_data(folder_path="./PowerLine Scenario Data-20241031")


def extract_object_positions(folder_path):
    # List of names to exclude
    moving_objects = ["ChainEyeTracker", "GrabbableObject", "crane_hook", 
                    "CabinPanelEyeTracker", "CraneJibEyeTracker", "LinkBelow",
                    'Moving_Black_Car', 'Moving_Black_Volkswagen_Car', 
                    'Moving_Blue_Car', 'Moving_CocaCola_Red_Truck', 
                    'Moving_Dark_Blue_Audi_Car', 
                    'Moving_Red_Car', 'Moving_White_Maserati_Car', 
                    'Moving_Yellow_BUS', 'Moving_Yellow_Car',
                    'Rope']  # Add or modify this list as needed

    known_positions_df = pd.DataFrame()

    for filename in os.listdir(folder_path):
        if filename.startswith("VREyeTracking_PowerLineScenario") and filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            known_positions_df = pd.concat([known_positions_df, df], ignore_index=True)

    required_columns = ["Name", "Object X", "Object Y", "Object Z"]
    if not all(col in known_positions_df.columns for col in required_columns):
        raise ValueError(f"The dataset must contain the following columns: {required_columns}")

    # Create a dictionary for unique "Name" values, mapping them to the first occurrence of (Object X, Object Y, Object Z),
    # excluding the specified names
    object_positions_dict = {}
    for name, group in known_positions_df.groupby("Name"):
        if name not in moving_objects:
            first_row = group.iloc[0]
            object_positions_dict[name] = (first_row["Object X"], first_row["Object Y"], first_row["Object Z"])

    return object_positions_dict


def create_graph_snapshots(folder_path, eye_df, load_df, downsampling_factor, powerline_allowed_distance, edge_attribute_option):
    temporal_snapshots = []  # Graph snapshots in downsampled timeframes
    labels = []  # Labels for node classification
    last_known_positions = extract_object_positions(folder_path)  # Dictionary that stores object positions

    # Get the start time for loading from LoadData_PowerLineScenario
    loading_start_time = load_df.loc[load_df['LoadingStarted'] == 1, 'Time'].iloc[0]

    # Find the corresponding start time in the eye-tracking dataset
    start_index = eye_df[eye_df['Timeframe'] >= loading_start_time].index[0]
    eye_df_filtered = eye_df[(eye_df['Timeframe'] >= loading_start_time) & 
                             (eye_df['Powerline and Grabbable X Distance'] <= powerline_allowed_distance)]

    # Extract all dominant objects across downsampled dataframes
    all_objects = set()
    for i in range(0, len(eye_df_filtered), downsampling_factor):
        downsampled_df = eye_df_filtered[i:i + downsampling_factor]
        dominant_name = downsampled_df['Name'].value_counts().idxmax()
        all_objects.add(dominant_name)
    all_objects = list(all_objects)

    # Pre-populate last known positions by going back before loading start time
    for obj_name in all_objects:
        obj_data = eye_df[(eye_df['Name'] == obj_name) & (eye_df.index < start_index)]
        if not obj_data.empty:
            # Use last known position
            last_instance = obj_data.iloc[-1]
            last_known_positions[obj_name] = (last_instance['Object X'], last_instance['Object Y'], last_instance['Object Z'])

    # Iterate through downsampled frames starting from the loading start index to create individual graph snapshots
    for i in range(0, len(eye_df_filtered), downsampling_factor):
        current_time = eye_df_filtered.iloc[i]['Timeframe']
        downsampled_df = eye_df_filtered[i:i + downsampling_factor]

        # Create a new graph for this snapshot (initialize a new empty graph each time)
        G = nx.DiGraph()

        # Add temporal nodes for all objects at the current timeframe
        for obj_name in all_objects:
            temporal_node = (obj_name, current_time)

            # Calculate average spatial coordinates if the object is present in the downsampled frame
            if obj_name in downsampled_df['Name'].values:
                x = downsampled_df[downsampled_df['Name'] == obj_name]['Object X'].mean()
                y = downsampled_df[downsampled_df['Name'] == obj_name]['Object Y'].mean()
                z = downsampled_df[downsampled_df['Name'] == obj_name]['Object Z'].mean()
                last_known_positions[obj_name] = (x, y, z)
            else:
                x, y, z = last_known_positions.get(obj_name, (0, 0, 0))

            # Add temporal node with spatial coordinates and name
            G.add_node(temporal_node, x=x, y=y, z=z, name=obj_name)

        # Add spatial edges between nodes in the current timeframe
        nodes_at_time = list(G.nodes)

        threshold = 20  # Define spatial proximity threshold
        for i, node_i in enumerate(nodes_at_time):
            for j in range(i + 1, len(nodes_at_time)):
                node_j = nodes_at_time[j]
                if G.nodes[node_i]['x'] is not None and G.nodes[node_j]['x'] is not None:
                    distance = euclidean(
                        (G.nodes[node_i]['x'], G.nodes[node_i]['y'], G.nodes[node_i]['z']),
                        (G.nodes[node_j]['x'], G.nodes[node_j]['y'], G.nodes[node_j]['z'])
                    )
                    if distance < threshold:
                        # Determine edge weight based on the chosen option
                        if edge_attribute_option == 1:
                            weight = 1 / distance if distance != 0 else 0  # Inverse distance
                        elif edge_attribute_option == 2:
                            sigma = 10  # You can tune sigma for the Gaussian kernel
                            weight = np.exp(-distance ** 2 / (2 * sigma ** 2))
                        else:
                            raise ValueError("Invalid edge_attribute_option. Choose 1 (inverse distance) or 2 (Gaussian kernel).")

                        # Add an edge in both directions with the computed weight
                        G.add_edge(node_i, node_j, weight=weight)
                        G.add_edge(node_j, node_i, weight=weight)

        # Generate label for the dominant node in the current graph snapshot
        dominant_name = downsampled_df['Name'].value_counts().idxmax()
        graph_nodes = list(G.nodes)
        dominant_node_index = next((idx for idx, node in enumerate(graph_nodes) if node[0] == dominant_name), -1)
        labels.append(dominant_node_index)

        # Convert the graph to PyTorch Geometric Data and store it
        pyg_data = convert_to_pyg_data(G)
        temporal_snapshots.append(pyg_data)

    return temporal_snapshots, labels, eye_df, eye_df_filtered


# Convert NetworkX graph to PyTorch Geometric data format
def convert_to_pyg_data(G):
    node_features = []
    node_indices = list(G.nodes)
    node_mapping = {node: idx for idx, node in enumerate(node_indices)}

    edge_index = []
    edge_attr = []

    for node in node_indices:
        node_features.append([G.nodes[node]['x'], G.nodes[node]['y'], G.nodes[node]['z']])

    for u, v, d in G.edges(data=True):
        edge_index.append([node_mapping[u], node_mapping[v]])
        # Only include weight as edge attribute
        edge_attr.append(d['weight'])

    # Convert to PyTorch tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    node_features = torch.tensor(node_features, dtype=torch.float)

    # Create PyTorch Geometric Data object
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

    return data

# Load dataset and create temporal snapshots
load_df = pd.read_csv('./PowerLine Scenario Data-20241031/LoadData_PowerLineScenario_20241031184156.csv')  # Example LoadData file

temporal_snapshots, labels, eye_df, eye_df_filtered = \
    create_graph_snapshots(folder_path, eye_df, load_df, \
                           downsampling_factor=15, powerline_allowed_distance=40, \
                           edge_attribute_option=2)


from torch.utils.data import Dataset, DataLoader
import torch

# Create a custom dataset for node classification
class NodeClassificationDataset(Dataset):
    def __init__(self, snapshots, labels):
        """
        Args:
            snapshots (list): List of graph snapshots (PyTorch Geometric Data objects).
            labels (list): List of labels for the fixated node in each graph snapshot. 
                           Each label corresponds to the index of the fixated node.
        """
        super(NodeClassificationDataset, self).__init__()
        self.snapshots = snapshots
        self.labels = labels

        # Ensure snapshots and labels have the same length
        if len(self.snapshots) != len(self.labels):
            raise ValueError(f"Number of snapshots ({len(self.snapshots)}) does not match number of labels ({len(self.labels)})")

    def __len__(self):
        return len(self.snapshots)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the snapshot to retrieve.
        
        Returns:
            data (Data): Graph snapshot with a node classification target (data.y).
        """
        data = self.snapshots[idx]

        # Assign the node-level label (fixated node) for this snapshot
        fixated_node_index = self.labels[idx]  # Label corresponds to the fixated node's index
        node_labels = torch.zeros(data.num_nodes, dtype=torch.long)  # Initialize all node labels to 0
        node_labels[fixated_node_index] = 1  # Mark the fixated node as the target (binary classification)
        data.y = node_labels  # Assign labels for all nodes

        return data



# Create the dataset and DataLoader
dataset = NodeClassificationDataset(temporal_snapshots, labels)
print(dataset[0].y)
print(dataset[1].y)
print(dataset[2].y)
print(dataset[3].y)
print(dataset[4].y)
print(dataset[5].y)
print(dataset[6].y)
print(dataset[21].y)

raise ValueError
batch_size = 32
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
print("Dataset Overview:")
print(f"Number of snapshots (graphs): {len(dataset)}")
print(f"Example graph structure from the dataset:")

# Print details of the first snapshot
example_data = dataset[0]
print(f"Graph 0:")
print(f" - Number of nodes: {example_data.num_nodes}")
print(f" - Number of edges: {example_data.edge_index.shape[1]}")
print(f" - Node features shape: {example_data.x.shape}")
print(f" - Edge attributes shape: {example_data.edge_attr.shape}")
print(f" - Labels (y): {example_data.y.tolist()}")

# Inspect the labels and snapshots
print("\nInspecting labels and snapshots:")
for i in range(min(4, len(dataset))):  # Print details for the first 5 graphs
    snapshot = dataset[i]
    print(f"Snapshot {i}:")
    print(f" - Fixated node index: {labels[i]}")
    print(f" - Node features (x): {snapshot.x[:5]}")  # Print first 5 nodes' features
    print(f" - Edge attributes (first 5): {snapshot.edge_attr[:5] if len(snapshot.edge_attr) > 0 else 'No edge attributes'}")
    print(f" - Label tensor (y): {snapshot.y.tolist()}")



raise ValueError

# # Visualize a sample batch (for debugging purposes)
# for batch in loader:
#     print(batch)
#     break


# Step 1: Feature Scaling for All Graph Snapshots
scaler = StandardScaler()

# Convert all node features into a single numpy array for fitting the scaler
node_features_list = [pyg_data.x.numpy() for pyg_data in temporal_snapshots]
# Stack all features to fit the scaler
all_node_features_np = np.vstack(node_features_list)
scaler.fit(all_node_features_np)

# Apply the scaling and update node features in the temporal snapshots
for pyg_data in temporal_snapshots:
    node_features_np = pyg_data.x.numpy()
    node_features_scaled = scaler.transform(node_features_np)  # Scale the features
    pyg_data.x = torch.tensor(node_features_scaled, dtype=torch.float)  # Convert back to tensor


# Sliding Window Parameters
train_window_size = 6  # 8 frames for training
val_window_size = 2    # 1 frame for validation
test_window_size = 2   # 1 frame for testing
window_step_size = 3   # Step size for the sliding window

# Create train, validation, and test sets
train_snapshots, val_snapshots, test_snapshots = [], [], []
train_labels, val_labels, test_labels = [], [], []

total_window_size = train_window_size + val_window_size + test_window_size

# Extract labels as numpy array for easier indexing
labels_np = np.array(labels)

# Iterate through temporal snapshots using the sliding window
for start_idx in range(0, len(temporal_snapshots) - total_window_size + 1, window_step_size):
    # Create train, validation, and test datasets
    train_snapshots.extend(temporal_snapshots[start_idx : start_idx + train_window_size])
    val_snapshots.extend(temporal_snapshots[start_idx + train_window_size : start_idx + train_window_size + val_window_size])
    test_snapshots.extend(temporal_snapshots[start_idx + train_window_size + val_window_size : start_idx + total_window_size])

    # Extract labels for training, validation, and testing
    train_labels.extend(labels_np[start_idx : start_idx + train_window_size])
    val_labels.extend(labels_np[start_idx + train_window_size : start_idx + train_window_size + val_window_size])
    test_labels.extend(labels_np[start_idx + train_window_size + val_window_size : start_idx + total_window_size])

# Check if the lengths of the snapshots and labels are equal, and add values if they don't match
if len(train_snapshots) != len(train_labels):
    diff = len(train_snapshots) - len(train_labels)
    if diff > 0:
        train_labels.extend([0] * diff)  # Add placeholder labels if train_labels are fewer
    else:
        train_snapshots.extend([train_snapshots[-1]] * abs(diff))  # Duplicate the last snapshot if snapshots are fewer

if len(val_snapshots) != len(val_labels):
    diff = len(val_snapshots) - len(val_labels)
    if diff > 0:
        val_labels.extend([0] * diff)  # Add placeholder labels if val_labels are fewer
    else:
        val_snapshots.extend([val_snapshots[-1]] * abs(diff))  # Duplicate the last snapshot if snapshots are fewer

if len(test_snapshots) != len(test_labels):
    diff = len(test_snapshots) - len(test_labels)
    if diff > 0:
        test_labels.extend([0] * diff)  # Add placeholder labels if test_labels are fewer
    else:
        test_snapshots.extend([test_snapshots[-1]] * abs(diff))  # Duplicate the last snapshot if snapshots are fewer

# Print lengths to verify that everything matches
print(f"Train Snapshots: {len(train_snapshots)}, Train Labels: {len(train_labels)}")
print(f"Val Snapshots: {len(val_snapshots)}, Val Labels: {len(val_labels)}")
print(f"Test Snapshots: {len(test_snapshots)}, Test Labels: {len(test_labels)}")


# Create NodeClassificationDataset instances for train, validation, and test
train_dataset = NodeClassificationDataset(train_snapshots, train_labels)
val_dataset = NodeClassificationDataset(val_snapshots, val_labels)
test_dataset = NodeClassificationDataset(test_snapshots, test_labels)

# Create DataLoaders for each dataset
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Visualize a sample batch (for debugging purposes)
for batch in train_loader:
    print("Train Batch:", batch)
    break

for batch in val_loader:
    print("Validation Batch:", batch)
    break

for batch in test_loader:
    print("Test Batch:", batch)
    break



class NodeClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
            input_dim (int): Number of input features per node.
            hidden_dim (int): Number of hidden units in the graph convolutional layers.
            output_dim (int): Number of output classes for node classification.
        """
        super(NodeClassifier, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = Linear(hidden_dim, output_dim)  # Fully connected layer for node classification

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.fc(x)  # Final layer for classification
        return x  # Output shape: [num_nodes, output_dim]



# Determine the number of unique labels
num_classes = len(set(labels))  # This should give you the correct number of unique classes, which is 8 in your case
print(f"Number of classes: {num_classes}")  # This should print 8


# print(set(labels))
# raise ValueError
# Update the model initialization with the correct number of output channels
in_channels = temporal_snapshots[0].num_node_features
hidden_channels = 16
out_channels = 1  # Set output channels to 1

model = NodeClassifier(in_channels, hidden_channels, out_channels)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_function = torch.nn.CrossEntropyLoss()


# Training Loop
model.train()
for epoch in range(50):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch)  # Forward pass

        # Ensure that the output dimensions match the number of nodes in batch and number of classes
        try:
            loss = loss_function(out, batch.y)  # Compute loss
        except IndexError as e:
            print(out)
            print(out.shape)
            print("******************")
            print(batch.y)
            print(batch.y.shape)
            print("**************************")
            print(f"Label issue found: {batch.y}")
            raise e
        
        loss.backward()  # Backpropagation
        optimizer.step()  # Update parameters
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")





# Ensure the model is in evaluation mode
model.eval()

# Initialize lists to accumulate true labels and predicted labels
all_true_labels = []
all_pred_labels = []

with torch.no_grad():  # No need to calculate gradients during evaluation
    for batch in test_loader:
        # Forward pass through the model
        out = model(batch)

        # Predictions: Take the class with the highest log-probability
        pred = out.argmax(dim=1)

        # Convert ground truth and predictions to binary labels (0 = powerline, 1 = other)
        binary_true = (batch.y != 0).long()  # 0 if powerline, 1 otherwise
        binary_pred = (pred != 0).long()  # 0 if powerline, 1 otherwise

        # Iterate through the batch in chunks of 5 shifts (next 5 shifts)
        for i in range(0, len(binary_true), 1):
            true_chunk = binary_true[i:i + 1]
            pred_chunk = binary_pred[i:i + 1]

            # Determine if powerline (`P`) appears in true labels and predicted labels
            true_has_powerline = (true_chunk == 0).any().item()  # True if powerline is in true labels
            pred_has_powerline = (pred_chunk == 0).any().item()  # True if powerline is in predictions

            # Append the binary values (0 for powerline present, 1 for no powerline)
            all_true_labels.append(0 if true_has_powerline else 1)
            all_pred_labels.append(0 if pred_has_powerline else 1)

# Generate the classification report
print(classification_report(all_true_labels, all_pred_labels, target_names=["Powerline", "Other"]))












'''''''''
# Model Training:
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, EdgeConv
from torch_geometric.nn import Sequential, Linear
from torch_geometric.data import DataLoader
from torch.optim import Adam

# Defining the STGCN Model for Attention Shift Prediction
class STGCNAttentionShift(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(STGCNAttentionShift, self).__init__()
        
        # Graph convolutional layers to capture spatial relationships
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Linear layers for edge prediction
        self.edge_predictor = Sequential(
            'x_i, x_j, edge_attr',
            [
                (Linear(2 * hidden_channels + 1, hidden_channels), 'relu'),
                Linear(hidden_channels, out_channels)
            ]
        )
        
    def forward(self, x, edge_index, edge_attr):
        # Apply spatial convolutions
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        
        # Edge feature prediction
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col], edge_attr], dim=1)
        
        return self.edge_predictor(edge_features)

# Instantiate the model
in_channels = pyg_data.num_node_features  # Number of input features per node
hidden_channels = 64  # Number of hidden channels in the model
out_channels = 1  # Output is a binary classification (attention shift to powerline or not)

model = STGCNAttentionShift(in_channels, hidden_channels, out_channels)

# Optimizer and loss function
optimizer = Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()  # Binary Cross-Entropy with logits

# Training Loop
model.train()
epochs = 50
for epoch in range(epochs):
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        
        # Define ground truth for edges: Predict if attention shifts to "PowerLine1"
        y = torch.zeros(out.size(0), dtype=torch.float)
        for i, (u, v) in enumerate(batch.edge_index.t().tolist()):
            if batch.x[u].tolist()[0] == 'PowerLine1' or batch.x[v].tolist()[0] == 'PowerLine1':
                y[i] = 1.0
        
        # Compute loss
        loss = criterion(out.squeeze(), y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Evaluation (Example)
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in loader:
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        prediction = (torch.sigmoid(out) > 0.5).float()
        y = torch.zeros(out.size(0), dtype=torch.float)
        for i, (u, v) in enumerate(batch.edge_index.t().tolist()):
            if batch.x[u].tolist()[0] == 'PowerLine1' or batch.x[v].tolist()[0] == 'PowerLine1':
                y[i] = 1.0
        correct += (prediction.squeeze() == y).sum().item()
        total += y.size(0)

accuracy = correct / total
print(f"Evaluation Accuracy: {accuracy:.4f}")
'''''''''



'''''''''


# Helper function to calculate fixation duration
def calculate_fixation_duration(data):
    data['Fixation Duration'] = 0.0
    data['Fixation Group'] = (data['Name'] != data['Name'].shift()).cumsum()

    for group in data['Fixation Group'].unique():
        group_indices = data[data['Fixation Group'] == group].index
        if len(group_indices) > 1:
            fixation_duration = data.loc[group_indices, 'Timeframe'].diff().sum()
            data.loc[group_indices, 'Fixation Duration'] = fixation_duration

    data = data.drop(columns=['Fixation Group'])
    return data

# Helper function to calculate fixation history
def calculate_fixation_history(data):
    data['Fixation History'] = 0.0
    data['Fixation Group'] = (data['Name'] != data['Name'].shift()).cumsum()

    fixation_history_dict = {}

    for group in data['Fixation Group'].unique():
        group_indices = data[data['Fixation Group'] == group].index
        object_name = data.loc[group_indices[0], 'Name']

        if object_name not in fixation_history_dict:
            fixation_history_dict[object_name] = 0

        if len(group_indices) > 1:
            start_time = data.loc[group_indices[0], 'Timeframe']
            end_time = data.loc[group_indices[-1], 'Timeframe']
            current_fixation_duration = end_time - start_time

            data.loc[group_indices, 'Fixation History'] = fixation_history_dict[object_name]

            fixation_history_dict[object_name] += current_fixation_duration

        else:
            data.loc[group_indices, 'Fixation History'] = fixation_history_dict[object_name]

    data = data.drop(columns=['Fixation Group'])
    return data

def calculate_attention_switch_frequency(data, window_size=10):
    # Calculate attention switches (exclude the first row from being considered a switch)
    data['Attention Switch'] = (data['Name'] != data['Name'].shift()).astype(int)
    
    # Ensure the first row is not counted as a switch
    data.loc[data.index[0], 'Attention Switch'] = 0
    
    # # Calculate rolling sum for attention switch frequency
    # data['Attention Switch Frequency'] = data['Attention Switch'].rolling(window=window_size).sum()
    
    # # Fill any NaN values (from rolling window) with 0
    # data['Attention Switch Frequency'] = data['Attention Switch Frequency'].fillna(0)
    
    return data

# Calculate node attributes
data = calculate_fixation_duration(data)
data = calculate_fixation_history(data)
# data = calculate_attention_switch_frequency_for_category(data)

# Fill NaN values with 0
data = data.fillna(0)

# Define the Graph Data with Time-Connected Edges
def create_graph_data_with_time_edges(data):
    POWERLINE_NAME = "GrabbableObject"
    unique_objects = data['Category'].unique()

    graphs = []  # List to store all the graphs

    # Mapping from object names to node indices
    node_mapping = {obj: idx for idx, obj in enumerate(unique_objects)}

    # Initialize current state of each object (fixation duration and history)
    current_state = {obj: {'Fixation Duration': 0.0, 'Fixation History': 0.0} for obj in unique_objects}

    # Global edge index that will store all the edges added over time
    global_edge_index = []
    
    # previous_object = None
    previous_object = data['Category'][0] # Track the previous object in the attention shift

    # Maintain a sliding window of edges
    edge_window_size = 5  # Number of recent edges to keep

    for _, row in data.iterrows():
        # If there was a previous attention shift, connect the previous object and current object
        if row['Attention Switch'] == 1:
            obj = row['Category']  # Current object (node) being fixated on
            node_idx = node_mapping[obj]

            # Update the current state of the fixated object
            current_state[obj]['Fixation Duration'] = row['Fixation Duration']
            current_state[obj]['Fixation History'] = row['Fixation History']

            # Create node features for all objects (global graph always includes all objects)
            node_features = []
            for obj_name in unique_objects:
                node_features.append([
                    current_state[obj_name]['Fixation Duration'],
                    current_state[obj_name]['Fixation History']
                ])

            previous_node_idx = node_mapping[previous_object]

            global_edge_index.append([previous_node_idx, node_idx])  # Add the new edge for this attention shift

            # Keep only the most recent `edge_window_size` edges
            if len(global_edge_index) > edge_window_size:
                global_edge_index = global_edge_index[-edge_window_size:]

            # Create the target label for the graph (whether the attention shift is on the powerline)
            target = 1 if row['Category'] == POWERLINE_NAME else 0

            # Convert node features to tensor
            node_features = torch.tensor(node_features, dtype=torch.float)

            # Convert edge index to tensor (transpose to match PyTorch Geometric format)
            edge_index = torch.tensor(global_edge_index, dtype=torch.long).t().contiguous()
            # Convert target label to tensor
            target = torch.tensor([target], dtype=torch.float)

            # Create the graph and add it to the list
            graph = Data(x=node_features, edge_index=edge_index, y=target)
            graphs.append(graph)
        
        # Update previous object to the current one
        previous_object = obj

    return graphs


# Generate the graphs list
graphs = create_graph_data_with_time_edges(data)

POWERLINE_NAME = "GrabbableObject"
unique_objects = data['Category'].unique()
# Mapping from object names to node indices
node_mapping = {obj: idx for idx, obj in enumerate(unique_objects)}



import matplotlib.pyplot as plt
import networkx as nx

# def visualize_graph_with_names(graph, node_mapping):
#     # Create a reverse mapping from index to name
#     reverse_node_mapping = {v: k for k, v in node_mapping.items()}

#     # Create a NetworkX directed graph from the PyG Data object
#     G = nx.DiGraph()

#     # Add nodes with names as labels
#     for node_idx in range(graph.x.size(0)):
#         node_name = reverse_node_mapping[node_idx]
#         G.add_node(node_idx, label=node_name)

#     # Add edges
#     for i in range(graph.edge_index.shape[1]):
#         source = graph.edge_index[0, i].item()
#         target = graph.edge_index[1, i].item()
#         G.add_edge(source, target)

#     pos = nx.spring_layout(G)
#     labels = nx.get_node_attributes(G, 'label')
#     nx.draw(G, pos, with_labels=True, labels=labels, node_size=400, font_size=5, edge_color='red', width=3, arrowstyle='->', arrowsize=15)  # Smaller nodes and thicker edges

#     plt.show()

# # Visualize a few graphs with names
# for i, graph in enumerate(graphs[20:100]):
#     print(f"Graph {i+1}")
#     visualize_graph_with_names(graph, node_mapping)

# raise ValueError

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader

# Define the STGCN model
class STGCN(torch.nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, num_classes, seq_length):
        super(STGCN, self).__init__()
        self.seq_length = seq_length

        # Temporal Layer (GRU) with sequence length
        self.gru = torch.nn.GRU(node_feature_dim, hidden_dim, batch_first=True)

        # Graph Convolutional Layers (GCNs)
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        # Fully Connected Layer for classification
        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, data_seq):
        all_x = []
        batch = None

        # Iterate through each graph in the sequence
        for data in data_seq:
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

            # Spatial Graph Convolution
            x = self.gcn1(x, edge_index).relu()
            x = self.gcn2(x, edge_index).relu()

            # Store graph-level embeddings for the sequence
            x = global_mean_pool(x, batch)  # Global mean pooling for graph-level representation
            all_x.append(x)

        # Stack the embeddings for the entire sequence
        all_x = torch.stack(all_x, dim=1)  # (batch_size, seq_length, feature_dim)

        # Pass through the GRU
        gru_out, _ = self.gru(all_x)  # (batch_size, seq_length, hidden_dim)

        # Use the last hidden state from the GRU for classification
        final_out = gru_out[:, -1, :]  # (batch_size, hidden_dim)

        # Classification layer
        out = self.fc(final_out)
        return F.log_softmax(out, dim=1)



# Example model initialization
node_feature_dim = 2  # Adjust based on the number of node features
edge_feature_dim = 5  # Adjust based on the number of edge features
hidden_dim = 64  # Number of hidden units in GCN and GRU layers
num_classes = 2  # Binary classification (0 or 1)

model = STGCN(node_feature_dim, edge_feature_dim, hidden_dim, num_classes)

from torch.utils.data import DataLoader

# Custom Dataset to group graphs into sequences
class GraphSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, seq_length):
        self.graphs = graphs
        self.seq_length = seq_length

    def __len__(self):
        return len(self.graphs) - self.seq_length + 1

    def __getitem__(self, idx):
        return self.graphs[idx:idx + self.seq_length]


# Assuming 'graphs' is a list of PyG Data objects created using create_graph_data()
train_data = graphs[:int(0.8 * len(graphs))]  # 80% for training
test_data = graphs[int(0.8 * len(graphs)):]  # 20% for testing

# Create DataLoader for sequences of graphs
train_dataset = GraphSequenceDataset(train_data, seq_length=32)
test_dataset = GraphSequenceDataset(test_data, seq_length=32)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)





import torch.optim as optim
from torch_geometric.loader import DataLoader



# Create DataLoader for batching
train_loader = DataLoader(train_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
def train(model, loader):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.view(-1).long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import torch

def test(model, loader):
    model.eval()
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient computation for evaluation
        for data in loader:
            out = model(data)
            pred = out.argmax(dim=1)
            correct += (pred == data.y.view(-1)).sum().item()
            
            all_preds.extend(pred.cpu().numpy())  # Collect predictions
            all_labels.extend(data.y.view(-1).cpu().numpy())  # Collect true labels

    accuracy = correct / len(loader.dataset)

    # Calculate precision, recall, F1 score
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    # Generate classification report
    class_report = classification_report(all_labels, all_preds)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Classification Report:\n{class_report}')
    print(f'Confusion Matrix:\n{conf_matrix}')
    
    return accuracy

# Train the model for a number of epochs
num_epochs = 20
for epoch in range(num_epochs):
    train_loss = train(model, train_loader)
    print(f'Epoch {epoch+1}, Loss: {train_loss:.4f}')
    print('Test Performance:')
    test_acc = test(model, test_loader)

# # Helper function to calculate speed of gaze movement
# def calculate_speed_of_gaze_movement(data):
#     data['Shift Time'] = data['Timeframe'].diff().fillna(0)
#     data['Speed of Gaze Movement'] = data['Shift Distance'] / data['Shift Time']
#     return data

# # Helper function to calculate gaze path direction
# def calculate_gaze_path_direction(data):
#     # Initialize the gaze path direction columns with NaN
#     data['Gaze Path Direction X'] = np.nan
#     data['Gaze Path Direction Y'] = np.nan
#     data['Gaze Path Direction Z'] = np.nan

#     # Calculate gaze path direction only when there is a shift of attention
#     attention_shifts = data['Name'] != data['Name'].shift()
    
#     data.loc[attention_shifts, 'Gaze Path Direction X'] = data['Gaze Direction X'].diff()
#     data.loc[attention_shifts, 'Gaze Path Direction Y'] = data['Gaze Direction Y'].diff()
#     data.loc[attention_shifts, 'Gaze Path Direction Z'] = data['Gaze Direction Z'].diff()
   
#     # Replace NaN values with 0 for the gaze path direction columns
#     data['Gaze Path Direction X'] = data['Gaze Path Direction X'].fillna(0)
#     data['Gaze Path Direction Y'] = data['Gaze Path Direction Y'].fillna(0)
#     data['Gaze Path Direction Z'] = data['Gaze Path Direction Z'].fillna(0)
    
#     return data

'''''''''
