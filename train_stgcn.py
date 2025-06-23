import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import math, random
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import torch.optim as optim
import matplotlib.pyplot as plt
import collections
#######################################################################################
# # Set up logging
import logging
import os
from datetime import datetime

os.makedirs("./logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"./logs/stgcn_training_log_{timestamp}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='w'  # Overwrite each run
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)
#########################################################################################


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(STGCNBlock, self).__init__()
        self.gcn = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return F.relu(self.gcn(x, edge_index))


class STGCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, lstm_hidden, num_layers, num_nodes):
        super(STGCNModel, self).__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(0.3)

        self.rnn = nn.GRU(input_size=hidden_channels * num_nodes,
                          hidden_size=lstm_hidden,
                          num_layers=num_layers,
                          batch_first=True)

        self.classifier = nn.Linear(lstm_hidden, 1)

    def forward(self, sequences, edge_index):
        batch_size, seq_len, num_nodes, num_feats = sequences.shape
        gcn_outputs = []

        for t in range(seq_len):
            x_t = sequences[:, t]  # [B, N, F]
            out = []
            for i in range(batch_size):
                x_i = self.gcn1(x_t[i], edge_index)
                x_i = F.relu(x_i)
                x_i = self.dropout(x_i)
                x_i = self.gcn2(x_i, edge_index)
                x_i = F.relu(x_i)
                x_i = self.dropout(x_i)
                out.append(x_i.flatten())
            gcn_outputs.append(torch.stack(out))

        gcn_outputs = torch.stack(gcn_outputs, dim=1)  # [B, T, N*H]
        rnn_out, _ = self.rnn(gcn_outputs)
        logits = self.classifier(rnn_out[:, -1])
        return logits.squeeze(1)


class STGCNDatasetSplitByParticipant:
    def __init__(self, data_path, test_fraction=0.2, seed=42):
        self.sequences, self.labels, self.object_names, self.edge_index, self.source_files = torch.load(data_path, weights_only=False)

        participant_to_indices = defaultdict(list)
        for i, src in enumerate(self.source_files):
            participant_to_indices[src].append(i)

        all_participants = sorted(participant_to_indices.keys())
        random.seed(seed)
        random.shuffle(all_participants)

        num_test = math.ceil(len(all_participants) * test_fraction)
        self.test_participants = all_participants[:num_test]
        self.train_participants = all_participants[num_test:]

        logging.info(f"Splitting {len(all_participants)} participants → {len(self.train_participants)} train, {len(self.test_participants)} test")

        self.train_indices = [i for p in self.train_participants for i in participant_to_indices[p]]
        self.test_indices = [i for p in self.test_participants for i in participant_to_indices[p]]

    def get_datasets(self):
        train_dataset = ParticipantIndexedDataset(self.sequences, self.labels, self.train_indices)
        test_dataset = ParticipantIndexedDataset(self.sequences, self.labels, self.test_indices)
        return train_dataset, test_dataset, self.edge_index


class ParticipantIndexedDataset(Dataset):
    def __init__(self, sequences, labels, indices):
        self.sequences = [sequences[i] for i in indices]
        self.labels = [labels[i] for i in indices]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        graph_seq = self.sequences[idx]
        label = self.labels[idx]
        x = torch.stack([g.x for g in graph_seq])  # [T, N, F]
        return x, torch.tensor(label, dtype=torch.float)


def train_one_epoch(model, dataloader, optimizer, criterion, device, edge_index):
    model.train()
    all_preds, all_labels = [], []

    for x_seq, y in dataloader:
        x_seq, y = x_seq.to(device), y.to(device)
        optimizer.zero_grad()

        out = model(x_seq, edge_index.to(device))
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        preds = (torch.sigmoid(out) > 0.5).int().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return loss.item(), acc, f1


def evaluate(model, dataloader, criterion, device, edge_index):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for x_seq, y in dataloader:
            x_seq, y = x_seq.to(device), y.to(device)
            out = model(x_seq, edge_index.to(device))
            loss = criterion(out, y.float())
            total_loss += loss.item() * x_seq.size(0)

            preds = (out > 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(out.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    return avg_loss, acc, f1, auc


split_loader = STGCNDatasetSplitByParticipant("stgcn_data/stgcn_dataset.pt", test_fraction=0.2)
train_dataset, test_dataset, edge_index = split_loader.get_datasets()

logging.info(f"Total samples: {len(train_dataset) + len(test_dataset)}")
logging.info(f"Train samples: {len(train_dataset)}")
logging.info(f"Test samples: {len(test_dataset)}")

labels = []
for _, label in train_dataset:
    labels.append(int(label.item()))
label_counter = collections.Counter(labels)
logging.info(f"Train label distribution: {label_counter}")
labels = []
for _, label in test_dataset:
    labels.append(int(label.item()))
label_counter = collections.Counter(labels)
print(f"Test label distribution: {label_counter}")
print(f"Number of nodes per graph: {train_dataset[0][0].shape[1]}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = STGCNModel(
    in_channels=4,
    hidden_channels=32,
    lstm_hidden=64,
    num_layers=1,
    num_nodes=train_dataset[0][0].shape[1]
).to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info(f"Total trainable parameters in STGCN model: {num_params:,}")

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

all_labels = []
for _, y in train_loader:
    all_labels.extend(y.numpy())
all_labels = torch.tensor(all_labels)
num_pos = (all_labels == 1).sum()
num_neg = (all_labels == 0).sum()
pos_weight = num_neg / num_pos
logging.info(f"pos_weight: {pos_weight:.4f}")
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

history = {
    "train_loss": [],
    "train_acc": [],
    "train_f1": [],
    "test_loss": [],
    "test_acc": [],
    "test_f1": [],
    "test_auc": []
}

logging.info("Model Architecture:")
logging.info(model)
logging.info(f"Optimizer: {optimizer}")
logging.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
logging.info(f"Loss function: BCEWithLogitsLoss with pos_weight={pos_weight.item():.4f}")
logging.info(f"Device: {device}")
logging.info(f"Number of nodes per graph: {train_dataset[0][0].shape[1]}")

num_epochs = 30
for epoch in range(1, num_epochs + 1):
    train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, optimizer, criterion, device, edge_index)
    test_loss, test_acc, test_f1, test_auc = evaluate(model, test_loader, criterion, device, edge_index)
    scheduler.step(test_loss)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["train_f1"].append(train_f1)
    history["test_loss"].append(test_loss)
    history["test_acc"].append(test_acc)
    history["test_f1"].append(test_f1)
    history["test_auc"].append(test_auc)

    logging.info(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} F1: {train_f1:.3f} | "
                f"Test Loss: {test_loss:.4f} Acc: {test_acc:.3f} F1: {test_f1:.3f} AUC: {test_auc:.3f}")


epochs = range(1, num_epochs + 1)

# Compute and plot confusion matrix on the test set
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for x_seq, y in test_loader:
        x_seq, y = x_seq.to(device), y.to(device)
        out = model(x_seq, edge_index.to(device))
        preds = (out > 0.5).int().cpu().numpy()
        labels = y.cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Powerline", "Powerline"])

plt.figure(figsize=(6, 5))
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix on Test Set")
plt.grid(False)
plt.tight_layout()
plt.savefig("./plots/confusion_matrix_test.png", dpi=300)
plt.show()

logging.info("\nClassification Report (on Test Set):\n" + classification_report(all_labels, all_preds, target_names=["No Powerline", "Powerline"], digits=4))

def plot_metric(train_vals, test_vals, label, ylabel):
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_vals, label=f"Train {label}")
    plt.plot(epochs, test_vals, label=f"Test {label}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{label} over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./plots/fig_{label.lower().replace(' ', '_')}.png", dpi=300)
    plt.close()

plot_metric(history["train_loss"], history["test_loss"], "Loss", "BCE Loss")
plot_metric(history["train_acc"], history["test_acc"], "Accuracy", "Accuracy")
plot_metric(history["train_f1"], history["test_f1"], "F1 Score", "F1 Score")
plot_metric(history["test_auc"], history["test_auc"], "AUC", "AUC Score")
