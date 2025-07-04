import math, random
import torch
from collections import defaultdict


class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, seqs, labels, indices):
        self.x = [seqs[i] for i in indices]
        self.y = [labels[i] for i in indices]
        self.y_tensor = torch.tensor(self.y, dtype=torch.float)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y_tensor[idx]


def load_rnn_dataset(path, seed=42, **params):
    test_fraction = params.get("test_fraction")
    val_fraction = params.get("val_fraction")

    seqs, labels, srcs = torch.load(path, weights_only=False)

    labels_tensor = torch.tensor(labels)
    print("\nLabel distribution in the FULL dataset:")
    print("Positive samples:", (labels_tensor == 1).sum().item())
    print("Negative samples:", (labels_tensor == 0).sum().item())

    # Group indices by participant
    participant_to_indices = defaultdict(list)
    for i, src in enumerate(srcs):
        participant_to_indices[src].append(i)

    all_participants = sorted(participant_to_indices.keys())
    random.seed(seed)
    random.shuffle(all_participants)

    num_total = len(all_participants)
    num_test = math.ceil(num_total * test_fraction)
    num_val  = math.ceil(num_total * val_fraction)

    test_participants = all_participants[:num_test]
    val_participants  = all_participants[num_test:num_test+num_val]
    train_participants = all_participants[num_test+num_val:]

    # Collect sample indices
    train_indices = [i for p in train_participants for i in participant_to_indices[p]]
    val_indices   = [i for p in val_participants   for i in participant_to_indices[p]]
    test_indices  = [i for p in test_participants  for i in participant_to_indices[p]]

    train_ds = SeqDataset(seqs, labels, train_indices)
    val_ds   = SeqDataset(seqs, labels, val_indices)
    test_ds  = SeqDataset(seqs, labels, test_indices)

    # Print label distributions
    def label_stats(name, dataset):
        y = torch.stack([label for _, label in dataset])
        pos = (y == 1).sum().item()
        neg = (y == 0).sum().item()
        print(f"{name:<10} - Pos: {pos:5d}  Neg: {neg:5d}  Total: {len(dataset):5d}")
        return y

    print("\nLabel distribution:")
    y_train = label_stats("Train set", train_ds)
    y_val   = label_stats("Val set", val_ds)
    y_test  = label_stats("Test set", test_ds)

    # Compute pos_weight for training
    pos_w = (y_train == 0).sum() / (y_train == 1).sum()
    pos_w = torch.tensor(pos_w, dtype=torch.float)

    input_dim = train_ds[0][0].shape[-1]

    return train_ds, val_ds, test_ds, pos_w, input_dim
