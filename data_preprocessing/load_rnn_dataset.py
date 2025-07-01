import math, random
import torch
from collections import defaultdict


class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, seqs, labels, indices):
        self.x = [seqs[i] for i in indices]
        self.y = [labels[i] for i in indices]
    def __len__(self): 
        return len(self.x)
    def __getitem__(self, idx): 
        return self.x[idx], torch.tensor(self.y[idx], dtype=torch.float)


def load_rnn_dataset(path, seed=42, **params):
    test_fraction = params.get("test_fraction")

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

    num_test = math.ceil(len(all_participants) * test_fraction)
    test_participants = all_participants[:num_test]
    train_participants = all_participants[num_test:]

    # Collect corresponding sample indices
    train_indices = [i for p in train_participants for i in participant_to_indices[p]]
    test_indices  = [i for p in test_participants  for i in participant_to_indices[p]]

    train_ds = SeqDataset(seqs, labels, train_indices)
    test_ds  = SeqDataset(seqs, labels, test_indices)
    y_train  = torch.tensor([y for _, y in train_ds])
    pos_w    = (y_train == 0).sum() / (y_train == 1).sum()
    pos_w = torch.tensor(pos_w, dtype=torch.float)
    input_dim = train_ds[0][0].shape[-1]

    return train_ds, test_ds, pos_w, input_dim
