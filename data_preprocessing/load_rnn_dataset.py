import math, random
import torch
from collections import defaultdict


class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, seqs, labels, indices):
        self.x = [seqs[i] for i in indices]
        self.y = [labels[i] for i in indices]
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], torch.tensor(self.y[idx], dtype=torch.float)


def load_rnn_dataset(path="./processed_data/rnn_dataset.pt", **params):
    test_fraction = params.get("test_fraction")

    seqs, labels, srcs = torch.load(path, weights_only=False)
    p2idx = defaultdict(list)
    for i, s in enumerate(srcs):
        p2idx[s].append(i)

    participants = sorted(p2idx.keys())
    random.seed(42); random.shuffle(participants)
    n_test = math.ceil(len(participants) * test_fraction)
    test_p = participants[:n_test]
    train_p = participants[n_test:]
    tr_idx = [i for p in train_p for i in p2idx[p]]
    te_idx = [i for p in test_p  for i in p2idx[p]]

    train_ds = SeqDataset(seqs, labels, tr_idx)
    test_ds  = SeqDataset(seqs, labels, te_idx)
    y_train  = torch.tensor([y for _, y in train_ds])
    pos_w    = (y_train == 0).sum() / (y_train == 1).sum()
    input_dim = train_ds[0][0].shape[-1]

    return train_ds, test_ds, pos_w, input_dim
