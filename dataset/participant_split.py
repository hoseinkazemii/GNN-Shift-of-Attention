import torch
from torch.utils.data import Dataset
from collections import defaultdict
import math
import random
import logging


class STGCNDatasetSplitByParticipant:
    def __init__(self, data_path, test_fraction=0.2, seed=42):
        self.sequences, self.labels, self.object_names, self.edge_index, self.source_files = torch.load(data_path, weights_only=False)
        self.object_names = self.object_names
        self.edge_index = self.edge_index

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
        logging.info(f"Total samples: {len(train_dataset) + len(test_dataset)}")
        logging.info(f"Train samples: {len(train_dataset)}")
        logging.info(f"Test samples: {len(test_dataset)}")
        return train_dataset, test_dataset, self.edge_index


class ParticipantIndexedDataset(Dataset):
    def __init__(self, sequences, labels, indices):
        self.sequences = [sequences[i] for i in indices]
        self.labels    = [labels[i]    for i in indices]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], torch.tensor(self.labels[idx],
                                                 dtype=torch.float)
