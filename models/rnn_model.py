import torch
import torch.nn as nn


class FlatGRU(nn.Module):
    def __init__(self, in_dim, hid=32, layers=1, p_dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(in_dim, hid, num_layers=layers,
                          batch_first=True, dropout=0. if layers==1 else p_dropout)
        self.drop = nn.Dropout(p_dropout)
        self.head = nn.Linear(hid, 1)

    def forward(self, x):  # x: [B,T,D]
        # if torch.isnan(x).any():
        #     print("[NaN DETECTED] Input x has NaN")

        _, h = self.gru(x)  # h: [layers,B,D]

        # if torch.isnan(h).any():
        #     print("[NaN DETECTED] GRU output h has NaN")

        h = self.drop(h[-1])  # [B, D]
        # if torch.isnan(h).any():
        #     print("[NaN DETECTED] After dropout h has NaN")

        logits = self.head(h).squeeze(1)  # [B]
        # if torch.isnan(logits).any():
        #     print("[NaN DETECTED] Output logits have NaN")
        
        return logits

