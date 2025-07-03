import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, 1, bias=False)

    def forward(self, H):  # H: [B, T, D]
        α = self.attn(H).squeeze(-1)              # [B, T]
        α = F.softmax(α, dim=1)                   # attention weights
        context = (α.unsqueeze(-1) * H).sum(1)    # weighted sum: [B, D]
        return context


class RNNModel(nn.Module):
    def __init__(self, in_dim, hid=64, layers=2, p_dropout=0.5):
        super().__init__()
        self.gru = nn.GRU(in_dim, hid, num_layers=layers,
                          dropout=0. if layers == 1 else p_dropout,
                          bidirectional=True, batch_first=True)
        self.pool = AttnPool(hid * 2)
        self.drop = nn.Dropout(p_dropout)
        self.head = nn.Linear(hid * 2, 1)

    def forward(self, x):  # x: [B, T, D]
        H, _ = self.gru(x)                # H: [B, T, 2*hid]
        h = self.pool(H)                  # [B, 2*hid]
        h = self.drop(h)
        logits = self.head(h).squeeze(1)  # [B]
        return logits
