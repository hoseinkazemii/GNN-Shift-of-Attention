import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# ---------- utility ----------------------------------------------------------
def repeat_edge_index(edge_index: torch.Tensor, num_repeats: int, num_nodes: int):
    """
    Tile `edge_index` num_repeats times, offsetting node IDs each time.
    Returns tensor shape [2, E * num_repeats].
    """
    device   = edge_index.device
    offsets  = torch.arange(num_repeats, device=device) * num_nodes        # [R]
    edge_rep = edge_index.unsqueeze(0) + offsets.view(-1, 1, 1)            # [R,2,E]
    return edge_rep.reshape(2, -1).contiguous()                            # [2, R*E]

# ---------- model ------------------------------------------------------------
class STGCNModel(nn.Module):
    """
    Vectorised implementation of your original ST-GCN block:
      – shared two-layer GCN per frame
      – concatenate node embeddings (no pooling)
      – GRU over time, classifier on last hidden state
    """

    def __init__(self,
                 num_nodes: int,
                 in_channels: int = 4,
                 hidden_channels: int = 32,
                 rnn_hidden: int = 64,
                 num_rnn_layers: int = 1,
                 dropout_p: float = 0.30):
        super().__init__()
        self.num_nodes = num_nodes

        self.gcn1  = GCNConv(in_channels,  hidden_channels)
        self.gcn2  = GCNConv(hidden_channels, hidden_channels)
        self.drop  = nn.Dropout(dropout_p)

        self.rnn   = nn.GRU(input_size=hidden_channels * num_nodes,       # ← unchanged
                            hidden_size=rnn_hidden,
                            num_layers=num_rnn_layers,
                            batch_first=True)

        self.classifier = nn.Linear(rnn_hidden, 1)

    def forward(self, seq: torch.Tensor, edge_index: torch.Tensor):
        """
        seq:        [B, T, N, F]
        edge_index: [2, E]   (single-graph edge list, device-matched)
        """
        B, T, N, in_feats = seq.shape
        assert N == self.num_nodes, "num_nodes mismatch"

        # (1) fold batch & time  →  [B*T, N, F] → [B*T*N, F] for PyG
        x = seq.reshape(B * T, N, in_feats).view(-1, in_feats)

        # (2) replicate & offset edges for every (B,T) graph copy
        edge_rep = repeat_edge_index(edge_index, B * T, N)                

        # (3) two shared GCN layers
        x = F.relu(self.gcn1(x, edge_rep))
        x = self.drop(x)
        x = F.relu(self.gcn2(x, edge_rep))
        x = self.drop(x)

        # (4) reshape back, *flatten* nodes (no mean-pool)
        x = x.view(B * T, N, -1).flatten(start_dim=1)                      # [B*T, N*H]
        x = x.view(B, T, -1)                                               # [B, T, N*H]

        # (5) GRU over time
        rnn_out, _ = self.rnn(x)                                           # [B, T, rnn_hidden]
        logits = self.classifier(rnn_out[:, -1])                           # last timestep
        return logits.squeeze(1)                                           # [B]
