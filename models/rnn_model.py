import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=2, dropout=0.5, rnn_type='GRU'):
        super(RNNModel, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.bidirectional = True

        self.input_norm = nn.LayerNorm(in_dim)
        rnn_output_dim = hidden_dim * 2 if self.bidirectional else hidden_dim

        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=in_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=self.bidirectional
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=in_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=self.bidirectional
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(rnn_output_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, in_dim)
        x = self.input_norm(x)

        # rnn_out: (batch_size, seq_len, hidden_dim)
        rnn_out, _ = self.rnn(x)

        # Apply mean pooling over time
        pooled_output = torch.mean(rnn_out, dim=1)  # (batch_size, hidden_dim)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits.squeeze(-1)
