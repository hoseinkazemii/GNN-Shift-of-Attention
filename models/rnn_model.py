import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=1, dropout=0.5, rnn_type='LSTM'):
        super(RNNModel, self).__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(in_dim)
        
        # Simpler RNN architecture (no bidirectional to reduce complexity)
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=in_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=False  # Simplified
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=in_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=False  # Simplified
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # Much simpler classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        # L2 regularization will be handled by weight_decay in optimizer
        
    def forward(self, x):
        batch_size, seq_len, feat_dim = x.shape
        
        # Normalize input features across the feature dimension
        x_reshaped = x.view(-1, feat_dim)
        x_norm = self.input_norm(x_reshaped)
        x = x_norm.view(batch_size, seq_len, feat_dim)
        
        # RNN forward pass
        rnn_out, _ = self.rnn(x)
        
        # Use last time step output
        last_output = rnn_out[:, -1, :]
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        # Classification
        logits = self.classifier(last_output)
        
        return logits.squeeze(-1)
