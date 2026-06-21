"""
models.py — Day 3+: Model architectures
Day 3: SimpleLSTM baseline (single LSTM on capacity-fade curve)
Day 4 will add: CNN, GRU, and the Ensemble combining all three.
"""

import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):
    """
    A single LSTM that reads the capacity-fade curve (first `checkpoint`
    cycles of SOH%) as a sequence and predicts total cycle_life.

    Input shape:  (batch, seq_len, 1)
    Output shape: (batch,)
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 64,
                 num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # forget-gate bias init trick: start the LSTM "remembering" more by
        # default, which substantially helps gradient flow through long
        # sequences when training data is limited (standard, well-known fix)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                n = param.size(0)
                start, end = n // 4, n // 2   # forget gate slice
                param.data[start:end].fill_(1.0)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]           # (batch, hidden_size) — final LSTM state
        out = self.head(last_hidden)    # (batch, 1)
        return out.squeeze(-1)          # (batch,)


class SimpleMLP(nn.Module):
    """
    A small feedforward network for regression on scalar (non-sequential)
    features — e.g. the 18 engineered features from Day 2. Used as Day 3's
    'neural network baseline' to compare against Elastic Net on equal
    footing (same inputs). SimpleLSTM above is reserved for genuine
    multi-timestep sequence tasks (e.g. Day 4's dynamic checkpoints).
    """

    def __init__(self, n_features: int, hidden_size: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        # x: (batch, n_features)
        return self.net(x).squeeze(-1)