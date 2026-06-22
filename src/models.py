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


# ─────────────────────────────────────────────────────
#  DAY 4 — ENSEMBLE COMPONENTS
# ─────────────────────────────────────────────────────

class CNNBranch(nn.Module):
    """
    1D CNN over the raw Qdlin voltage-capacity curve (1000 points,
    downsampled). Convolution kernels scan the curve for local shape
    patterns (peak height/width/position) directly from signal —
    complementary to the hand-engineered IC peak stats.
    """

    def __init__(self, seq_len: int, out_dim: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),   # global average pool — keeps params low
        )
        self.head = nn.Linear(16, out_dim)

    def forward(self, x):
        # x: (batch, seq_len) -> (batch, 1, seq_len) for Conv1d
        x = x.unsqueeze(1)
        x = self.conv(x)              # (batch, 16, 1)
        x = x.squeeze(-1)             # (batch, 16)
        return self.head(x)           # (batch, out_dim)


class SequenceBranch(nn.Module):
    """
    LSTM or GRU over the (n_checkpoints, n_features) feature sequence —
    learns how engineered features evolve as the battery ages.
    """

    def __init__(self, n_features: int, hidden_size: int = 24,
                 out_dim: int = 16, cell_type: str = "lstm"):
        super().__init__()
        rnn_cls = nn.LSTM if cell_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=1, batch_first=True,
        )
        # forget-gate bias init trick (LSTM only) — helps gradient flow
        if cell_type == "lstm":
            for name, param in self.rnn.named_parameters():
                if "bias" in name:
                    n = param.size(0)
                    param.data[n // 4: n // 2].fill_(1.0)

        self.head = nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        # x: (batch, n_checkpoints, n_features)
        out, state = self.rnn(x)
        last_hidden = state[0][-1] if isinstance(state, tuple) else state[-1]
        return self.head(last_hidden)   # (batch, out_dim)


class RULEnsemble(nn.Module):
    """
    Combines CNN (raw curve shape) + LSTM + GRU (feature sequences)
    branches. Each branch outputs a small embedding; embeddings are
    concatenated and passed through a final regression head.

    A learned scalar weight per branch lets the model lean on whichever
    branch is most useful, rather than forcing equal-weight averaging.
    """

    def __init__(self, n_features: int, cnn_seq_len: int,
                 branch_dim: int = 16, hidden_size: int = 24):
        super().__init__()
        self.cnn  = CNNBranch(cnn_seq_len, out_dim=branch_dim)
        self.lstm = SequenceBranch(n_features, hidden_size, branch_dim, cell_type="lstm")
        self.gru  = SequenceBranch(n_features, hidden_size, branch_dim, cell_type="gru")

        self.combine_head = nn.Sequential(
            nn.Linear(branch_dim * 3, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 1),
        )

    def forward(self, qdlin, feature_seq):
        # qdlin:       (batch, cnn_seq_len)
        # feature_seq: (batch, n_checkpoints, n_features)
        cnn_emb  = self.cnn(qdlin)
        lstm_emb = self.lstm(feature_seq)
        gru_emb  = self.gru(feature_seq)

        combined = torch.cat([cnn_emb, lstm_emb, gru_emb], dim=-1)
        out = self.combine_head(combined)
        return out.squeeze(-1)