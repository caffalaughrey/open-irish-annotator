from __future__ import annotations

import torch
from torch import nn


class CharCNN(nn.Module):
    def __init__(self, vocab_size: int, char_emb_dim: int = 48, out_channels: int = 64, kernel_size: int = 3):
        super().__init__()
        self.char_embedding = nn.Embedding(vocab_size, char_emb_dim, padding_idx=0)
        self.conv = nn.Conv1d(char_emb_dim, out_channels, kernel_size, padding=kernel_size // 2)
        self.activation = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        # char_ids: [batch, tokens, char_len]
        bsz, tseq, clen = char_ids.shape
        x = self.char_embedding(char_ids)  # [b, t, c, e]
        x = x.view(bsz * tseq, clen, -1).transpose(1, 2)  # [b*t, e, c]
        x = self.activation(self.conv(x))  # [b*t, out, c]
        x = self.pool(x).squeeze(-1)  # [b*t, out]
        x = x.view(bsz, tseq, -1)  # [b, t, out]
        return x


class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, tokens, input_dim]
        out, _ = self.lstm(x)
        return out  # [batch, tokens, hidden_dim]


