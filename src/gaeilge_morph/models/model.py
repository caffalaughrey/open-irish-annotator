from __future__ import annotations

import torch
from torch import nn

from .components import CharCNN, BiLSTMEncoder


class GaelicMorphModel(nn.Module):
    def __init__(
        self,
        word_vocab_size: int,
        char_vocab_size: int,
        tagset_size: int,
        word_emb_dim: int = 128,
        char_emb_dim: int = 48,
        char_cnn_out: int = 64,
        encoder_hidden: int = 256,
        lemma_max_len: int = 24,
    ) -> None:
        super().__init__()
        self.lemma_max_len = lemma_max_len

        self.word_embedding = nn.Embedding(word_vocab_size, word_emb_dim, padding_idx=0)
        self.char_encoder = CharCNN(char_vocab_size, char_emb_dim, char_cnn_out)
        self.encoder = BiLSTMEncoder(word_emb_dim + char_cnn_out, encoder_hidden)

        self.tag_head = nn.Linear(encoder_hidden, tagset_size)

        # Simple lemma decoder: project encoder state to char logits per step (tied across steps)
        self.lemma_project = nn.Linear(encoder_hidden, char_vocab_size)

    def forward(
        self,
        word_ids: torch.Tensor,
        char_ids: torch.Tensor,
    ):
        # word_ids: [b, t]
        # char_ids: [b, t, c]
        word_vec = self.word_embedding(word_ids)
        char_vec = self.char_encoder(char_ids)
        enc_in = torch.cat([word_vec, char_vec], dim=-1)
        enc_out = self.encoder(enc_in)  # [b, t, h]

        tag_logits = self.tag_head(enc_out)  # [b, t, tagset]

        # Lemma logits for each step share the same projection from encoder states
        # Output shape: [b, t, lemma_len, char_vocab]
        lemma_logits = self.lemma_project(enc_out).unsqueeze(2).repeat(1, 1, self.lemma_max_len, 1)
        return tag_logits, lemma_logits


