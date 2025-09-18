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
        # Lemma decoder: GRU over characters with teacher forcing during training
        self.lemma_char_embedding = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        self.lemma_init = nn.Linear(encoder_hidden, encoder_hidden)
        self.lemma_gru = nn.GRU(input_size=char_emb_dim, hidden_size=encoder_hidden, batch_first=True)
        self.lemma_project = nn.Linear(encoder_hidden, char_vocab_size)

    def forward(
        self,
        word_ids: torch.Tensor,
        char_ids: torch.Tensor,
        lemma_char_ids: torch.Tensor | None = None,
    ):
        # word_ids: [b, t]
        # char_ids: [b, t, c]
        word_vec = self.word_embedding(word_ids)
        char_vec = self.char_encoder(char_ids)
        enc_in = torch.cat([word_vec, char_vec], dim=-1)
        enc_out = self.encoder(enc_in)  # [b, t, h]

        tag_logits = self.tag_head(enc_out)  # [b, t, tagset]

        # Lemma decoding
        bsz, tseq, _ = enc_out.shape
        lemma_len = self.lemma_max_len
        # Teacher forcing path expects gold lemma ids
        if lemma_char_ids is None:
            # Greedy inference with previous predictions; simple loop
            device = enc_out.device
            bos_id = 3  # matches data.vocab BOS
            prev_ids = torch.full((bsz, tseq), bos_id, dtype=torch.long, device=device)
            h0 = torch.tanh(self.lemma_init(enc_out.reshape(bsz * tseq, -1))).unsqueeze(0)
            logits_steps: list[torch.Tensor] = []
            h = h0
            for _ in range(lemma_len):
                emb = self.lemma_char_embedding(prev_ids).reshape(bsz * tseq, 1, -1)
                out, h = self.lemma_gru(emb, h)
                step_logits = self.lemma_project(out)  # [b*t, 1, C]
                logits_steps.append(step_logits)
                prev_ids = step_logits.squeeze(1).argmax(-1).reshape(bsz, tseq)
            lemma_logits = torch.cat(logits_steps, dim=1).reshape(bsz, tseq, lemma_len, -1)
        else:
            # Teacher forcing: input = BOS + gold[:-1]
            device = enc_out.device
            bos_id = 3
            inp = torch.full_like(lemma_char_ids, bos_id)
            inp[:, :, 1:] = lemma_char_ids[:, :, :-1]
            emb = self.lemma_char_embedding(inp)  # [b, t, l, e]
            emb = emb.reshape(bsz * tseq, lemma_len, -1)
            h0 = torch.tanh(self.lemma_init(enc_out.reshape(bsz * tseq, -1))).unsqueeze(0)
            out, _ = self.lemma_gru(emb, h0)  # [b*t, l, h]
            proj = self.lemma_project(out)  # [b*t, l, C]
            lemma_logits = proj.reshape(bsz, tseq, lemma_len, -1)

        return tag_logits, lemma_logits


