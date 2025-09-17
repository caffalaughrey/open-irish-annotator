from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Dict

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from gaeilge_morph.models.model import GaelicMorphModel
from gaeilge_morph.data.dataset import JSONLSentenceDataset, make_loader, EOS_CHAR_ID


@dataclass
class TrainConfig:
    batch_size: int = 16
    lr: float = 2e-3
    epochs: int = 5
    device: str = "cpu"
    max_chars: int = 24
    max_lemma: int = 24
    tag_loss_weight: float = 1.0
    lemma_loss_weight: float = 0.5


def load_resources(processed: Path):
    tag2id = json.loads((processed / "tagset.json").read_text(encoding="utf-8"))
    word2id = json.loads((processed / "word_vocab.json").read_text(encoding="utf-8"))
    char2id = json.loads((processed / "char_vocab.json").read_text(encoding="utf-8"))
    return tag2id, word2id, char2id


def compute_losses(
    tag_logits: torch.Tensor,
    lemma_logits: torch.Tensor,
    tag_ids: torch.Tensor,
    lemma_char_ids: torch.Tensor,
    token_mask: torch.Tensor,
    tag_loss_weight: float,
    lemma_loss_weight: float,
) -> torch.Tensor:
    # Tag loss
    tag_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    b, t, k = tag_logits.shape
    tag_loss = tag_loss_fn(tag_logits.view(b * t, k), tag_ids.view(b * t))

    # Lemma loss: only over non-pad time steps; EOS marks end, but we train full length
    b, t, l, c = lemma_logits.shape
    lemma_loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # PAD_CHAR_ID == 0
    lemma_loss = lemma_loss_fn(lemma_logits.view(b * t * l, c), lemma_char_ids.view(b * t * l))

    return tag_loss_weight * tag_loss + lemma_loss_weight * lemma_loss


def train_one_epoch(
    model: GaelicMorphModel,
    loader: DataLoader,
    optimizer: AdamW,
    cfg: TrainConfig,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0
    for batch in loader:
        word_ids, char_ids, tag_ids, lemma_char_ids, token_mask = [x.to(device) for x in batch]
        optimizer.zero_grad(set_to_none=True)
        tag_logits, lemma_logits = model(word_ids, char_ids)
        loss = compute_losses(
            tag_logits, lemma_logits, tag_ids, lemma_char_ids, token_mask, cfg.tag_loss_weight, cfg.lemma_loss_weight
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.item())
        steps += 1
    return total_loss / max(steps, 1)


def evaluate(model: GaelicMorphModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    correct_tags = 0
    total_tags = 0
    correct_lemmas = 0
    total_lemmas = 0
    with torch.no_grad():
        for batch in loader:
            word_ids, char_ids, tag_ids, lemma_char_ids, token_mask = [x.to(device) for x in batch]
            tag_logits, lemma_logits = model(word_ids, char_ids)
            # Tag accuracy
            preds = tag_logits.argmax(-1)
            mask = token_mask
            correct_tags += int(((preds == tag_ids) & mask).sum().item())
            total_tags += int(mask.sum().item())
            # Lemma accuracy (exact match per token)
            pred_lemma_ids = lemma_logits.argmax(-1)
            correct_lemmas += int(((pred_lemma_ids == lemma_char_ids).all(-1) & mask).sum().item())
            total_lemmas += int(mask.sum().item())
    return {
        "tag_acc": correct_tags / max(total_tags, 1),
        "lemma_acc": correct_lemmas / max(total_lemmas, 1),
    }


def run_training(cfg: TrainConfig) -> None:
    processed = Path("data/processed")
    tag2id, word2id, char2id = load_resources(processed)
    train_path = processed / "train.jsonl"
    dev_path = processed / "dev.jsonl"

    train_ds = JSONLSentenceDataset(train_path, word2id, char2id, tag2id, cfg.max_chars, cfg.max_lemma)
    dev_ds = JSONLSentenceDataset(dev_path, word2id, char2id, tag2id, cfg.max_chars, cfg.max_lemma)
    train_loader = make_loader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    dev_loader = make_loader(dev_ds, batch_size=cfg.batch_size, shuffle=False)

    device = torch.device(cfg.device)
    model = GaelicMorphModel(
        word_vocab_size=len(word2id),
        char_vocab_size=len(char2id),
        tagset_size=len(tag2id),
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr)

    for epoch in range(1, cfg.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, cfg, device)
        metrics = evaluate(model, dev_loader, device)
        print(f"epoch {epoch}: loss={loss:.4f} tag_acc={metrics['tag_acc']:.3f} lemma_acc={metrics['lemma_acc']:.3f}")

    # Save checkpoint minimally
    ckpt_dir = Path("artifacts/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "model.pt")


