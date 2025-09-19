from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Dict

import torch
from torch import nn
from torch.optim import AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from gaeilge_morph.models.model import GaelicMorphModel
from gaeilge_morph.data.dataset import (
    JSONLSentenceDataset,
    make_loader,
    make_length_bucketed_loader,
    PAD_CHAR_ID,
)


@dataclass
class TrainConfig:
    batch_size: int = 16
    lr: float = 2e-3
    max_lr: float = 1e-2
    epochs: int = 5
    device: str = "cpu"
    max_chars: int = 24
    max_lemma: int = 24
    tag_loss_weight: float = 1.0
    lemma_loss_weight: float = 0.5
    label_smoothing: float = 0.0
    optimizer: str = "adamw"  # or "sgd"
    batch_workers: int = 2
    subset_frac: float = 1.0
    early_stop_patience: int = 3
    save_best_only: bool = True
    resume_path: str | None = None
    freeze_encoder: bool = False
    freeze_embeddings: bool = False
    bucket_by_len: bool = False
    use_onecycle: bool = True
    use_kd: bool = False
    kd_tag_weight: float = 0.3
    kd_lemma_weight: float = 0.3
    # Model capacity
    word_emb_dim: int = 128
    char_emb_dim: int = 48
    char_cnn_out: int = 64
    encoder_hidden: int = 256


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
    label_smoothing: float = 0.0,
    teacher_tag_ids: torch.Tensor | None = None,
    teacher_lemma_char_ids: torch.Tensor | None = None,
    kd_tag_weight: float = 0.0,
    kd_lemma_weight: float = 0.0,
) -> torch.Tensor:
    # Tag loss
    if label_smoothing > 0:
        tag_loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing)
        lemma_loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=label_smoothing)
    else:
        tag_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        lemma_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    b, t, k = tag_logits.shape
    tag_loss = tag_loss_fn(tag_logits.view(b * t, k), tag_ids.view(b * t))

    # Lemma loss: only over non-pad time steps; EOS marks end, but we train full length
    b, t, lemma_len, num_chars = lemma_logits.shape
    lemma_loss = lemma_loss_fn(lemma_logits.view(b * t * lemma_len, num_chars), lemma_char_ids.view(b * t * lemma_len))

    total = tag_loss_weight * tag_loss + lemma_loss_weight * lemma_loss
    # Optional KD with hard teacher targets
    if teacher_tag_ids is not None and kd_tag_weight > 0:
        kd_mask = (teacher_tag_ids != -100) & token_mask
        if kd_mask.any():
            b, t, k = tag_logits.shape
            kd_loss = tag_loss_fn(tag_logits.view(b * t, k), teacher_tag_ids.view(b * t))
            total = total + kd_tag_weight * kd_loss
    if teacher_lemma_char_ids is not None and kd_lemma_weight > 0:
        b, t, lemma_len_kd, num_chars_kd = lemma_logits.shape
        kd_lemma_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        kd_le_loss = kd_lemma_loss_fn(
            lemma_logits.view(b * t * lemma_len_kd, num_chars_kd),
            teacher_lemma_char_ids.view(b * t * lemma_len_kd),
        )
        total = total + kd_lemma_weight * kd_le_loss
    return total


def train_one_epoch(
    model: GaelicMorphModel,
    loader: DataLoader,
    optimizer: Optimizer,
    cfg: TrainConfig,
    device: torch.device,
    scheduler: OneCycleLR | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0
    for batch in loader:
        # Support optional teacher tensors at the end of batch
        if len(batch) >= 7:
            word_ids, char_ids, tag_ids, lemma_char_ids, token_mask, teacher_tag_ids, teacher_lemma_char_ids = [
                x.to(device) for x in batch
            ]
        else:
            word_ids, char_ids, tag_ids, lemma_char_ids, token_mask = [x.to(device) for x in batch]
            teacher_tag_ids = None
            teacher_lemma_char_ids = None
        optimizer.zero_grad(set_to_none=True)
        tag_logits, lemma_logits = model(word_ids, char_ids, lemma_char_ids)
        loss = compute_losses(
            tag_logits,
            lemma_logits,
            tag_ids,
            lemma_char_ids,
            token_mask,
            cfg.tag_loss_weight,
            cfg.lemma_loss_weight,
            cfg.label_smoothing,
            teacher_tag_ids,
            teacher_lemma_char_ids,
            cfg.kd_tag_weight,
            cfg.kd_lemma_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
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
            # Handle possible teacher tensors appended; ignore them here
            word_ids, char_ids, tag_ids, lemma_char_ids, token_mask = [x.to(device) for x in batch[:5]]
            tag_logits, lemma_logits = model(word_ids, char_ids, lemma_char_ids)
            # Tag accuracy
            preds = tag_logits.argmax(-1)
            mask = token_mask
            correct_tags += int(((preds == tag_ids) & mask).sum().item())
            total_tags += int(mask.sum().item())
            # Lemma accuracy (exact match per token) ignoring PAD positions
            pred_lemma_ids = lemma_logits.argmax(-1)
            valid_pos = lemma_char_ids != PAD_CHAR_ID
            pos_equal = (pred_lemma_ids == lemma_char_ids) | (~valid_pos)
            exact_match = pos_equal.all(-1) & mask
            correct_lemmas += int(exact_match.sum().item())
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

    # If KD is enabled and teacher files exist, use them
    train_json = train_path.with_suffix(".teacher.jsonl") if cfg.use_kd and train_path.with_suffix(".teacher.jsonl").exists() else train_path
    dev_json = dev_path.with_suffix(".teacher.jsonl") if cfg.use_kd and dev_path.with_suffix(".teacher.jsonl").exists() else dev_path
    train_ds = JSONLSentenceDataset(train_json, word2id, char2id, tag2id, cfg.max_chars, cfg.max_lemma)
    dev_ds = JSONLSentenceDataset(dev_json, word2id, char2id, tag2id, cfg.max_chars, cfg.max_lemma)

    # Optional subset for fast pilot runs
    if cfg.subset_frac < 1.0:
        import math
        import random
        n = len(train_ds)
        k = max(1, int(math.ceil(n * cfg.subset_frac)))
        idx = list(range(n))
        random.shuffle(idx)
        train_ds = Subset(train_ds, idx[:k])
    if cfg.bucket_by_len:
        train_loader = make_length_bucketed_loader(train_ds, batch_size=cfg.batch_size, buckets=10, shuffle=True)
        dev_loader = make_length_bucketed_loader(dev_ds, batch_size=cfg.batch_size, buckets=10, shuffle=False)
    else:
        train_loader = make_loader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        dev_loader = make_loader(dev_ds, batch_size=cfg.batch_size, shuffle=False)

    device = torch.device(cfg.device)
    model = GaelicMorphModel(
        word_vocab_size=len(word2id),
        char_vocab_size=len(char2id),
        tagset_size=len(tag2id),
        lemma_max_len=cfg.max_lemma,
        word_emb_dim=cfg.word_emb_dim,
        char_emb_dim=cfg.char_emb_dim,
        char_cnn_out=cfg.char_cnn_out,
        encoder_hidden=cfg.encoder_hidden,
    ).to(device)
    if cfg.freeze_embeddings:
        for p in model.word_embedding.parameters():
            p.requires_grad = False
    if cfg.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False

    if cfg.optimizer == "sgd":
        optimizer = SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
    else:
        optimizer = AdamW(model.parameters(), lr=cfg.lr)
    scheduler = (
        OneCycleLR(optimizer, max_lr=cfg.max_lr, epochs=cfg.epochs, steps_per_epoch=max(1, len(train_loader)))
        if cfg.use_onecycle
        else None
    )

    # Optional resume
    start_epoch = 1
    best_metric = -1.0
    no_improve = 0
    ckpt_dir = Path("artifacts/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if cfg.resume_path and Path(cfg.resume_path).exists():
        state = torch.load(cfg.resume_path, map_location=device)
        # Warm start: load only model weights to avoid optimizer/scheduler incompatibilities
        model_state = state.get("model", state)
        model.load_state_dict(model_state)
        start_epoch = int(state.get("epoch", 0)) + 1
        best_metric = float(state.get("best_metric", -1.0))

    for epoch in range(start_epoch, cfg.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, cfg, device, scheduler)
        metrics = evaluate(model, dev_loader, device)
        print(f"epoch {epoch}: loss={loss:.4f} tag_acc={metrics['tag_acc']:.3f} lemma_acc={metrics['lemma_acc']:.3f}")
        # epoch-level scheduler step not needed for OneCycle (stepped per batch)

        # Early stopping on combined metric prioritizing lemma
        combined = 0.6 * metrics["lemma_acc"] + 0.4 * metrics["tag_acc"]
        improved = combined > best_metric
        if improved:
            best_metric = combined
            no_improve = 0
            if cfg.save_best_only:
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": (scheduler.state_dict() if scheduler is not None else {}),
                    "best_metric": best_metric,
                }, ckpt_dir / "best.pt")
        else:
            no_improve += 1
            if no_improve >= cfg.early_stop_patience:
                break

    # Save final (and best if not saved-only)
    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    if not cfg.save_best_only:
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": (scheduler.state_dict() if scheduler is not None else {}),
            "best_metric": best_metric,
        }, ckpt_dir / "last.pt")


