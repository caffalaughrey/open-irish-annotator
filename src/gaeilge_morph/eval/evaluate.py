from __future__ import annotations

from pathlib import Path
import json
import torch

from src.gaeilge_morph.models.model import GaelicMorphModel
from src.gaeilge_morph.data.dataset import JSONLSentenceDataset, make_loader


def main() -> None:
    processed = Path("data/processed")
    tag2id = json.loads((processed / "tagset.json").read_text(encoding="utf-8"))
    word2id = json.loads((processed / "word_vocab.json").read_text(encoding="utf-8"))
    char2id = json.loads((processed / "char_vocab.json").read_text(encoding="utf-8"))

    test_ds = JSONLSentenceDataset(processed / "test.jsonl", word2id, char2id, tag2id)
    loader = make_loader(test_ds, batch_size=16, shuffle=False)

    model = GaelicMorphModel(len(word2id), len(char2id), len(tag2id))
    state_path = Path("artifacts/checkpoints/model.pt")
    if state_path.exists():
        model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.eval()

    # Simple accuracy on test
    correct_tags = 0
    total_tags = 0
    with torch.no_grad():
        for batch in loader:
            word_ids, char_ids, tag_ids, lemma_char_ids, token_mask = batch
            tag_logits, _ = model(word_ids, char_ids)
            preds = tag_logits.argmax(-1)
            mask = token_mask
            correct_tags += int(((preds == tag_ids) & mask).sum().item())
            total_tags += int(mask.sum().item())

    acc = correct_tags / max(total_tags, 1)
    print(f"test tag_acc={acc:.3f}")


if __name__ == "__main__":
    main()


