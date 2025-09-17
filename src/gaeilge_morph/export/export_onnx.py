from __future__ import annotations

import json
from pathlib import Path

import torch

from src.gaeilge_morph.models.model import GaelicMorphModel


def main() -> None:
    processed = Path("data/processed")
    tagset = json.loads((processed / "tagset.json").read_text(encoding="utf-8"))
    word_vocab = json.loads((processed / "word_vocab.json").read_text(encoding="utf-8"))
    char_vocab = json.loads((processed / "char_vocab.json").read_text(encoding="utf-8"))

    model = GaelicMorphModel(
        word_vocab_size=len(word_vocab),
        char_vocab_size=len(char_vocab),
        tagset_size=len(tagset),
    )
    model.eval()

    # Dummy dynamic shapes
    bsz, tlen, clen = 1, 5, 12
    word_ids = torch.zeros((bsz, tlen), dtype=torch.long)
    char_ids = torch.zeros((bsz, tlen, clen), dtype=torch.long)

    onnx_path = Path("artifacts/onnx/model.onnx")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = {
        "word_ids": {0: "batch", 1: "tokens"},
        "char_ids": {0: "batch", 1: "tokens", 2: "chars"},
        "tag_logits": {0: "batch", 1: "tokens"},
        "lemma_logits": {0: "batch", 1: "tokens", 2: "lemma_len"},
    }

    with torch.no_grad():
        torch.onnx.export(
            model,
            (word_ids, char_ids),
            onnx_path.as_posix(),
            input_names=["word_ids", "char_ids"],
            output_names=["tag_logits", "lemma_logits"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
        )
    print(f"Exported ONNX to {onnx_path}")


if __name__ == "__main__":
    main()


