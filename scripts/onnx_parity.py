#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from gaeilge_morph.models.model import GaelicMorphModel


def main() -> None:
    processed = Path("data/processed")
    tag2id = json.loads((processed / "tagset.json").read_text(encoding="utf-8"))
    word2id = json.loads((processed / "word_vocab.json").read_text(encoding="utf-8"))
    char2id = json.loads((processed / "char_vocab.json").read_text(encoding="utf-8"))

    model = GaelicMorphModel(len(word2id), len(char2id), len(tag2id))
    ckpt = Path("artifacts/checkpoints/model.pt")
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    b, t, c = 1, 4, 6
    word_ids = torch.randint(low=0, high=len(word2id), size=(b, t))
    char_ids = torch.randint(low=0, high=len(char2id), size=(b, t, c))

    with torch.no_grad():
        torch_tag, torch_lemma = model(word_ids, char_ids)

    onnx_path = Path("artifacts/onnx/model.onnx")
    if not onnx_path.exists():
        raise SystemExit("Export ONNX first (make export)")

    sess = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    outs = sess.run(
        None,
        {
            "word_ids": word_ids.numpy().astype(np.int64),
            "char_ids": char_ids.numpy().astype(np.int64),
        },
    )
    onnx_tag, onnx_lemma = (torch.from_numpy(outs[0]), torch.from_numpy(outs[1]))

    def close(a: torch.Tensor, b: torch.Tensor) -> bool:
        return torch.allclose(a, b, atol=1e-4, rtol=1e-4)

    ok1 = close(torch_tag, onnx_tag)
    ok2 = close(torch_lemma, onnx_lemma)
    print(f"parity: tag_logits={ok1} lemma_logits={ok2}")


if __name__ == "__main__":
    main()


