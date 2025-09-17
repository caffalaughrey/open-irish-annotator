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
    torch.manual_seed(0)
    word_ids = torch.randint(low=0, high=len(word2id), size=(b, t))
    char_ids = torch.randint(low=0, high=len(char2id), size=(b, t, c))

    with torch.no_grad():
        torch_tag, torch_lemma = model(word_ids, char_ids)
        torch_tag_idx = torch_tag.argmax(-1)
        torch_lemma_idx = torch_lemma.argmax(-1)

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
    onnx_tag_idx = onnx_tag.argmax(-1)
    onnx_lemma_idx = onnx_lemma.argmax(-1)

    tag_ok = torch.equal(torch_tag_idx, onnx_tag_idx)
    lemma_ok = torch.equal(torch_lemma_idx, onnx_lemma_idx)
    print(f"parity(argmax): tag={tag_ok} lemma={lemma_ok}")


if __name__ == "__main__":
    main()


