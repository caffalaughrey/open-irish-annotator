#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort


def load_resources(res_dir: Path):
    tag2id = json.loads((res_dir / "tagset.json").read_text(encoding="utf-8"))
    word2id = json.loads((res_dir / "word_vocab.json").read_text(encoding="utf-8"))
    char2id = json.loads((res_dir / "char_vocab.json").read_text(encoding="utf-8"))
    tag_id2str = [t for t, _ in sorted(tag2id.items(), key=lambda kv: kv[1])]
    lex_path = res_dir / "lemma_lexicon.json"
    lemma_lex = json.loads(lex_path.read_text(encoding="utf-8")) if lex_path.exists() else {}
    return tag_id2str, word2id, char2id, lemma_lex


def encode(tokens: List[str], word2id, char2id, max_chars: int = 24):
    t = len(tokens)
    word_ids = np.zeros((1, t), dtype=np.int64)
    char_ids = np.zeros((1, t, max_chars), dtype=np.int64)
    for i, tok in enumerate(tokens):
        word_ids[0, i] = word2id.get(tok, 1)
        for j, ch in enumerate(tok[: max_chars - 1]):
            char_ids[0, i, j] = char2id.get(ch, 2)
    return word_ids, char_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokens", nargs="*", help="space tokenized sentence")
    ap.add_argument("--model", default="artifacts/onnx/model.onnx")
    ap.add_argument(
        "--resources",
        default="rust/morphology_runtime/resources",
        help="directory with tagset/word_vocab/char_vocab",
    )
    ap.add_argument("--prefer-lexicon", action="store_true", help="prefer lemma_lexicon.json if present")
    args = ap.parse_args()

    res_dir = Path(args.resources)
    tag_id2str, word2id, char2id, lemma_lex = load_resources(res_dir)

    if not args.tokens:
        print("usage: onnx_analyze.py <token> [<token> ...]")
        return
    word_ids, char_ids = encode(args.tokens, word2id, char2id)

    sess = ort.InferenceSession(Path(args.model).as_posix(), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"word_ids": word_ids, "char_ids": char_ids})
    tag_logits, lemma_logits = outs
    tag_ids = tag_logits.argmax(axis=-1)[0]
    lemma_ids = lemma_logits.argmax(axis=-1)[0]

    # Convert lemma char IDs to strings until EOS=1
    id2ch = [ch for ch, _ in sorted(char2id.items(), key=lambda kv: kv[1])]
    results = []
    for i, tok in enumerate(args.tokens):
        tag = tag_id2str[tag_ids[i]]
        lemma_chars = []
        for cid in lemma_ids[i]:
            if cid == 1:
                break
            lemma_chars.append(id2ch[cid] if cid < len(id2ch) else "?")
        decoded = "".join(lemma_chars) or tok
        lemma = lemma_lex.get(tok, decoded) if args.prefer_lexicon else (decoded or lemma_lex.get(tok, decoded))
        results.append((tok, tag, lemma))

    for tok, tag, lemma in results:
        print(f"{tok}\t{tag}\t{lemma}")


if __name__ == "__main__":
    main()


