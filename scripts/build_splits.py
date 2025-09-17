#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import json
from typing import List

from src.gaeilge_morph.data import read_conllu_sentences


def find_split_files(raw_root: Path):
    # Prefer official train/dev/test file names where available
    candidates = list(raw_root.glob("UD_Irish-IDT-*/**/*.conllu"))
    if not candidates:
        candidates = list(raw_root.glob("**/*.conllu"))

    train, dev, test = [], [], []
    for p in candidates:
        name = p.name.lower()
        if "train" in name:
            train.append(p)
        elif "dev" in name or "development" in name:
            dev.append(p)
        elif "test" in name:
            test.append(p)
    return train, dev, test


def write_jsonl(sentences, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for s in sentences:
            obj = {"tokens": s.tokens, "lemmas": s.lemmas, "tags": s.tag_strings}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    raw_root = Path("data/raw")
    processed = Path("data/processed")
    train_files, dev_files, test_files = find_split_files(raw_root)
    if not (train_files and dev_files and test_files):
        raise SystemExit("Could not find train/dev/test .conllu files. Check data/raw contents.")

    splits = {
        "train": train_files,
        "dev": dev_files,
        "test": test_files,
    }
    for split, files in splits.items():
        sentences = []
        for p in files:
            sentences.extend(list(read_conllu_sentences(p)))
        write_jsonl(sentences, processed / f"{split}.jsonl")
        print(f"Wrote {len(sentences)} sentences to {processed / f'{split}.jsonl'}")


if __name__ == "__main__":
    main()


