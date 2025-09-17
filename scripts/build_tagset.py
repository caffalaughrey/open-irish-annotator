#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
from typing import List
import json

from src.gaeilge_morph.data import (
    read_conllu_sentences,
    build_tagset,
    build_vocabs,
    save_json,
)


def find_ud_files(raw_root: Path) -> List[Path]:
    # Common locations for UD files when unzipped
    candidates = list(raw_root.glob("UD_Irish-IDT-*/**/*.conllu"))
    if not candidates:
        # Also allow user-provided path
        candidates = list(raw_root.glob("**/*.conllu"))
    return sorted(candidates)


def main() -> None:
    raw_root = Path("data/raw")
    processed_root = Path("data/processed")
    processed_root.mkdir(parents=True, exist_ok=True)

    files = find_ud_files(raw_root)
    if not files:
        raise SystemExit("No .conllu files found under data/raw. Run scripts/download_ud_irish.sh first.")

    # Read all sentences from train/dev/test for vocab and tagset
    all_sentences = []
    for p in files:
        for s in read_conllu_sentences(p):
            all_sentences.append(s)

    tag2id = build_tagset(all_sentences)
    word2id, char2id = build_vocabs(all_sentences)

    save_json(tag2id, processed_root / "tagset.json")
    save_json(word2id, processed_root / "word_vocab.json")
    save_json(char2id, processed_root / "char_vocab.json")
    print(
        f"Built resources: {len(tag2id)} tags, {len(word2id)} words, {len(char2id)} chars → data/processed"
    )


if __name__ == "__main__":
    main()


