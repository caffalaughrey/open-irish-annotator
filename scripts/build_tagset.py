#!/usr/bin/env python
from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    raw_root = Path("data/raw")
    processed_root = Path("data/processed")
    processed_root.mkdir(parents=True, exist_ok=True)

    # Stub: write empty resources to unblock runtime; implement real builder later
    (processed_root / "tagset.json").write_text(json.dumps({"__stub__": True}, ensure_ascii=False, indent=2))
    (processed_root / "word_vocab.json").write_text(json.dumps({"__stub__": True}, ensure_ascii=False, indent=2))
    (processed_root / "char_vocab.json").write_text(json.dumps({"__stub__": True}, ensure_ascii=False, indent=2))
    print("Wrote stub tagset and vocabs to data/processed/")


if __name__ == "__main__":
    main()


