#!/usr/bin/env python
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    processed = Path("data/processed")
    train_path = processed / "train.jsonl"
    if not train_path.exists():
        raise SystemExit("Missing data/processed/train.jsonl; run make splits first")

    token_to_lemmas: dict[str, Counter[str]] = defaultdict(Counter)
    with train_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            for tok, lemma in zip(obj["tokens"], obj["lemmas"]):
                if tok and lemma:
                    token_to_lemmas[tok][lemma] += 1

    lexicon: dict[str, str] = {}
    for tok, counts in token_to_lemmas.items():
        lemma, _ = counts.most_common(1)[0]
        lexicon[tok] = lemma

    out = processed / "lemma_lexicon.json"
    out.write_text(json.dumps(lexicon, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote lemma lexicon entries: {len(lexicon)} → {out}")


if __name__ == "__main__":
    main()


