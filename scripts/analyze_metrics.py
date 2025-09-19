#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


def load_resources(processed: Path):
    tag2id = json.loads((processed / "tagset.json").read_text(encoding="utf-8"))
    word2id = json.loads((processed / "word_vocab.json").read_text(encoding="utf-8"))
    char2id = json.loads((processed / "char_vocab.json").read_text(encoding="utf-8"))
    id2tag = [t for t, _ in sorted(tag2id.items(), key=lambda kv: kv[1])]
    return tag2id, id2tag, word2id, char2id


def conll_from_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def oov_stats(rows: List[Dict], word2id: Dict[str,int]) -> Tuple[int,int,float]:
    total = 0
    oov = 0
    for r in rows:
        for w in r["tokens"]:
            total += 1
            if w not in word2id:
                oov += 1
    rate = (oov / max(total,1)) if total else 0.0
    return total, oov, rate


def length_buckets(rows: List[Dict]) -> Counter:
    buckets = Counter()
    for r in rows:
        n = len(r["tokens"])
        if n <= 5: buckets['<=5'] += 1
        elif n <= 10: buckets['6-10'] += 1
        elif n <= 20: buckets['11-20'] += 1
        else: buckets['>20'] += 1
    return buckets


def upos_confusion(rows: List[Dict]) -> Dict[Tuple[str,str], int]:
    # Placeholder: we need model predictions to compute confusion. Here we can compute gold UPOS distribution.
    conf = Counter()
    for r in rows:
        for tag in r["tags"]:
            upos = tag.split('|',1)[0]
            conf[(upos, upos)] += 1
    return dict(conf)


def feats_f1_placeholder(rows: List[Dict]) -> Dict[str, float]:
    # Placeholder: requires predictions. Report coverage of FEATS types in gold for now.
    feat_counts = Counter()
    for r in rows:
        for tag in r["tags"]:
            parts = tag.split('|')[1:]
            for p in parts:
                k = p.split('=')[0]
                feat_counts[k] += 1
    return {k: float(v) for k,v in feat_counts.items()}


def main() -> None:
    processed = Path("data/processed")
    tag2id, id2tag, word2id, char2id = load_resources(processed)
    train = conll_from_jsonl(processed / "train.jsonl")
    dev = conll_from_jsonl(processed / "dev.jsonl")
    test = conll_from_jsonl(processed / "test.jsonl")

    for split, rows in [("train", train), ("dev", dev), ("test", test)]:
        total, oov, rate = oov_stats(rows, word2id)
        buckets = length_buckets(rows)
        feats_cov = feats_f1_placeholder(rows)
        print(f"[{split}] tokens={total} OOV={oov} ({rate:.2%}) len_buckets={dict(buckets)} feats_seen={len(feats_cov)}")


if __name__ == "__main__":
    main()


