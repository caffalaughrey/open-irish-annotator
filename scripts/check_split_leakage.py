#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Set


def load_sentences(p: Path) -> List[str]:
    s: List[str] = []
    with p.open('r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            s.append(' '.join(obj['tokens']))
    return s


def shingles(text: str, n: int = 5) -> Set[str]:
    toks = text.split()
    return { ' '.join(toks[i:i+n]) for i in range(0, max(0, len(toks)-n+1)) }


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def main() -> None:
    proc = Path('data/processed')
    train = load_sentences(proc/'train.jsonl')
    dev = load_sentences(proc/'dev.jsonl')
    test = load_sentences(proc/'test.jsonl')

    # Compare dev/test against train using 5-gram Jaccard; flag pairs > 0.8
    train_sh = [shingles(s) for s in train]
    for name, split in [('dev', dev), ('test', test)]:
        flagged = 0
        for si, s in enumerate(split):
            sh = shingles(s)
            for tj, tsh in enumerate(train_sh):
                if jaccard(sh, tsh) > 0.8:
                    flagged += 1
                    break
        print(f"{name}: near-dupes vs train (5-gram Jaccard>0.8): {flagged}/{len(split)}")


if __name__ == '__main__':
    main()


