#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import stanza


def combine_tag(upos: str, feats: str | None) -> str:
    if not feats:
        return upos
    parts = sorted(feats.split("|"))
    parts = [p for p in parts if p]
    return upos if not parts else f"{upos}|{'|'.join(parts)}"


def annotate_sentence(nlp, tokens: List[str]) -> tuple[List[str], List[str]]:
    # Pre-tokenized input: one sentence of tokens
    doc = nlp([tokens])
    words = doc.sentences[0].words
    teacher_tags = [combine_tag(w.upos or "X", w.feats) for w in words]
    teacher_lemmas = [w.lemma or "" for w in words]
    return teacher_tags, teacher_lemmas


def process_split(jsonl_path: Path, out_path: Path, nlp) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("r", encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            obj: Dict = json.loads(line)
            tokens: List[str] = obj["tokens"]
            t_tags, t_lemmas = annotate_sentence(nlp, tokens)
            obj["teacher_tags"] = t_tags
            obj["teacher_lemmas"] = t_lemmas
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    stanza.download("ga")  # ensure models present
    nlp = stanza.Pipeline("ga", processors="tokenize,pos,lemma", tokenize_pretokenized=True)

    processed = Path("data/processed")
    for split in ["train", "dev", "test"]:
        inp = processed / f"{split}.jsonl"
        out = processed / f"{split}.teacher.jsonl"
        if not inp.exists():
            raise SystemExit(f"Missing {inp}")
        print(f"Annotating {split} with Stanza → {out}")
        process_split(inp, out, nlp)

    print("Done. To use KD, point loader to the *.teacher.jsonl files or overwrite originals if desired.")


if __name__ == "__main__":
    main()


