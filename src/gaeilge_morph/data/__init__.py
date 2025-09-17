from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Dict
from pathlib import Path
import json

from conllu import parse_incr


@dataclass
class Sentence:
    tokens: List[str]
    lemmas: List[str]
    tag_strings: List[str]  # Combined UPOS+FEATS string


def read_conllu_sentences(path: Path) -> Iterable[Sentence]:
    with path.open("r", encoding="utf-8") as f:
        for toklist in parse_incr(f):
            tokens, lemmas, tag_strings = [], [], []
            for tok in toklist:
                if tok.get("misc", {}).get("SpaceAfter") == "No":
                    pass
                form = tok.get("form") or ""
                lemma = tok.get("lemma") or ""
                upos = tok.get("upostag") or "X"
                feats = tok.get("feats") or {}
                feat_str = "|".join(f"{k}={v}" for k, v in sorted(feats.items()))
                tag = upos if not feat_str else f"{upos}|{feat_str}"
                tokens.append(form)
                lemmas.append(lemma)
                tag_strings.append(tag)
            yield Sentence(tokens=tokens, lemmas=lemmas, tag_strings=tag_strings)


def build_tagset(sentences: Iterable[Sentence]) -> Dict[str, int]:
    tag2id: Dict[str, int] = {}
    for s in sentences:
        for tag in s.tag_strings:
            if tag not in tag2id:
                tag2id[tag] = len(tag2id)
    return tag2id


def build_vocabs(sentences: Iterable[Sentence]):
    word2id: Dict[str, int] = {"<pad>": 0, "<unk>": 1}
    char2id: Dict[str, int] = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
    for s in sentences:
        for w in s.tokens:
            if w not in word2id:
                word2id[w] = len(word2id)
            for ch in w:
                if ch not in char2id:
                    char2id[ch] = len(char2id)
    return word2id, char2id


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


