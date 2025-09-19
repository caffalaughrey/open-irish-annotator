#!/usr/bin/env python
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from collections import defaultdict


SRC_URL = "https://raw.githubusercontent.com/michmech/lemmatization-lists/master/lemmatization-ga.txt"


def download_txt(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(SRC_URL) as r:  # nosec: B310 (trusted source)
        data = r.read()
    dest.write_bytes(data)


def build_json(txt_path: Path, out_json: Path) -> None:
    # File format: lemma\tform per line (UTF-8)
    form_to_lemmas: dict[str, set[str]] = defaultdict(set)
    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            lemma, form = line.split("\t", 1)
            if not form:
                continue
            form_to_lemmas[form].add(lemma)

    # Pick a single lemma per form: prefer the shortest (often base form)
    form_to_lemma: dict[str, str] = {}
    for form, lemmas in form_to_lemmas.items():
        best = min(lemmas, key=len)
        form_to_lemma[form] = best

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(form_to_lemma, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    out_dir = Path("data/external")
    txt_path = out_dir / "lemmatization-ga.txt"
    json_path = out_dir / "mechura_lemma.json"

    print("Downloading Mechura lemma list…")
    download_txt(txt_path)
    print(f"Saved: {txt_path}")
    print("Building JSON lexicon…")
    build_json(txt_path, json_path)
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()


