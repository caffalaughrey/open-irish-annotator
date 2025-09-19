# Release <VERSION>

## Artifacts

- `artifacts/onnx/model.onnx`
- `data/processed/tagset.json`
- `data/processed/word_vocab.json`
- `data/processed/char_vocab.json`
- `data/processed/lemma_lexicon.json` (optional)

## Usage

Rust CLI (CPU):

```bash
cargo run --manifest-path examples/rust/morphology_runtime/Cargo.toml --features inference --bin analyze -- "Is" "maidin" "bhreá" "í"
```

Python:

```bash
python examples/python/analyze.py --model artifacts/onnx/model.onnx --resources data/processed Is maidin bhreá í
```

Node:

```bash
node examples/nodejs/analyze.mjs --model artifacts/onnx/model.onnx --resources data/processed Is maidin bhreá í
```

## Changes

- <short bullets>

## Notes

- Built from commit: <SHA>
- Python/Node examples support `--prefer-lexicon` for stable lemmas.

