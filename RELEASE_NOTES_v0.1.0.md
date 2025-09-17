# Release v0.1.0

## Artifacts

- artifacts/onnx/model.onnx
- data/processed/tagset.json
- data/processed/word_vocab.json
- data/processed/char_vocab.json
- data/processed/lemma_lexicon.json (optional)

## Usage

Rust CLI (CPU):

```bash
cargo run --manifest-path examples/rust/morphology_runtime/Cargo.toml --features inference --bin analyze -- "Is" "maidin" "bhreá" "í"
# or
echo "Is maidin bhreá í" | cargo run --manifest-path examples/rust/morphology_runtime/Cargo.toml --features inference --bin analyze --
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

- Initial ONNX export of Irish morphology model (tags + lemma logits)
- Rust runtime/CLI with inference (tract-onnx) and lexicon fallback
- Examples for Python and Node; lexicon preference flag for clean lemmas
- CI: mypy strict, clippy -D warnings; parity script with thresholds
- Release packaging: tarball with model + resources and checksum

## Notes

- Built from commit: a57debc
- Lemma decoder is early-stage; examples can prefer training lexicon for stability.
