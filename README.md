Open Irish Morphological Analyzer (UD-based)
===========================================

This repo hosts training code for an Irish (Gaeilge) morphology model and a Rust runtime that loads a compact ONNX model for token-level analysis: UD UPOS+FEATS and lemmas.

Quickstart
----------

1) Python setup (3.11 recommended):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[dev]
```

2) Download UD Irish-IDT data and prepare splits/resources:

```bash
make download
make splits
make build-tagset
```

3) Train / Evaluate / Export ONNX:

```bash
make train
make eval
make export
make parity
```

Containers (optional)
---------------------

Build and open a devcontainer:

```bash
docker build -t open-irish-annotator .
docker run -it --rm -v "$PWD":/workspace -w /workspace open-irish-annotator bash
# inside container
make setup
```

Rust runtime
------------

The Rust crate under `rust/morphology_runtime` will load the ONNX model and associated resources and expose a simple API. For now it contains a stub implementation that compiles without ONNX; the ONNX dependency will be added once the model export stabilizes.

ONNX I/O
--------

- Input: `word_ids` [batch, tokens] int64; `char_ids` [batch, tokens, chars] int64
- Output: `tag_logits` [batch, tokens, num_tags]; `lemma_logits` [batch, tokens, lemma_len, num_chars]

- CLI examples:

```bash
# via args
cargo run --manifest-path rust/morphology_runtime/Cargo.toml --features inference --bin analyze -- "Is" "maidin" "bhreá" "í"

# via stdin (one sentence per line, space tokenized)
echo "Is maidin bhreá í" | cargo run --manifest-path rust/morphology_runtime/Cargo.toml --features inference --bin analyze --
```

Use from another Rust project
-----------------------------

Assets you need to ship with your app:

- `model.onnx`
- `tagset.json`
- `word_vocab.json`
- `char_vocab.json`
- `lemma_lexicon.json` (optional; improves lemmas)

How to depend on the runtime crate:

- Add this repo as a git submodule, then a path dependency to the crate under `rust/morphology_runtime`.

```toml
# Cargo.toml
[dependencies]
morphology_runtime = { path = "submodules/open-irish-annotator/rust/morphology_runtime", features = ["inference"] }
```

Minimal usage:

```rust
use morphology_runtime::api::MorphologyRuntime;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rt = MorphologyRuntime::new_from_resources(
        "./resources/model.onnx",
        "./resources",
    )?;
    let toks = vec!["Is".to_string(), "maidin".to_string(), "bhreá".to_string(), "í".to_string()];
    let out = rt.analyze(toks)?;
    for a in out { println!("{}\t{}\t{}", a.token, a.tag, a.lemma); }
    Ok(())
}
```

Notes:

- Enable the `inference` feature to run the ONNX model (CPU via `tract-onnx`).
- Resource directory must contain the JSONs listed above; paths are app-defined.
- If you prefer to manage the model yourself, you can call the Python exporter here to regenerate `model.onnx` when you retrain.

Release
-------

```bash
make export
make release VERSION=0.1.0
ls artifacts/releases/0.1.0/
```

Parity thresholds (CI)
----------------------

We check averaged argmax parity over randomized batches. Defaults can be tuned via:

```bash
python scripts/onnx_parity.py --tag-thresh 0.90 --lemma-thresh 0.60
```

Use the artifact from other languages
-------------------------------------

Python (ONNX Runtime):

```bash
python scripts/onnx_analyze.py Is maidin bhreá í
```

Node.js (onnxruntime-node):

```bash
node scripts/node_analyze.mjs Is maidin bhreá í
```


License
-------

See `LICENSE`. UD Irish-IDT data is licensed separately; see the UD repository for details.


