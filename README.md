Open Irish Morphological Analyzer (UD-based)
===========================================

This repo hosts training code for an Irish (Gaeilge) morphology model and a Rust runtime that loads a compact ONNX model for token-level analysis: UD UPOS+FEATS and lemmas. See `RESEARCH.md` for the roadmap and design choices.

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


License
-------

See `LICENSE`. UD Irish-IDT data is licensed separately; see the UD repository for details.


