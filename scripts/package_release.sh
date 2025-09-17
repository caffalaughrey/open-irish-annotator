#!/usr/bin/env bash
set -euo pipefail

VERSION=${1:-"0.1.0"}
OUT_DIR="artifacts/releases/${VERSION}"
RES_DIR="examples/rust/morphology_runtime/resources"
MODEL="${RES_DIR}/model.onnx"

if [ ! -f "$MODEL" ]; then
  echo "Model not found at $MODEL. Export first (make export)." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
TARBALL="${OUT_DIR}/gaeilge-morph-${VERSION}.tar.gz"

tar -czf "$TARBALL" \
  -C "$RES_DIR" model.onnx tagset.json word_vocab.json char_vocab.json lemma_lexicon.json || \
  tar -czf "$TARBALL" -C "$RES_DIR" model.onnx tagset.json word_vocab.json char_vocab.json

shasum -a 256 "$TARBALL" > "${TARBALL}.sha256"

echo "Packaged $TARBALL"
