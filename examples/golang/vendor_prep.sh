#!/usr/bin/env bash
set -euo pipefail

# Prepare vendored Go binding for ONNX Runtime so Docker builds offline

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
EX_DIR="${ROOT_DIR}/examples/golang"
VENDOR_DIR="${EX_DIR}/vendor/github.com/microsoft/onnxruntime-go"

mkdir -p "$(dirname "$VENDOR_DIR")"
TMP=$(mktemp -d)
echo "Downloading microsoft/onnxruntime-go..."
curl -fsSL -o "${TMP}/ortgo.tgz" "https://github.com/microsoft/onnxruntime-go/archive/refs/heads/master.tar.gz"
tar -xzf "${TMP}/ortgo.tgz" -C "${TMP}"
rm -rf "$VENDOR_DIR"
mv "${TMP}"/onnxruntime-go-* "$VENDOR_DIR"
rm -rf "$TMP"

cat > "${EX_DIR}/go.mod" <<'EOF'
module gaeilge_morph_go_example

go 1.22

require github.com/microsoft/onnxruntime-go v0.0.0

replace github.com/microsoft/onnxruntime-go => ./vendor/github.com/microsoft/onnxruntime-go
EOF

echo "Vendored Go binding prepared under examples/golang/vendor and go.mod written."

