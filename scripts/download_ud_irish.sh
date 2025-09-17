#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="data/raw"
URL="https://github.com/UniversalDependencies/UD_Irish-IDT/archive/refs/heads/master.zip"

mkdir -p "$DEST_DIR"
TMPZIP=$(mktemp /tmp/ud_irish_zip.XXXXXX.zip)
echo "Downloading UD_Irish-IDT..."
curl -L "$URL" -o "$TMPZIP"
unzip -q -o "$TMPZIP" -d "$DEST_DIR"
rm -f "$TMPZIP"
echo "Downloaded to $DEST_DIR"


