.PHONY: help setup download splits build-tagset train eval export parity fmt lint rust-build rust-test
PY ?= python3
export PYTHONPATH := src

# Prefer venv python if available
ifneq (,$(wildcard .venv/bin/python))
  PY := .venv/bin/python
endif

help:
	@echo "Targets: setup, download, splits, build-tagset, train, eval, export, parity, fmt, lint, rust-build, rust-test"

setup:
	$(PY) -m venv .venv && . .venv/bin/activate && $(PY) -m pip install --upgrade pip && $(PY) -m pip install -e .[dev]

download:
	bash scripts/download_ud_irish.sh

splits:
	$(PY) scripts/build_splits.py

build-tagset:
	$(PY) scripts/build_splits.py
	$(PY) scripts/build_tagset.py

train:
	$(PY) -m gaeilge_morph.training.train || echo "Training stub not implemented yet"

eval:
	$(PY) -m gaeilge_morph.eval.evaluate || echo "Eval stub not implemented yet"

export:
	$(PY) -m gaeilge_morph.export.export_onnx || echo "Export stub not implemented yet"

parity:
	$(PY) scripts/onnx_parity.py

fmt:
	ruff check --select I --fix . || true
	black . || true
	cargo fmt --manifest-path rust/morphology_runtime/Cargo.toml || true

lint:
	ruff check . || true
	mypy src || true
	cargo clippy --manifest-path rust/morphology_runtime/Cargo.toml -- -D warnings || true

rust-build:
	cargo build --manifest-path rust/morphology_runtime/Cargo.toml

rust-test:
	cargo test --manifest-path rust/morphology_runtime/Cargo.toml


# Docker helpers
.PHONY: docker-build docker-shell
docker-build:
	docker build -t open-irish-annotator .

docker-shell:
	docker run -it --rm -v "$(PWD)":/workspace -w /workspace open-irish-annotator bash


