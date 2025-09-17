.PHONY: help setup download splits build-tagset train eval export fmt lint rust-build rust-test

help:
	@echo "Targets: setup, download, build-tagset, train, eval, export, fmt, lint, rust-build, rust-test"

setup:
	python -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -e .[dev]

download:
	bash scripts/download_ud_irish.sh

splits:
	python scripts/build_splits.py

build-tagset:
	python scripts/build_splits.py
	python scripts/build_tagset.py

train:
	python -m src.gaeilge_morph.training.train || echo "Training stub not implemented yet"

eval:
	python -m src.gaeilge_morph.eval.evaluate || echo "Eval stub not implemented yet"

export:
	python -m src.gaeilge_morph.export.export_onnx || echo "Export stub not implemented yet"

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


