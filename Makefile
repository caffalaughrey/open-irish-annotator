.PHONY: help setup download splits build-tagset train eval export parity fmt lint rust-build rust-test release
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
	. .venv/bin/activate && ruff --version && mypy --version || true

download:
	bash scripts/download_ud_irish.sh

splits:
	$(PY) scripts/build_splits.py

build-tagset:
	$(PY) scripts/build_splits.py
	$(PY) scripts/build_tagset.py

train:
	$(PY) -m gaeilge_morph.training.train || echo "Training stub not implemented yet"
pilot:
	$(PY) -m gaeilge_morph.training.train --epochs 5 --batch-size 64 --lr 0.002 --device cpu --max-chars 32 --max-lemma 32 --tag-loss-weight 1.0 --lemma-loss-weight 1.0 || true

resume:
	$(PY) -m gaeilge_morph.training.train --epochs 20 --batch-size 64 --lr 0.002 --device cpu --resume-path artifacts/checkpoints/best.pt || true

eval:
	$(PY) -m gaeilge_morph.eval.evaluate || echo "Eval stub not implemented yet"

export:
	$(PY) -m gaeilge_morph.export.export_onnx || echo "Export stub not implemented yet"

parity:
	$(PY) scripts/onnx_parity.py

teacher:
	$(PY) scripts/build_teacher_stanza.py

fmt:
	ruff check --select I --fix . || true
	black . || true
	cargo fmt --manifest-path examples/rust/morphology_runtime/Cargo.toml || true

lint:
	ruff check . || true
	mypy src || true
	cargo clippy --manifest-path examples/rust/morphology_runtime/Cargo.toml --features inference -- -D warnings || true

rust-build:
	cargo build --manifest-path examples/rust/morphology_runtime/Cargo.toml

rust-test:
	cargo test --manifest-path examples/rust/morphology_runtime/Cargo.toml

release:
	bash scripts/package_release.sh $(VERSION)


# Docker helpers
.PHONY: docker-build docker-shell
docker-build:
	docker build -t open-irish-annotator .

docker-shell:
	docker run -it --rm -v "$(PWD)":/workspace -w /workspace open-irish-annotator bash


