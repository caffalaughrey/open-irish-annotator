from __future__ import annotations

from gaeilge_morph.training.runner import run_training, TrainConfig


def main() -> None:
    cfg = TrainConfig()
    run_training(cfg)


if __name__ == "__main__":
    main()


