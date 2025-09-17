from __future__ import annotations

from gaeilge_morph.training.runner import run_training, TrainConfig


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-chars", type=int, default=32)
    parser.add_argument("--max-lemma", type=int, default=32)
    parser.add_argument("--tag-loss-weight", type=float, default=1.0)
    parser.add_argument("--lemma-loss-weight", type=float, default=1.0)
    args = parser.parse_args()

    cfg = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
        max_chars=args.max_chars,
        max_lemma=args.max_lemma,
        tag_loss_weight=args.tag_loss_weight,
        lemma_loss_weight=args.lemma_loss_weight,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()


