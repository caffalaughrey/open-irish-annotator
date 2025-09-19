from __future__ import annotations

from gaeilge_morph.training.runner import run_training, TrainConfig


def main() -> None:
    import argparse
    import os
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-lr", type=float, default=1e-2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-chars", type=int, default=32)
    parser.add_argument("--max-lemma", type=int, default=32)
    parser.add_argument("--word-emb-dim", type=int, default=128)
    parser.add_argument("--char-emb-dim", type=int, default=48)
    parser.add_argument("--char-cnn-out", type=int, default=64)
    parser.add_argument("--encoder-hidden", type=int, default=256)
    parser.add_argument("--tag-loss-weight", type=float, default=1.0)
    parser.add_argument("--lemma-loss-weight", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw","sgd"])
    parser.add_argument("--use-onecycle", action="store_true")
    parser.add_argument("--no-onecycle", dest="use_onecycle", action="store_false")
    parser.set_defaults(use_onecycle=True)
    parser.add_argument("--subset-frac", type=float, default=1.0)
    parser.add_argument("--bucket-by-len", action="store_true")
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--use-kd", action="store_true")
    parser.add_argument("--kd-tag-weight", type=float, default=0.3)
    parser.add_argument("--kd-lemma-weight", type=float, default=0.3)
    args = parser.parse_args()

    # CPU threading defaults
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    torch.set_num_interop_threads(1)

    cfg = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        max_lr=args.max_lr,
        epochs=args.epochs,
        device=args.device,
        max_chars=args.max_chars,
        max_lemma=args.max_lemma,
        tag_loss_weight=args.tag_loss_weight,
        lemma_loss_weight=args.lemma_loss_weight,
        label_smoothing=args.label_smoothing,
        optimizer=args.optimizer,
        use_onecycle=args.use_onecycle,
        subset_frac=args.subset_frac,
        bucket_by_len=args.bucket_by_len,
        resume_path=args.resume_path,
        early_stop_patience=args.early_stop_patience,
        use_kd=args.use_kd,
        kd_tag_weight=args.kd_tag_weight,
        kd_lemma_weight=args.kd_lemma_weight,
        word_emb_dim=args.word_emb_dim,
        char_emb_dim=args.char_emb_dim,
        char_cnn_out=args.char_cnn_out,
        encoder_hidden=args.encoder_hidden,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()


