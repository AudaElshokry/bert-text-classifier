# classify/train.py
import argparse, os, json
import types
import torch
from transformers import AutoTokenizer

from classify.data import TextDataset
from classify.model import build_model
from classify.trainer import BertTrainer, TrainArgs
from classify.utils import (
    seed_everything,
    read_csv_required,
    build_label_maps,
    apply_label_map,
    save_label_map,
)

def build_argparser():
    ap = argparse.ArgumentParser(description="Train a BERT text classifier on CSV (text,label)")
    ap.add_argument("--train_path", required=True)
    ap.add_argument("--val_path", required=True)
    ap.add_argument("--test_path", required=True)
    ap.add_argument("--output_path", default="output")
    ap.add_argument("--bert_model", default="bert-base-multilingual-cased")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_epochs", type=int, default=5)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.0)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    ap.add_argument("--gpus", type=int, nargs="*", default=[0], help="Visible GPU IDs, e.g., --gpus 0 1")
    return ap

def _coerce_args(args):
    """
    Accept:
      - None  -> parse from CLI
      - dict  -> convert to argparse.Namespace
      - Namespace -> use as-is
    """
    if args is None:
        return build_argparser().parse_args()
    if isinstance(args, dict):
        return argparse.Namespace(**args)
    if isinstance(args, argparse.Namespace):
        return args
    raise TypeError("args must be None, dict, or argparse.Namespace")

def main(args=None):
    # ---- parse args (flexible) ----
    args = _coerce_args(args)

    # ---- GPU visibility (if any) ----
    if torch.cuda.is_available() and getattr(args, "gpus", None):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in args.gpus)

    # ---- housekeeping ----
    seed_everything(args.seed)
    os.makedirs(args.output_path, exist_ok=True)

    # ---- load data ----
    train_df = read_csv_required(args.train_path)
    val_df   = read_csv_required(args.val_path)
    test_df  = read_csv_required(args.test_path)

    # build/apply label maps (from TRAIN only)
    label2id, id2label = build_label_maps(train_df)
    train_df = apply_label_map(train_df, label2id)
    val_df   = apply_label_map(val_df, label2id)
    test_df  = apply_label_map(test_df, label2id)
    save_label_map(args.output_path, label2id, id2label)

    num_labels = len(id2label)

    # ---- tokenizer & datasets ----
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    train_ds = TextDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_len=args.max_len)
    val_ds   = TextDataset(val_df["text"].tolist(),   val_df["label"].tolist(),   tokenizer, max_len=args.max_len)
    test_ds  = TextDataset(test_df["text"].tolist(),  test_df["label"].tolist(),  tokenizer, max_len=args.max_len)

    # ---- model ----
    model = build_model(
        args.bert_model,
        num_labels=num_labels,
        id2label={i: id2label[i] for i in range(num_labels)},
        label2id=label2id,
    )

    # ---- trainer ----
    targs = TrainArgs(
        lr=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.max_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_len=args.max_len,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_workers=args.num_workers,
        grad_clip=args.grad_clip,
        patience=args.patience,
        fp16=getattr(args, "fp16", False),
    )
    trainer = BertTrainer(
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        args=targs,
        label_names=[id2label[i] for i in range(num_labels)],
    )

    # ---- train (with early stopping) ----
    trainer.train(output_dir=args.output_path)

    # ---- evaluate on test & save artifacts ----
    test_metrics, trues, preds = trainer.eval(test_ds)
    with open(os.path.join(args.output_path, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)
    trainer.save_predictions(test_ds, os.path.join(args.output_path, "preds_test.csv"))
    trainer.write_report(trues, preds, label_names=[id2label[i] for i in range(num_labels)], out_dir=args.output_path)

    print("[test]", test_metrics)

if __name__ == "__main__":
    main()
