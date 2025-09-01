# train.py
import argparse
import os
import json
import time
from pathlib import Path
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
    validate_dataset_splits,
)

# ---- BEGIN compat defaults shim ----
def _ensure_defaults(ns):
    """Guarantee our newer flags exist even if not passed."""
    defaults = dict(
        dropout_rate=None,    # float|None
        freeze_layers=None,   # int|None  (-1 means freeze all)
        gpus=None,            # str|list|None
        resume_from=None,     # str|None
        grad_accum_steps=1,   # int (legacy name)
        # ensure argparse-style defaults when args is a dict
        num_workers = 2,
        grad_clip = 1.0,
        patience = 2,
        fp16 = False,
        gradient_accumulation_steps = 1,
        eval_steps = None,
        save_steps = None,
    )
    for k, v in defaults.items():
        if not hasattr(ns, k):
            setattr(ns, k, v)
    # Mirror gradient_accumulation_steps into legacy name if present
    if hasattr(ns, "gradient_accumulation_steps") and not hasattr(ns, "grad_accum_steps"):
        setattr(ns, "grad_accum_steps", getattr(ns, "gradient_accumulation_steps"))
    return ns
# ---- END compat defaults shim ----

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

    # New(er) knobs
    ap.add_argument("--dropout_rate", type=float, default=None)
    ap.add_argument("--freeze_layers", type=int, default=None)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    ap.add_argument("--gpus", type=int, nargs="*", default=None, help="Visible GPU IDs, e.g., --gpus 0 1")

    # Research/quality-of-life features
    ap.add_argument("--class_weights", type=float, nargs="+", default=None,
                    help="Class weights for imbalance (e.g., --class_weights 1.0 2.0)")
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="Accumulate gradients before optimizer step")
    ap.add_argument("--eval_steps", type=int, default=None,
                    help="Evaluate every N optimizer steps (default: end of epoch)")
    ap.add_argument("--save_steps", type=int, default=None,
                    help="Save checkpoint every N optimizer steps")
    ap.add_argument("--resume_from", default=None,
                    help="Path to checkpoint-*/checkpoint.pt (optional)")
    return ap

def _coerce_args(args):
    """Accept None|dict|Namespace."""
    if args is None:
        return build_argparser().parse_args()
    if isinstance(args, dict):
        return argparse.Namespace(**args)
    if isinstance(args, argparse.Namespace):
        return args
    raise TypeError("args must be None, dict, or argparse.Namespace")

def main(args=None):
    # ---------------- parse args ----------------
    args = _coerce_args(args)
    args = _ensure_defaults(args)

    # ---------------- GPU visibility (set first) ----------------
    if getattr(args, "gpus", None) is not None:
        if isinstance(args.gpus, (list, tuple)):
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in args.gpus)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
        print(f"✅ Set CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")

    # Re-evaluate CUDA after visibility change
    if "CUDA_VISIBLE_DEVICES" in os.environ and torch.cuda.is_available():
        torch.cuda.empty_cache()
        available_gpus = torch.cuda.device_count()
        print(f"📊 Available GPUs after setting visibility: {available_gpus}")
        if available_gpus == 0:
            print("⚠️  Warning: No GPUs available after setting CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # ---------------- housekeeping ----------------
    seed_everything(args.seed)
    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("🔧 Setting up experiment...")
    print(f"💻 Final device setup: {torch.cuda.device_count()} GPUs available")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # ---------------- load data ----------------
    train_df = read_csv_required(args.train_path, verbose=True)
    val_df   = read_csv_required(args.val_path,   verbose=True)
    test_df  = read_csv_required(args.test_path,  verbose=True)

    # Split-integrity check
    overlap = validate_dataset_splits(args.train_path, args.val_path, args.test_path)
    if overlap.get("error"):
        raise RuntimeError(f"Split validation error: {overlap['error']}")
    if overlap.get("has_overlap"):
        raise RuntimeError(
            f"Found {overlap['overlap_count']} duplicate texts across splits "
            f"({overlap.get('overlap_percentage', 0):.2f}%). Fix splits before training."
        )

    # Label maps (build on train, apply to all)
    label2id, id2label = build_label_maps(train_df)
    train_df = apply_label_map(train_df, label2id)
    val_df   = apply_label_map(val_df,   label2id)
    test_df  = apply_label_map(test_df,  label2id)
    save_label_map(args.output_path, label2id, id2label)

    num_labels = len(id2label)
    classes_ordered = [id2label[i] for i in range(num_labels)]
    print(f"📊 Dataset: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples")
    print(f"🏷️  Classes: {num_labels} -> {classes_ordered}")

    # ---------------- tokenizer & datasets ----------------
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, use_fast=True)
    if hasattr(tokenizer, "model_max_length") and tokenizer.model_max_length and args.max_len:
        tokenizer.model_max_length = max(int(args.max_len), 8)
    train_ds = TextDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_len=args.max_len)
    val_ds   = TextDataset(val_df["text"].tolist(),   val_df["label"].tolist(),   tokenizer, max_len=args.max_len)
    test_ds  = TextDataset(test_df["text"].tolist(),  test_df["label"].tolist(),  tokenizer, max_len=args.max_len)

    # ---------------- model ----------------
    model = build_model(
        args.bert_model,
        num_labels=num_labels,
        id2label={i: id2label[i] for i in range(num_labels)},
        label2id=label2id,
        dropout_rate = getattr(args, "dropout_rate", None),
        freeze_layers=getattr(args, "freeze_layers", None),
    )
    # --- PARAM COUNTS (after freezing is applied in build_model) ---
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧮 Parameters: total={n_total:,}  trainable={n_train:,}")

    # ---------------- class weights ----------------
    class_weights = getattr(args, "class_weights", None)
    if class_weights is None:
        counts = train_df["label"].value_counts().to_dict()
        N, K = len(train_df), num_labels
        class_weights = [float(N) / (K * float(counts.get(i, 1))) for i in range(K)]
    if class_weights is not None and len(class_weights) != num_labels:
        raise ValueError(
            f"Class weights length ({len(class_weights)}) must match number of classes ({num_labels})"
        )

    # ---------------- trainer args ----------------
    device = "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"
    print(f"⚙️  Using device: {device}")

    targs = TrainArgs(
        lr=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.max_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_len=args.max_len,
        device=device,
        num_workers=args.num_workers,
        grad_clip=args.grad_clip,
        patience=args.patience,
        fp16=bool(getattr(args, "fp16", False)),
        gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", 1),
        eval_steps=getattr(args, "eval_steps", None),
        save_steps=getattr(args, "save_steps", None),
    )

    resume_from = getattr(args, "resume_from", None)
    if resume_from and not os.path.exists(resume_from):
        print(f"⚠️  Resume checkpoint not found: {resume_from}")
        resume_from = None
    setattr(targs, "resume_from", resume_from)

    trainer = BertTrainer(
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        args=targs,
        label_names=classes_ordered,
        class_weights=class_weights,
    )

    # ---------------- experiment config (base) ----------------
    experiment_config = {
        "bert_model": args.bert_model,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_len": args.max_len,
        "seed": args.seed,
        "fp16": bool(getattr(args, "fp16", False)),
        "gpus": getattr(args, "gpus", None),
        "dropout_rate": getattr(args, "dropout_rate", None),
        "freeze_layers": getattr(args, "freeze_layers", None),
        "gradient_accumulation_steps": getattr(args, "gradient_accumulation_steps", 1),
        "eval_steps": getattr(args, "eval_steps", None),
        "save_steps": getattr(args, "save_steps", None),
        "resume_from": resume_from,
        "resolved_class_weights": class_weights,
        "dataset_stats": {
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "num_classes": num_labels,
        },
        "output_path": args.output_path,
        "label2id": label2id,
        # For Params count
        "num_parameters": int(n_total),
        "num_trainable_parameters": int(n_train),
    }

    # ---------------- train + time it ----------------
    train_start = time.time()
    best_f1 = trainer.train(output_dir=args.output_path)
    # Persist best validation F1 for aggregators
    best_metrics_path = os.path.join(args.output_path, "best_metrics.json")
    with open(best_metrics_path, "w", encoding="utf-8") as f:
        json.dump({"best_val_f1": float(best_f1), "f1_macro": float(best_f1)}, f, ensure_ascii=False, indent=2)

    # Also record into experiment_config so it's in one place
    # (this assumes 'experiment_config' dict exists in scope — in your file it does)
    experiment_config["best_val_f1"] = float(best_f1)
    with open(os.path.join(args.output_path, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(experiment_config, f, ensure_ascii=False, indent=2)
        
    train_end = time.time()
    training_time_seconds = train_end - train_start
    training_time_human = time.strftime("%H:%M:%S", time.gmtime(training_time_seconds))

    # ---------------- research artifacts ----------------
    total_start = train_start
    trainer.generate_research_report(test_ds, args.output_path)
    total_end = time.time()
    total_time_seconds = total_end - total_start
    total_time_human = time.strftime("%H:%M:%S", time.gmtime(total_time_seconds))

    # Enrich config and save once
    try:
        import transformers, sys
        experiment_config["lib_versions"] = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }
    except Exception:
        pass
    experiment_config["device"] = device
    experiment_config["effective_batch_size"] = int(args.batch_size) * int(getattr(args, "gradient_accumulation_steps", 1))
    experiment_config["visible_gpus"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    experiment_config["cuda_device_count"] = torch.cuda.device_count()
    experiment_config["training_time_seconds"] = training_time_seconds
    experiment_config["training_time_human"] = training_time_human
    experiment_config["total_time_seconds"] = total_time_seconds
    experiment_config["total_time_human"] = total_time_human

    with open(os.path.join(args.output_path, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(experiment_config, f, ensure_ascii=False, indent=2)

    # Load metrics for printing
    with open(os.path.join(args.output_path, "test_metrics.json"), "r", encoding="utf-8") as f:
        test_metrics = json.load(f)

    print(f"🎯 Training completed! Best validation F1: {best_f1:.4f}")
    print(f"⏱️  Training time: {training_time_human} (≈ {training_time_seconds:.1f}s)")
    print(f"⏱️  Total time (train + report): {total_time_human} (≈ {total_time_seconds:.1f}s)")
    print(f"📊 Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"📊 Test F1 macro: {test_metrics['f1_macro']:.4f}")
    print(f"📁 Results saved to: {args.output_path}")

if __name__ == "__main__":
    main()

