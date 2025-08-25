# trainer.py
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import os
import time
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    precision_score, recall_score, precision_recall_fscore_support
)
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


@dataclass
class TrainArgs:
    lr: float = 2e-5
    batch_size: int = 16
    epochs: int = 3
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    max_len: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2
    grad_clip: float = 1.0
    patience: int = 2  # early stopping on val f1_macro
    fp16: bool = False  # enable if you want mixed precision
    gradient_accumulation_steps: int = 1  # for larger effective batch sizes
    eval_steps: Optional[int] = None  # evaluation frequency (by optimizer steps)
    save_steps: Optional[int] = None  # checkpoint frequency (by optimizer steps)


class BertTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_ds,
        val_ds=None,
        test_ds=None,
        args: Optional[TrainArgs] = None,
        label_names: Optional[List[str]] = None,
        class_weights: Optional[List[float]] = None
    ):
        """
        BERT Trainer for text classification with research-grade features.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.args = args or TrainArgs()
        self.label_names = label_names

        # device must be set before any tensor uses it
        self.device = self.args.device
        self.model.to(self.device)

        # Optional class-weighted loss
        self.class_weights = class_weights
        self.loss_fn: Optional[nn.Module] = None
        if class_weights is not None:
            num_labels = getattr(getattr(self.model, "config", None), "num_labels", None)
            if num_labels is not None and len(class_weights) != int(num_labels):
                raise ValueError(
                    f"class_weights length {len(class_weights)} != model.config.num_labels {num_labels}"
                )
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
            self.loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            self.loss_fn = None

        # Will store optimizer/scheduler/scaler for checkpoints
        self._optimizer = None
        self._scheduler = None
        self._scaler = None

    def _loader(self, ds, shuffle: bool = False) -> DataLoader:
        """Create a DataLoader with the dataset's own collate_fn (vectorized tokenization)."""
        pin = str(self.device).startswith("cuda")
        return DataLoader(
            ds,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            num_workers=self.args.num_workers,
            collate_fn=ds.collate_fn,
            pin_memory=pin,
        )

    def _optim_sched(self, total_steps: int):
        """AdamW + linear warmup schedule."""
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        warmup = int(total_steps * self.args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
        return optimizer, scheduler

    def train(self, output_dir: Optional[str] = None):
        """Train with early stopping, gradient accumulation, and comprehensive logging."""
        train_loader = self._loader(self.train_ds, shuffle=True)

        # Number of optimizer steps (not forward passes)
        total_steps = max(
            1, len(train_loader) * self.args.epochs // max(1, self.args.gradient_accumulation_steps)
        )
        optimizer, scheduler = self._optim_sched(total_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=self.args.fp16)

        # keep refs for checkpointing

        # ---- Resume support ----
        start_epoch = 1
        global_step = 0
        if hasattr(self.args, 'resume_from') and self.args.resume_from:
            try:
                global_step = self.load_checkpoint(self.args.resume_from)
                steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, self.args.gradient_accumulation_steps)))
                start_epoch = (global_step // steps_per_epoch) + 1
                tqdm.write(f"🔄 Resuming from epoch {start_epoch}, global_step {global_step}")
            except Exception as e:
                tqdm.write(f"⚠️ Failed to resume from checkpoint: {e}")
                global_step = 0
                start_epoch = 1
        self._optimizer, self._scheduler, self._scaler = optimizer, scheduler, scaler

        best_f1 = -1.0
        patience_left = self.args.patience
        best_path = None
        global_step = 0
        epoch_history = []

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            # Save label names if provided
            if self.label_names:
                with open(os.path.join(output_dir, "labels.json"), "w", encoding="utf-8") as f:
                    json.dump(self.label_names, f, ensure_ascii=False, indent=2)

        for epoch in range(start_epoch, self.args.epochs + 1):
            self.model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.args.epochs}")
            running_loss = 0.0
            accumulation_steps = 0
            optimizer.zero_grad(set_to_none=True)

            for batch_idx, batch in enumerate(pbar):
                # Move only tensors to device; keep strings/lists intact if present
                batch_on_device = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in batch.items()}

                with torch.amp.autocast("cuda", enabled=self.args.fp16):
                    out = self.model(**{k: v for k, v in batch_on_device.items() if k not in ("text", "texts")})

                    # Custom loss with class weights if requested
                    if self.loss_fn is not None:
                        logits = out.logits
                        labels = batch_on_device["labels"]
                        loss = self.loss_fn(logits, labels)
                    else:
                        loss = out.loss

                    # Normalize for grad accumulation
                    loss = loss / max(1, self.args.gradient_accumulation_steps)

                scaler.scale(loss).backward()
                running_loss += float(loss.detach()) * max(1, self.args.gradient_accumulation_steps)
                accumulation_steps += 1

                # optimizer step only after accumulation
                if (batch_idx + 1) % max(1, self.args.gradient_accumulation_steps) == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    global_step += 1
                    current_loss = running_loss / max(1, accumulation_steps)
                    pbar.set_postfix(loss=f"{current_loss:.4f}")

                    # Intermediate eval/checkpointing
                    if self.args.eval_steps and global_step % self.args.eval_steps == 0:
                        self._intermediate_eval(global_step, output_dir)

                    if self.args.save_steps and output_dir and global_step % self.args.save_steps == 0:
                        self._save_checkpoint(global_step, output_dir, optimizer, scheduler, scaler)

            # ---- Validation & Early Stopping ----
            if self.val_ds is not None:
                val_metrics, _, _ = self.eval(self.val_ds, return_per_class=True)
                tqdm.write(
                    f"[val] {{k: {round(v, 6) if isinstance(v, float) else v} for k, v in val_metrics.items() if k != 'per_class'}}"
                )

                # Record epoch metrics
                avg_train_loss = running_loss / max(1, len(train_loader))
                epoch_entry = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": avg_train_loss,
                    "val_loss": val_metrics.get("loss"),
                    "val_accuracy": val_metrics.get("accuracy"),
                    "val_f1_macro": val_metrics.get("f1_macro"),
                    "val_metrics": val_metrics
                }
                epoch_history.append(epoch_entry)

                if output_dir:
                    with open(os.path.join(output_dir, "epoch_history.json"), "w", encoding="utf-8") as f:
                        json.dump(epoch_history, f, ensure_ascii=False, indent=2)

                current_f1 = float(val_metrics.get("f1_macro", -1.0))
                improved = current_f1 > best_f1

                tqdm.write(f"[ES] Epoch {epoch}: f1_macro={current_f1:.4f}, best={best_f1:.4f}, patience={patience_left}")

                if improved:
                    best_f1 = current_f1
                    patience_left = self.args.patience
                    tqdm.write(f"[ES] Improved: {best_f1:.4f}; patience reset to {patience_left}")

                    if output_dir:
                        # Save torch state_dict
                        best_path = os.path.join(output_dir, "best_model.pt")
                        torch.save(self.model.state_dict(), best_path)

                        # Save HF snapshot
                        save_dir = os.path.join(output_dir, "best_hf")
                        self.model.save_pretrained(save_dir)
                        self.tokenizer.save_pretrained(save_dir)

                        with open(os.path.join(output_dir, "best_metrics.json"), "w") as f:
                            json.dump(val_metrics, f, indent=2)
                else:
                    patience_left -= 1
                    tqdm.write(f"[ES] No improvement; patience={patience_left}")
                    if patience_left <= 0:
                        tqdm.write("[ES] Early stopping triggered.")
                        break

        # Load best model if available
        if best_path and os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path, map_location=self.device))
            print(f"Loaded best model with validation f1_macro: {best_f1:.4f}")

        return best_f1

    def _intermediate_eval(self, global_step: int, output_dir: Optional[str] = None):
        """Intermediate evaluation during training for research monitoring."""
        if self.val_ds is None:
            return

        val_metrics, _, _ = self.eval(self.val_ds)

        eval_entry = {
            "global_step": global_step,
            "val_metrics": val_metrics,
            "timestamp": time.time()
        }

        if output_dir:
            history_path = os.path.join(output_dir, "training_history.json")
            history = []
            if os.path.exists(history_path):
                with open(history_path, "r") as f:
                    try:
                        history = json.load(f)
                    except Exception:
                        history = []
            history.append(eval_entry)
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)

        tqdm.write(f"[Step {global_step}] val_metrics: {val_metrics}")

    
    def _save_checkpoint(self, global_step: int, output_dir: str, optimizer: optim.Optimizer,
                         scheduler=None, scaler=None, is_best: bool = False):
        """Save training checkpoint with optional best-model flag."""
        checkpoint_dir = os.path.join(output_dir, "best_checkpoint" if is_best else f"checkpoint-{global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        to_save = {
            'global_step': global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
            'train_args': vars(self.args) if hasattr(self.args, '__dict__') else str(self.args),
            'timestamp': time.time(),
            'is_best': is_best,
            'best_metric': getattr(self, 'best_f1', None) if hasattr(self, 'best_f1') else None,
            'bad_epochs': getattr(self, 'bad_epochs', None) if hasattr(self, 'bad_epochs') else None,
        }

        torch.save(to_save, os.path.join(checkpoint_dir, "checkpoint.pt"))
        # Save tokenizer if available
        tok = getattr(self, 'tokenizer', None)
        if tok is not None:
            tok.save_pretrained(checkpoint_dir)
        print(("🏆 Best checkpoint" if is_best else "💾 Checkpoint" ) + f" saved at step {global_step} -> {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model/optimizer/scheduler/scaler states from a checkpoint file or directory."""
        if os.path.isdir(checkpoint_path):
            ckpt_file = os.path.join(checkpoint_path, 'checkpoint.pt')
        else:
            ckpt_file = checkpoint_path
        if not os.path.exists(ckpt_file):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        ckpt = torch.load(ckpt_file, map_location=self.device)

        self.model.load_state_dict(ckpt['model_state_dict'])

        if hasattr(self, '_optimizer') and ckpt.get('optimizer_state_dict') is not None:
            self._optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if hasattr(self, '_scheduler') and ckpt.get('scheduler_state_dict') is not None:
            self._scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if hasattr(self, '_scaler') and ckpt.get('scaler_state_dict') is not None:
            try:
                self._scaler.load_state_dict(ckpt['scaler_state_dict'])
            except Exception:
                pass

        self.best_f1 = ckpt.get('best_metric', getattr(self, 'best_f1', None))
        self.bad_epochs = ckpt.get('bad_epochs', getattr(self, 'bad_epochs', 0))

        global_step = ckpt.get('global_step', 0)
        tqdm.write(f"✅ Loaded checkpoint from step {global_step}")
        return global_step

    @torch.no_grad()
    def eval(self, dataset, return_per_class: bool = False) -> Tuple[Dict[str, float], List[int], List[int]]:
        """Enhanced evaluation with per-class metrics for research."""
        self.model.eval()
        loader = self._loader(dataset, shuffle=False)
        preds, trues = [], []
        probs_all = []
        total_loss = 0.0

        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch_on_device = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in batch.items()}
            out = self.model(**{k: v for k, v in batch_on_device.items() if k not in ("text", "texts")})

            # Keep loss consistent with training if class weights used
            if self.loss_fn is not None:
                loss = self.loss_fn(out.logits, batch_on_device["labels"])
            else:
                loss = out.loss

            total_loss += float(loss.detach())
            yhat = out.logits.argmax(-1)
            preds.extend(yhat.cpu().tolist())
            trues.extend(batch_on_device["labels"].cpu().tolist())

        avg_loss = total_loss / max(1, len(loader))

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy_score(trues, preds),
            "f1_macro": f1_score(trues, preds, average="macro"),
            "f1_micro": f1_score(trues, preds, average="micro"),
            "f1_weighted": f1_score(trues, preds, average="weighted"),
            "precision_macro": precision_score(trues, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(trues, preds, average="macro", zero_division=0),
        }

        # AUC (binary only)
        if len(set(trues)) == 2:
            try:
                probs_all = np.array(probs_all) if 'probs_all' in locals() else None
                if probs_all is None:
                    # recompute quickly
                    loader_auc = self._loader(dataset, shuffle=False)
                    probs_all = []
                    for batch in loader_auc:
                        batch = {k: (v.to(self.device) if hasattr(v, 'to') else v) for k, v in batch.items()}
                        out = self.model(**{k: v for k, v in batch.items() if k not in ("text","texts","labels")})
                        p = torch.softmax(out.logits, dim=-1).cpu().numpy()
                        probs_all.append(p)
                    probs_all = np.concatenate(probs_all, axis=0)
                # positive class is 1 assuming label ids 0..1
                from sklearn.metrics import roc_auc_score
                metrics["auc_roc"] = float(roc_auc_score(trues, probs_all[:, 1]))
            except Exception as e:
                metrics["auc_roc"] = f"Error: {e}"

        if return_per_class and self.label_names:
            precision, recall, f1, support = precision_recall_fscore_support(
                trues, preds, labels=range(len(self.label_names)), zero_division=0
            )
            metrics["per_class"] = {}
            for i, name in enumerate(self.label_names):
                metrics["per_class"][name] = {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i])
                }

        return metrics, trues, preds

    @torch.no_grad()
    def save_predictions(self, dataset, path_csv: str):
        """Evaluate and write a CSV with columns: label_true, label_pred."""
        metrics, trues, preds = self.eval(dataset)
        df = pd.DataFrame({"label_true": trues, "label_pred": preds})
        out_dir = os.path.dirname(path_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(path_csv, index=False)
        return metrics, df

    @torch.no_grad()
    def get_predictions_with_confidence(self, dataset) -> pd.DataFrame:
        """Get predictions with confidence scores for error analysis."""
        self.model.eval()
        loader = self._loader(dataset, shuffle=False)
        all_probs, all_preds, all_trues = [], [], []
        maybe_texts: List[str] = []

        for batch in tqdm(loader, desc="Predicting with confidence"):
            batch_on_device = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in batch.items()}
            out = self.model(**{k: v for k, v in batch_on_device.items() if k not in ("text", "texts")})
            probs = torch.softmax(out.logits, dim=-1)
            preds = out.logits.argmax(-1)

            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_trues.extend(batch_on_device["labels"].cpu().tolist())

            texts = batch.get("texts") if isinstance(batch, dict) else None
            if texts is None and "texts" in batch_on_device:
                texts = batch_on_device["texts"]
            if texts is not None:
                maybe_texts.extend(list(texts))

        results_df = pd.DataFrame({
            "true_label": all_trues,
            "predicted_label": all_preds,
            "confidence": [max(p) for p in all_probs],
            "all_probabilities": all_probs
        })

        if len(maybe_texts) == len(results_df):
            results_df["text"] = maybe_texts

        return results_df

    def analyze_errors(self, dataset, output_dir: Optional[str] = None) -> pd.DataFrame:
        """Comprehensive error analysis for research paper."""
        preds_df = self.get_predictions_with_confidence(dataset)

        # If texts weren't captured, try common dataset conventions:
        if "text" not in preds_df.columns:
            texts = None
            if hasattr(dataset, "texts"):
                texts = list(getattr(dataset, "texts"))
            elif hasattr(dataset, "get_texts") and callable(getattr(dataset, "get_texts")):
                try:
                    texts = list(dataset.get_texts())
                except Exception:
                    texts = None
            if texts is not None and len(texts) >= len(preds_df):
                preds_df["text"] = texts[:len(preds_df)]

        preds_df["is_correct"] = preds_df["true_label"] == preds_df["predicted_label"]
        errors_df = preds_df[~preds_df["is_correct"]].copy()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            errors_path = os.path.join(output_dir, "error_analysis.csv")
            errors_df.to_csv(errors_path, index=False, encoding='utf-8')

            error_summary = {
                "total_errors": int(len(errors_df)),
                "error_rate": float(len(errors_df) / max(1, len(preds_df))),
                "avg_confidence_errors": float(errors_df["confidence"].mean()) if len(errors_df) else None,
                "avg_confidence_correct": float(preds_df[preds_df["is_correct"]]["confidence"].mean()) if len(preds_df) else None,
                "low_confidence_errors_<0.7": int((errors_df["confidence"] < 0.7).sum()) if len(errors_df) else 0,
            }

            with open(os.path.join(output_dir, "error_summary.json"), "w") as f:
                json.dump(error_summary, f, indent=2)

        return errors_df

    @staticmethod
    def write_report(trues: List[int], preds: List[int], label_names: Optional[List[str]], out_dir: str):
        """Write a classification report and a confusion matrix image to out_dir."""
        os.makedirs(out_dir, exist_ok=True)

        labels_idx = list(range(len(label_names))) if label_names else None

        report = classification_report(
            trues,
            preds,
            labels=labels_idx,
            target_names=label_names,
            digits=4,
            zero_division=0
        )
        with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(report)

        cm = confusion_matrix(trues, preds, labels=labels_idx)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_title("Confusion Matrix", fontsize=14, pad=20)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_xticks(range(cm.shape[1]))
        ax.set_yticks(range(cm.shape[0]))

        if label_names:
            ax.set_xticklabels(label_names, rotation=45, ha="right")
            ax.set_yticklabels(label_names)

        for (i, j), v in np.ndenumerate(cm):
            color = "white" if v > cm.max() / 2 else "black"
            ax.text(j, i, str(v), ha="center", va="center", color=color, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Number of Samples', rotation=270, labelpad=20)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_training_history(self, output_dir: str):
        """Generate training plots (epoch-level)."""
        history_path = os.path.join(output_dir, "epoch_history.json")
        if not os.path.exists(history_path):
            return

        with open(history_path, "r") as f:
            history = json.load(f)

        if not history:
            return

        epochs = [entry["epoch"] for entry in history]
        val_acc = [entry.get("val_accuracy") for entry in history]
        val_f1 = [entry.get("val_f1_macro") for entry in history]
        val_loss = [entry.get("val_loss") for entry in history]
        train_loss = [entry.get("train_loss") for entry in history]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy vs epoch
        ax1.plot(epochs, val_acc, 'o-', linewidth=2, markersize=4, label='Validation Accuracy')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Validation Accuracy vs Epoch', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # F1 vs epoch
        ax2.plot(epochs, val_f1, 'o-', linewidth=2, markersize=4, label='Validation F1 Macro')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('F1 Macro', fontsize=12)
        ax2.set_title('Validation F1 Macro vs Epoch', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Loss vs epoch
        ax3.plot(epochs, train_loss, 'o-', linewidth=2, markersize=4, label='Training Loss')
        ax3.plot(epochs, val_loss, 'o-', linewidth=2, markersize=4, label='Validation Loss')
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Loss', fontsize=12)
        ax3.set_title('Training vs Validation Loss', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Combined metrics
        ax4.plot(epochs, val_acc, 'o-', linewidth=2, markersize=4, label='Accuracy')
        ax4.plot(epochs, val_f1, 'o-', linewidth=2, markersize=4, label='F1 Macro')
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Score', fontsize=12)
        ax4.set_title('Validation Metrics Comparison', fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "training_metrics.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def generate_research_report(self, test_ds, output_dir: str):
        """Generate comprehensive research report with all metrics and visualizations."""
        os.makedirs(output_dir, exist_ok=True)

        print("📊 Generating comprehensive research report...")

        # 1) Final evaluation on test set (with per-class metrics)
        test_metrics, test_trues, test_preds = self.eval(test_ds, return_per_class=True)

        # 2) Save test metrics
        with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)

        # 3) Save predictions
        pd.DataFrame({
            "true_label": test_trues,
            "predicted_label": test_preds
        }).to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

        # 4) Classification report & confusion matrix
        self.write_report(test_trues, test_preds, self.label_names, output_dir)

        # 5) Error analysis
        _ = self.analyze_errors(test_ds, output_dir)

        # 6) Training history plots
        self.plot_training_history(output_dir)

        # 7) Summary report
        epoch_hist_path = os.path.join(output_dir, "epoch_history.json")
        best_val_f1 = 0.0
        if os.path.exists(epoch_hist_path):
            try:
                eh = json.load(open(epoch_hist_path, "r"))
                if eh:
                    best_val_f1 = max(float(e.get("val_f1_macro") or 0.0) for e in eh)
            except Exception:
                best_val_f1 = 0.0

        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_metrics": {k: v for k, v in test_metrics.items() if k != 'per_class'},
            "best_validation_f1": best_val_f1,
            "total_parameters": int(sum(p.numel() for p in self.model.parameters())),
            "trainable_parameters": int(sum(p.numel() for p in self.model.parameters() if p.requires_grad)),
        }

        with open(os.path.join(output_dir, "research_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print("✅ Research report generated successfully!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📈 Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"🎯 Test F1 Macro: {test_metrics['f1_macro']:.4f}")