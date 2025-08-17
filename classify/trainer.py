from dataclasses import dataclass
from typing import Optional, Dict, List
import os
import json
import torch
from torch.utils.data import DataLoader
#from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm.auto import tqdm


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
    patience: int = 2          # early stopping on val f1_macro
    fp16: bool = False         # enable if you want mixed precision


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
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.args = args or TrainArgs()
        self.label_names = label_names
        self.model.to(self.args.device)

    def _loader(self, ds, shuffle=False):
        return DataLoader(
            ds,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            num_workers=self.args.num_workers,
            collate_fn=ds.collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    def _optim_sched(self, total_steps):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        warmup = int(total_steps * self.args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
        return optimizer, scheduler

    def train(self, output_dir: Optional[str] = None):
        self.model.train()
        train_loader = self._loader(self.train_ds, shuffle=True)
        total_steps = max(1, len(train_loader) * self.args.epochs)
        optimizer, scheduler = self._optim_sched(total_steps)
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.fp16)

        best_f1 = -1.0
        patience_left = self.args.patience
        best_path = None

        for epoch in range(1, self.args.epochs + 1):
            self.model.train()  # <-- ADD THIS LINE
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.args.epochs}")
            running_loss = 0.0

            for batch in pbar:
                batch = {k: v.to(self.args.device) for k, v in batch.items()}

                with torch.cuda.amp.autocast(enabled=self.args.fp16):
                    out = self.model(**batch)
                    loss = out.loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                running_loss += float(loss.detach())
                pbar.set_postfix(loss=running_loss / max(1, pbar.n))

            # Validation
            val_metrics = {}
            if self.val_ds is not None:
                val_metrics, _, _ = self.eval(self.val_ds)
                tqdm.write(f"[val] {val_metrics}")

                # Early stopping on macro F1
                if val_metrics.get("f1_macro", -1.0) > best_f1:
                    best_f1 = val_metrics["f1_macro"]
                    patience_left = self.args.patience
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    best_path = os.path.join(output_dir, "best_model.pt")
                    torch.save(self.model.state_dict(), best_path)

                    # Save HF-style snapshot for easy reload/deployment
                    save_dir = os.path.join(output_dir, "best_hf")
                    self.model.save_pretrained(save_dir)
                    self.tokenizer.save_pretrained(save_dir)
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        tqdm.write("Early stopping triggered.")
                        break

        # Load best model if we saved one
        if best_path and os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path, map_location=self.args.device))

    @torch.no_grad()
    def eval(self, dataset):
        self.model.eval()
        loader = self._loader(dataset, shuffle=False)
        preds, trues = [], []
        total_loss = 0.0

        for batch in loader:
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            out = self.model(**batch)
            total_loss += float(out.loss.detach())
            yhat = out.logits.argmax(-1)
            preds.extend(yhat.cpu().tolist())
            trues.extend(batch["labels"].cpu().tolist())

        avg_loss = total_loss / max(1, len(loader))
        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy_score(trues, preds),
            "f1_macro": f1_score(trues, preds, average="macro"),
            "f1_weighted": f1_score(trues, preds, average="weighted"),
        }
        return metrics, trues, preds

    @torch.no_grad()
    def save_predictions(self, dataset, path_csv: str):
        import pandas as pd
        metrics, trues, preds = self.eval(dataset)
        df = pd.DataFrame({"label_true": trues, "label_pred": preds})
        os.makedirs(os.path.dirname(path_csv), exist_ok=True)
        df.to_csv(path_csv, index=False)
        return metrics, df

    @staticmethod
    def write_report(trues: List[int], preds: List[int], label_names: Optional[List[str]], out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        # Classification report
        report = classification_report(trues, preds, target_names=label_names, digits=4, zero_division=0)
        with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(report)

        # Confusion matrix plot
        import matplotlib.pyplot as plt
        import numpy as np
        #cm = confusion_matrix(trues, preds)
        labels_idx = list(range(len(label_names))) if label_names else None
        cm = confusion_matrix(trues, preds, labels=labels_idx)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(len(cm))); ax.set_yticks(range(len(cm)))
        if label_names:
            ax.set_xticklabels(label_names, rotation=45, ha="right")
            ax.set_yticklabels(label_names)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "confusion_matrix.png"))
        plt.close(fig)
