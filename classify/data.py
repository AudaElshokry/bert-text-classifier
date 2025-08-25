# data.py
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd


class TextDataset(Dataset):
    """
    A PyTorch Dataset for text classification with dynamic tokenization.

    Features:
    - Vectorized tokenization in collate_fn for efficiency
    - Automatic padding and truncation
    - Optional text passthrough in batches for error analysis
    - Dataset statistics helper (get_stats) for research reporting
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_len: int = 256,
        padding: bool = True,
        truncation: bool = True,
    ):
        assert len(texts) == len(labels), "Texts and labels must have the same length"
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.padding = padding
        self.truncation = truncation

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        # Lightweight item; real tokenization is done in collate_fn (vectorized)
        return {"text": self.texts[i], "label": self.labels[i]}

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [b["text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

        enc = self.tokenizer(
            texts,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_len,
            return_tensors="pt",
        )
        # Ensure correct dtypes for HF models
        if enc["attention_mask"].dtype != torch.long and enc["attention_mask"].dtype != torch.bool:
            enc["attention_mask"] = enc["attention_mask"].long()

        enc["labels"] = labels
        enc["texts"] = texts  # pass-through for analysis
        return enc

    def get_stats(self) -> Dict[str, Any]:
        """Return dataset statistics for research reporting."""
        text_lengths = [len(text) for text in self.texts]
        unique_labels, counts = np.unique(self.labels, return_counts=True)

        return {
            "num_samples": len(self),
            "num_classes": len(unique_labels),
            "label_counts": {int(lbl): int(cnt) for lbl, cnt in zip(unique_labels, counts)},
            "avg_text_length": float(np.mean(text_lengths)) if text_lengths else 0.0,
            "max_text_length": int(np.max(text_lengths)) if text_lengths else 0,
            "p95_text_length": float(np.percentile(text_lengths, 95)) if text_lengths else 0.0,
        }


def build_datasets(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: Optional[List[str]],
    val_labels: Optional[List[int]],
    test_texts: Optional[List[str]],
    test_labels: Optional[List[int]],
    tokenizer,
    max_len: int = 256,
):
    """Convenience builder returning train/val/test datasets with the same tokenizer settings."""
    train_ds = TextDataset(train_texts, train_labels, tokenizer, max_len=max_len)
    val_ds = TextDataset(val_texts, val_labels, tokenizer, max_len=max_len) if val_texts is not None else None
    test_ds = TextDataset(test_texts, test_labels, tokenizer, max_len=max_len) if test_texts is not None else None
    return train_ds, val_ds, test_ds


def read_csv_dataset(path: str) -> pd.DataFrame:
    """Read a CSV with required columns 'text' and 'label'. Useful for quick checks."""
    df = pd.read_csv(path)
    required_columns = {"text", "label"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"CSV missing required columns: {missing}")

    return df
