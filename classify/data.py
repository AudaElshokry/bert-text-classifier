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
    - Hugging Face tokenizer compatibility
    - Type safety and validation

    Args:
        texts: List of text samples
        labels: List of corresponding integer labels
        tokenizer: Hugging Face tokenizer instance
        max_len: Maximum sequence length for tokenization
        padding: Padding strategy (True, 'longest', 'max_length', etc.)
        truncation: Whether to truncate sequences
    """

    def __init__(self, texts: List[str], labels: List[int], tokenizer,
                 max_len: int = 256, padding: bool = True, truncation: bool = True):
        # Validation
        if len(texts) != len(labels):
            raise ValueError(f"Texts and labels must have same length. "
                             f"Got {len(texts)} texts and {len(labels)} labels.")
        if not texts:
            raise ValueError("Texts list cannot be empty.")

        self.texts = list(map(str, texts))
        self.labels = list(map(int, labels))
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
        return enc

    def get_stats(self) -> Dict[str, Any]:
        """Return dataset statistics for research reporting."""
        text_lengths = [len(text) for text in self.texts]
        unique_labels, counts = np.unique(self.labels, return_counts=True)

        return {
            "num_samples": len(self),
            "num_classes": len(unique_labels),
            "class_distribution": dict(zip(unique_labels.tolist(), counts.tolist())),
            "avg_text_length": float(np.mean(text_lengths)),
            "max_text_length": int(np.max(text_lengths)),
            "min_text_length": int(np.min(text_lengths)),
            "text_length_std": float(np.std(text_lengths)),
        }

    def show_samples(self, indices: Optional[List[int]] = None, n: int = 5) -> pd.DataFrame:
        """Display sample texts with labels for debugging and analysis."""
        if indices is None:
            indices = range(min(n, len(self)))

        samples = []
        for i in indices:
            sample = self[i]
            samples.append({
                "index": i,
                "text": sample["text"][:100] + "..." if len(sample["text"]) > 100 else sample["text"],
                "label": sample["label"],
                "text_length": len(sample["text"])
            })

        return pd.DataFrame(samples)