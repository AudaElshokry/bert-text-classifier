#data.py
from typing import List, Dict, Any
from torch.utils.data import Dataset
import torch


class TextDataset(Dataset):
    """
    Minimal dataset holding texts and numeric labels.
    Uses the provided tokenizer inside collate_fn for efficiency.
    """
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = 256):
        assert len(texts) == len(labels), "texts and labels must have the same length"
        self.texts = list(map(str, texts))
        self.labels = list(map(int, labels))
        self.tokenizer = tokenizer
        self.max_len = max_len

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
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        # Ensure correct dtypes for HF models
        # (input_ids: long, attention_mask: long/bool, labels: long)
        if enc["attention_mask"].dtype != torch.long and enc["attention_mask"].dtype != torch.bool:
            enc["attention_mask"] = enc["attention_mask"].long()

        enc["labels"] = labels
        return enc
