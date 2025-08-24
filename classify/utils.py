# classify/utils.py
from typing import Tuple, Dict, List, Optional
import random, os, json
import numpy as np, pandas as pd
import torch


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_csv_required(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert {"text", "label"}.issubset(df.columns), "CSV must have columns: text,label"
    return df


def build_label_maps(train_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Builds label2id from TRAIN set; returns (label2id, id2label)."""
    # Support both string and numeric labels: if numeric, just normalize to int
    if pd.api.types.is_string_dtype(train_df["label"]):
        labels = sorted(train_df["label"].unique().tolist())
        label2id = {lbl: i for i, lbl in enumerate(labels)}
    else:
        labels = sorted(map(int, train_df["label"].unique().tolist()))
        label2id = {str(lbl): int(lbl) for lbl in labels}  # identity mapping for ints
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label


def apply_label_map(df: pd.DataFrame, label2id: Dict[str, int]) -> pd.DataFrame:
    """Maps df['label'] to numeric ids using label2id. Works for string/num labels."""
    if pd.api.types.is_string_dtype(df["label"]):
        df["label"] = df["label"].map(label2id).astype(int)
    else:
        # Already numeric → ensure int dtype
        df["label"] = df["label"].astype(int)
    return df


def save_label_map(out_dir: str, label2id: Dict[str, int], id2label: Dict[int, str]):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "id2label.json"), "w", encoding="utf-8") as f:
        json.dump({int(k): v for k, v in id2label.items()}, f, ensure_ascii=False, indent=2)
