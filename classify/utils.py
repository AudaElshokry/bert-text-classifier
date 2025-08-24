# utils.py
from typing import Tuple, Dict, List, Optional, Any
import random, os, json
import numpy as np, pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight


def seed_everything(seed: int = 42, deterministic: bool = False):
    """
    Set all random seeds for complete reproducibility across runs.

    Args:
        seed: Random seed value (default: 42)
        deterministic: Whether to enable CUDA determinism (may impact performance)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Set all seeds to: {seed}")


def read_csv_required(path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Read CSV file with required 'text' and 'label' columns.

    Args:
        path: Path to the CSV file
        verbose: Whether to print dataset statistics

    Returns:
        pandas.DataFrame with text and label columns
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    required_columns = {"text", "label"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"CSV missing required columns: {missing}")

    if verbose:
        print(f"📊 Loaded {len(df)} samples from {path}")
        if len(df) > 0:
            print(f"   Label distribution:")
            label_counts = df['label'].value_counts()
            for label, count in label_counts.items():
                print(f"     {label}: {count} ({count / len(df) * 100:.1f}%)")

            text_lengths = df['text'].str.len()
            print(f"   Text length - Avg: {text_lengths.mean():.1f}, "
                  f"Max: {text_lengths.max()}, 95th %ile: {text_lengths.quantile(0.95):.1f}")

    return df


def build_label_maps(train_df: pd.DataFrame, verbose: bool = True) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build label mappings from training data.

    Args:
        train_df: Training DataFrame with 'label' column
        verbose: Whether to print label statistics

    Returns:
        Tuple of (label2id, id2label) mappings
    """
    if pd.api.types.is_string_dtype(train_df["label"]):
        labels = sorted(train_df["label"].unique().tolist())
        label2id = {lbl: i for i, lbl in enumerate(labels)}
    else:
        labels = sorted(map(int, train_df["label"].unique().tolist()))
        label2id = {str(lbl): int(lbl) for lbl in labels}

    id2label = {v: k for k, v in label2id.items()}

    if verbose and len(train_df) > 0:
        print(f"🏷️  Label mapping created:")
        print(f"   Number of classes: {len(label2id)}")
        print(f"   Classes: {list(label2id.keys())}")

    return label2id, id2label


def apply_label_map(df: pd.DataFrame, label2id: Dict[str, int]) -> pd.DataFrame:
    """Maps df['label'] to numeric ids using label2id. Works for string/num labels."""
    df = df.copy()
    if pd.api.types.is_string_dtype(df["label"]):
        df["label"] = df["label"].map(label2id).astype(int)
    else:
        df["label"] = df["label"].astype(int)
    return df


def save_label_map(out_dir: str, label2id: Dict[str, int], id2label: Dict[int, str]):
    """Save label mappings to JSON files."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "id2label.json"), "w", encoding="utf-8") as f:
        json.dump({int(k): v for k, v in id2label.items()}, f, ensure_ascii=False, indent=2)


def calculate_class_weights(train_df: pd.DataFrame, method: str = 'balanced') -> List[float]:
    """
    Calculate class weights for imbalanced datasets.

    Args:
        train_df: Training DataFrame
        method: 'balanced' for sklearn-style weights or 'inverse' for inverse frequency

    Returns:
        List of class weights corresponding to class indices
    """
    if pd.api.types.is_string_dtype(train_df["label"]):
        labels = train_df["label"].tolist()
    else:
        labels = train_df["label"].astype(int).tolist()

    if method == 'balanced':
        classes = np.unique(labels)
        weights = compute_class_weight('balanced', classes=classes, y=labels)
        return weights.tolist()

    elif method == 'inverse':
        class_counts = train_df['label'].value_counts().sort_index()
        total_samples = len(train_df)
        weights = total_samples / (len(class_counts) * class_counts)
        return weights.tolist()

    else:
        raise ValueError(f"Unknown method: {method}. Use 'balanced' or 'inverse'.")


def validate_dataset_splits(train_path: str, val_path: str, test_path: str) -> Dict[str, Any]:
    """
    Validate and compare dataset splits for research integrity.
    """
    results = {'has_overlap': False, 'overlap_count': 0}

    try:
        splits = {
            'train': read_csv_required(train_path, verbose=False),
            'val': read_csv_required(val_path, verbose=False),
            'test': read_csv_required(test_path, verbose=False)
        }

        all_texts = []
        for df in splits.values():
            all_texts.extend(df['text'].tolist())

        total_unique = len(set(all_texts))
        total_samples = len(all_texts)

        results['has_overlap'] = total_unique != total_samples
        results['overlap_count'] = total_samples - total_unique
        results['overlap_percentage'] = (results['overlap_count'] / total_samples * 100) if total_samples > 0 else 0

    except Exception as e:
        results['error'] = str(e)

    return results


def save_experiment_config(config: Dict[str, Any], output_dir: str):
    """Save experiment configuration for reproducibility."""
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "experiment_config.json")

    serializable_config = {}
    for k, v in config.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            serializable_config[k] = v
        else:
            serializable_config[k] = str(v)

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_config, f, indent=2, ensure_ascii=False)

    print(f"📝 Experiment config saved to: {config_path}")