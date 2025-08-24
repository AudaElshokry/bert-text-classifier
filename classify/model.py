#model.py
from typing import Dict, Optional
from transformers import AutoModelForSequenceClassification


def build_model(
    model_name: str,
    num_labels: int,
    id2label: Optional[Dict[int, str]] = None,
    label2id: Optional[Dict[str, int]] = None,
):
    """
    Builds a sequence classification model with num_labels and optional label maps.
    Using id2label/label2id helps nice metric reports on the Hub / logs.
    """
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
