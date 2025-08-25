# model.py
from typing import Dict, Optional, Any
from transformers import AutoModelForSequenceClassification, AutoConfig
import torch


def build_model(
        model_name: str,
        num_labels: int,
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        dropout_rate: Optional[float] = None,
        freeze_layers: Optional[int] = None,
        **kwargs
):
    """
    Builds a Hugging Face sequence classification model for fine-tuning.

    This function provides a standardized way to initialize transformer models
    for text classification tasks with proper label mapping for evaluation.

    Args:
        model_name: Hugging Face model identifier or path
        num_labels: Number of output classes for classification
        id2label: Mapping from label index to label name (for metrics)
        label2id: Mapping from label name to label index (for encoding)
        dropout_rate: Custom dropout rate for classifier and hidden layers
        freeze_layers: Number of initial layers to freeze (0 = none, -1 = all)
        **kwargs: Additional arguments passed to from_pretrained()

    Returns:
        A configured AutoModelForSequenceClassification instance

    Examples:
        >>> # Basic usage
        >>> model = build_model("bert-base-uncased", 2)

        >>> # With label mapping for better metrics
        >>> model = build_model("bert-base-uncased", 2, 
        ...                    {0: "negative", 1: "positive"},
        ...                    {"negative": 0, "positive": 1})

        >>> # With custom dropout and frozen layers
        >>> model = build_model("bert-base-uncased", 2, 
        ...                    dropout_rate=0.3, freeze_layers=8)
    """

    # Handle layer freezing (freeze all if -1)
    if freeze_layers == -1:
        # First load config to check number of layers
        config = AutoConfig.from_pretrained(model_name)
        freeze_layers = config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else 0

    # Configure model with custom dropout if specified
    model_config = {}
    if dropout_rate is not None:
        model_config["classifier_dropout"] = dropout_rate
        model_config["hidden_dropout_prob"] = dropout_rate
        # also set attention/dropout fields across arch types
        model_config["attention_probs_dropout_prob"] = dropout_rate
        model_config["dropout"] = dropout_rate  # DistilBERT
        model_config["attention_dropout"] = dropout_rate  # DistilBERT

    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        **{**model_config, **kwargs}
    )

    # Freeze embeddings if requested to freeze > 0 layers
    # (Optional improvement: expose a separate flag for embeddings)
    if (freeze_layers is not None) and (freeze_layers >= 0):
        _freeze_layers(model, freeze_layers)

    return model


def _freeze_layers(model, num_layers: int):
    """Freeze the first `num_layers` encoder layers if possible."""
    if num_layers is None or num_layers <= 0:
        return

    # Freeze embeddings too (often stabilizes small-data training)
    for param in model.get_input_embeddings().parameters():
        param.requires_grad = False

    if hasattr(model, 'bert') and hasattr(model.bert, 'encoder'):
        # BERT architecture
        for i, layer in enumerate(model.bert.encoder.layer):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
    elif hasattr(model, 'roberta') and hasattr(model.roberta, 'encoder'):
        # RoBERTa architecture
        for i, layer in enumerate(model.roberta.encoder.layer):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
    elif hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers'):
        # Generic transformer architecture
        print(f"Warning: Automatic layer freezing not supported for this architecture.")
        print(f"Model has {model.config.num_hidden_layers} layers, requested to freeze {num_layers}.")


def get_model_info(model) -> Dict[str, Any]:
    """Return comprehensive model information for research reporting."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    info = {
        "model_type": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "parameter_memory_mb": total_params * 4 / (1024 ** 2),  # 4 bytes per float32
    }

    # Add layer information if available
    num_layers = _count_model_layers(model)
    if num_layers > 0:
        info["num_hidden_layers"] = num_layers

    return info


def _count_model_layers(model) -> int:
    """Count the number of transformer layers in the model."""
    if hasattr(model, 'bert') and hasattr(model.bert, 'encoder'):
        return len(model.bert.encoder.layer)
    elif hasattr(model, 'roberta') and hasattr(model.roberta, 'encoder'):
        return len(model.roberta.encoder.layer)
    elif hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers'):
        return model.config.num_hidden_layers
    return -1  # Unknown architecture
