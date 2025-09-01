# model.py
from typing import Dict, Optional, Any
from transformers import AutoModelForSequenceClassification, AutoConfig

def build_model(
    model_name: str,
    num_labels: int,
    id2label: Optional[Dict[int, str]] = None,
    label2id: Optional[Dict[str, int]] = None,
    dropout_rate: Optional[float] = None,
    freeze_layers: Optional[int] = None,
    **kwargs: Any,
):
    """
    Build a Hugging Face sequence classification model with a proper config,
    optional dropout overrides, and optional layer freezing.
    """
    # 1) Build a config with the correct label mapping
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # 2) If a dropout is provided, override config fields (works across BERT/Distil/RoBERTa)
    if dropout_rate is not None:
        dr = float(dropout_rate)
        if hasattr(config, "hidden_dropout_prob"):
            config.hidden_dropout_prob = dr
        if hasattr(config, "attention_probs_dropout_prob"):
            config.attention_probs_dropout_prob = dr
        # DistilBERT-style fields (exist only on certain configs)
        if hasattr(config, "dropout"):
            config.dropout = dr
        if hasattr(config, "attention_dropout"):
            config.attention_dropout = dr

    # 3) Instantiate model with this config
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        **kwargs
    )

    # 4) Optional layer freezing
    if freeze_layers is not None:
        if freeze_layers == -1:
            # freeze entire encoder stack
            base = getattr(model, "base_model", None)
            params = base.parameters() if base is not None else model.parameters()
            for p in params:
                p.requires_grad = False
        elif isinstance(freeze_layers, int) and freeze_layers > 0:
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
