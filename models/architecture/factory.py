from models.architecture.meanflow_model import (
    MeanFlowModel,
    MeanFlowModelv2,
    MeanFlowModelAdaLNZero,
    ToySO3MeanFlowModel,
    ToySO3MeanFlowModelv2,
)

MODEL_REGISTRY = {
    "MeanFlowModel": MeanFlowModel,
    "MeanFlowModelv2": MeanFlowModelv2,
    "AdaLN": MeanFlowModelv2,
    "MeanFlowModelAdaLNZero": MeanFlowModelAdaLNZero,
    "ToySO3MeanFlowModel": ToySO3MeanFlowModel,
    "ToySO3MeanFlowModelv2": ToySO3MeanFlowModelv2,
}

LEGACY_VERSION_MAP = {
    "v1": "MeanFlowModel",
    "v2": "MeanFlowModelv2",
}


def create_meanflow_model(model_conf):
    """
    Factory function to create MeanFlow model instances.

    Supports two modes:
    1. New mode: model_conf.class_name specifies the model class name
    2. Legacy mode: model_conf.version specifies v1/v2 (backward compatible)

    Args:
        model_conf: Configuration object with either 'class_name' or 'version' field

    Returns:
        Instance of the specified model class

    Raises:
        ValueError: If neither class_name nor version is specified, or if class_name is invalid
    """
    # Check for new class_name-based specification
    if hasattr(model_conf, "class_name") and model_conf.class_name is not None:
        class_name = model_conf.class_name
        if class_name not in MODEL_REGISTRY:
            available = ", ".join(MODEL_REGISTRY.keys())
            raise ValueError(
                f"Unknown model class '{class_name}'. "
                f"Available classes: {available}"
            )
        model_class = MODEL_REGISTRY[class_name]
        return model_class(model_conf)

    # Legacy version-based specification (backward compatible)
    version = model_conf.get("version", "v1")
    if version in LEGACY_VERSION_MAP:
        class_name = LEGACY_VERSION_MAP[version]
        model_class = MODEL_REGISTRY[class_name]
        return model_class(model_conf)
    else:
        # If version is specified but not in legacy map, try direct lookup
        if version in MODEL_REGISTRY:
            model_class = MODEL_REGISTRY[version]
            return model_class(model_conf)
        else:
            raise ValueError(
                f"Unknown model version '{version}'. "
                f"Available versions: {', '.join(LEGACY_VERSION_MAP.keys())}, "
                f"or use class_name with one of: {', '.join(MODEL_REGISTRY.keys())}"
            )
