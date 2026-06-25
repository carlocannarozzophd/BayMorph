from __future__ import annotations

from .definition import ModelDefinition


_MODEL_REGISTRY: dict[str, ModelDefinition] = {}


def register_model(model: ModelDefinition) -> None:
    """Register a model definition."""

    if model.name in _MODEL_REGISTRY:
        raise ValueError(f"Model '{model.name}' already registered.")

    _MODEL_REGISTRY[model.name] = model


def get_model(name: str) -> ModelDefinition:
    """Return a registered model definition by name."""

    try:
        return _MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(registered_models())
        raise ValueError(
            f"Unknown model '{name}'. Available models: {available}"
        ) from exc


def registered_models() -> list[str]:
    """Return registered model names."""

    return sorted(_MODEL_REGISTRY)