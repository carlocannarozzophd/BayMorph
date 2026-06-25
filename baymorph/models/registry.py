from __future__ import annotations

from .definition import ModelDefinition

_MODEL_REGISTRY: dict[str, ModelDefinition] = {}


def register_model(
    model: ModelDefinition,
) -> None:
    if model.name in _MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model.name}' already registered."
        )

    _MODEL_REGISTRY[model.name] = model


def get_model(
    name: str,
) -> ModelDefinition:
    try:
        return _MODEL_REGISTRY[name]

    except KeyError as exc:
        raise ValueError(
            f"Unknown model '{name}'."
        ) from exc


def registered_models():
    return list(
        _MODEL_REGISTRY.keys()
    )