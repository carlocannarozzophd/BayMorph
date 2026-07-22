from __future__ import annotations

from baymorph.models.definition import ModelDefinition

_MODEL_REGISTRY: dict[str, ModelDefinition] = {}


def register_model(model: ModelDefinition) -> None:
    """Register a morphological model definition.

    Parameters
    ----------
    model : ModelDefinition
        Model definition to register.

    Raises
    ------
    ValueError
        If another model with the same name is already registered.
    """
    if model.name in _MODEL_REGISTRY:
        raise ValueError(f"Model '{model.name}' already registered.")

    _MODEL_REGISTRY[model.name] = model


def get_model(name: str) -> ModelDefinition:
    """Return a registered model definition.

    Parameters
    ----------
    name : str
        Name of the registered model.

    Returns
    -------
    ModelDefinition
        Registered model definition.

    Raises
    ------
    ValueError
        If the requested model is not registered.
    """
    try:
        return _MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(registered_models())
        raise ValueError(
            f"Unknown model '{name}'. Available models: {available}"
        ) from exc


def registered_models() -> list[str]:
    """Return the names of all registered models.

    Returns
    -------
    list of str
        Registered model names sorted alphabetically.
    """
    return sorted(_MODEL_REGISTRY)