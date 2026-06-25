from __future__ import annotations

from .definition import ModelDefinition
from .labels import parameter_labels, parameter_names
from .registry import register_model


def register_builtin_models() -> None:
    """Register BayMorph built-in model definitions."""

    for model_name in (
        "single_sersic",
        "double_sersic",
    ):
        register_model(
            ModelDefinition(
                name=model_name,
                parameter_names=parameter_names(model_name),
                parameter_labels=parameter_labels(model_name),
                factory=None,
            )
        )


register_builtin_models()