from __future__ import annotations

from .definition import ModelDefinition
from .labels import parameter_labels, parameter_names
from .registry import register_model


_BUILTIN_FACTORY_NAMES = {
    "single_sersic": "SingleSersicModelFactory",
    "double_sersic": "DoubleSersicModelFactory",
}


def register_builtin_models() -> None:
    """Register BayMorph built-in model definitions."""

    for model_name, factory_name in _BUILTIN_FACTORY_NAMES.items():
        register_model(
            ModelDefinition(
                name=model_name,
                parameter_names=parameter_names(model_name),
                parameter_labels=parameter_labels(model_name),
                factory_name=factory_name,
                factory=None,
            )
        )


register_builtin_models()