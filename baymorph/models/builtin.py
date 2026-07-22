from __future__ import annotations

from baymorph.models.definition import ModelDefinition
from baymorph.models.labels import parameter_labels, parameter_names
from baymorph.models.registry import register_model

_BUILTIN_FACTORY_NAMES = {
    "single_sersic": "SingleSersicModelFactory",
    "double_sersic": "DoubleSersicModelFactory",
}


def register_builtin_models() -> None:
    """Register the built-in BayMorph model definitions.

    This function populates the global model registry with the built-in
    morphological models shipped with BayMorph. Each registered model stores
    its parameter names, display labels, and the name of the factory class
    responsible for constructing the corresponding model.

    Notes
    -----
    This function is executed automatically when
    :mod:`baymorph.models.builtin` is imported. Users normally do not need to
    call it explicitly.
    """
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