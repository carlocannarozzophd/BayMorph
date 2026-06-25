from __future__ import annotations

from .definition import ModelDefinition
from .labels import (
    parameter_names,
    parameter_labels,
)
from .registry import register_model


register_model(
    ModelDefinition(
        name="single_sersic",
        parameter_names=parameter_names(
            "single_sersic",
        ),
        parameter_labels=parameter_labels(
            "single_sersic",
        ),
        factory=None,
    )
)

register_model(
    ModelDefinition(
        name="double_sersic",
        parameter_names=parameter_names(
            "double_sersic",
        ),
        parameter_labels=parameter_labels(
            "double_sersic",
        ),
        factory=None,
    )
)