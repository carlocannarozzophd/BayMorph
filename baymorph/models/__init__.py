"""Morphological models for BayMorph."""

# Register built-in model definitions.
from baymorph.models.composite import CompositeModel
from baymorph.models.convolved import ConvolvedModel
from baymorph.models.double_sersic import DoubleSersicModel
from baymorph.models.labels import (
    parameter_labels,
    parameter_names,
)
from baymorph.models.registry import (
    get_model,
    registered_models,
)
from baymorph.models.sersic import SersicModel

from baymorph.models import builtin as _builtin  # noqa: F401

__all__ = [
    "CompositeModel",
    "ConvolvedModel",
    "DoubleSersicModel",
    "SersicModel",
    "get_model",
    "parameter_labels",
    "parameter_names",
    "registered_models",
]