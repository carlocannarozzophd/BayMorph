from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelDefinition:
    """Definition of a morphological model registered in BayMorph.

    Parameters
    ----------
    name : str
        Unique model identifier.
    parameter_names : list of str
        Ordered names of the model parameters.
    parameter_labels : list of str
        Human-readable labels corresponding to ``parameter_names``.
    factory_name : str
        Name of the factory class responsible for constructing the model.
    factory : callable, optional
        Factory callable used to instantiate the model. This may be ``None``
        when the model definition is registered before the corresponding
        factory is available.
    """

    name: str
    parameter_names: list[str]
    parameter_labels: list[str]
    factory_name: str
    factory: Callable[..., Any] | None = None