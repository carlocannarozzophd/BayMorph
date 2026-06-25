from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable


@dataclass(frozen=True)
class ModelDefinition:
    """
    Complete description of a BayMorph model.
    """

    name: str

    parameter_names: list[str]

    parameter_labels: list[str]

    factory: Callable