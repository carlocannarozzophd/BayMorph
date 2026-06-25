from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelDefinition:
    """Complete description of a BayMorph model."""

    name: str
    parameter_names: list[str]
    parameter_labels: list[str]
    factory_name: str
    factory: Callable[..., Any] | None = None