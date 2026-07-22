"""Deterministic parameter grids for validation simulations."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Any


@dataclass(frozen=True)
class SimulationCase:
    """One indexed point in a validation simulation grid.

    Parameters
    ----------
    index : int
        Zero-based position of the case in the deterministic grid ordering.
    parameters : mapping
        Parameter values defining the simulation.
    """

    index: int
    parameters: Mapping[str, Any]

    @property
    def identifier(self) -> str:
        """str : Stable human-readable identifier for the case."""
        return f"case-{self.index:05d}"


class SimulationGrid:
    """Cartesian product of named validation parameters.

    Parameter order follows the insertion order of the input mapping, while
    values vary in standard Cartesian-product order. The grid is immutable
    from the caller's perspective because all value sequences are copied to
    tuples during construction.

    Parameters
    ----------
    axes : mapping of str to sequence
        Named parameter axes. Every axis must contain at least one value.
    """

    def __init__(self, axes: Mapping[str, Sequence[Any]]) -> None:
        if not axes:
            raise ValueError("Simulation grid must contain at least one axis.")

        normalized: dict[str, tuple[Any, ...]] = {}
        for name, values in axes.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "Simulation grid axis names must be non-empty strings."
                )

            axis_values = tuple(values)
            if not axis_values:
                raise ValueError(f"Simulation grid axis {name!r} must not be empty.")

            normalized[name] = axis_values

        self._axes = normalized

    @property
    def axes(self) -> dict[str, tuple[Any, ...]]:
        """dict : Copy of the normalized parameter axes."""
        return dict(self._axes)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """tuple of str : Parameter names in deterministic grid order."""
        return tuple(self._axes)

    def __len__(self) -> int:
        """Return the number of cases in the Cartesian product."""
        return prod(len(values) for values in self._axes.values())

    def __iter__(self) -> Iterator[SimulationCase]:
        """Iterate over indexed simulation cases."""
        names = self.parameter_names
        for index, values in enumerate(product(*(self._axes[name] for name in names))):
            yield SimulationCase(index=index, parameters=dict(zip(names, values)))

    def case(self, index: int) -> SimulationCase:
        """Return one case by zero-based index.

        Raises
        ------
        IndexError
            If the index lies outside the grid.
        """
        if index < 0 or index >= len(self):
            raise IndexError("Simulation grid case index out of range.")

        return next(case for case in self if case.index == index)
