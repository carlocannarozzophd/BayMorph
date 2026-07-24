"""Geometry model for transformed proposal coordinates."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence

import numpy as np

from .blocks import ProposalBlock
from .parameter_geometry import ParameterGeometry


class GeometryModel:
    """Ordered geometry definition for a model parameter vector.

    Parameters
    ----------
    parameters : sequence of ParameterGeometry
        Ordered geometry metadata matching the physical parameter order used
        by the corresponding ``ParameterSet`` and likelihood.
    """

    def __init__(
        self,
        parameters: Sequence[ParameterGeometry],
    ) -> None:
        if not parameters:
            raise ValueError(
                "GeometryModel requires at least one parameter."
            )

        self._parameters = tuple(parameters)

        if not all(
            isinstance(parameter, ParameterGeometry)
            for parameter in self._parameters
        ):
            raise TypeError(
                "All geometry entries must be ParameterGeometry instances."
            )

        names = [
            parameter.name
            for parameter in self._parameters
        ]

        if len(set(names)) != len(names):
            raise ValueError(
                "GeometryModel parameter names must be unique."
            )

        self._index = {
            parameter.name: index
            for index, parameter in enumerate(self._parameters)
        }

        self._blocks = self._derive_blocks()

    @property
    def parameters(self) -> tuple[ParameterGeometry, ...]:
        """Ordered parameter-geometry definitions."""
        return self._parameters

    @property
    def names(self) -> list[str]:
        """Ordered parameter names."""
        return [
            parameter.name
            for parameter in self._parameters
        ]

    @property
    def ndim(self) -> int:
        """Number of model parameters."""
        return len(self._parameters)

    @property
    def blocks(self) -> dict[str, tuple[str, ...]]:
        """Proposal blocks derived from parameter metadata."""
        return dict(self._blocks)

    @property
    def periodic_indices(self) -> tuple[int, ...]:
        """Indices of periodic parameters."""
        return tuple(
            index
            for index, parameter in enumerate(self._parameters)
            if parameter.is_periodic
        )

    @property
    def periodic(self) -> list[int]:
        """Indices of periodic parameters."""
        return list(self.periodic_indices)

    @property
    def proposal_blocks(self) -> tuple[ProposalBlock, ...]:
        """Proposal blocks as structured metadata."""
        return tuple(
            ProposalBlock(
                name=block_name,
                parameters=parameter_names,
            )
            for block_name, parameter_names in self._blocks.items()
        )

    def __len__(self) -> int:
        return self.ndim

    def __iter__(self) -> Iterable[ParameterGeometry]:
        return iter(self._parameters)

    def __getitem__(
        self,
        key: int | str,
    ) -> ParameterGeometry:
        if isinstance(key, str):
            try:
                return self._parameters[self._index[key]]
            except KeyError as error:
                raise KeyError(
                    f"Unknown geometry parameter {key!r}."
                ) from error

        return self._parameters[key]

    def geometry(
        self,
        name: str,
    ) -> ParameterGeometry:
        """Return geometry metadata for one named parameter."""
        return self[name]

    def to_proposal_space(
        self,
        values: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Transform a physical parameter vector into proposal space."""
        physical = self._validate_vector(
            values,
            space_name="physical",
        )

        return np.asarray(
            [
                parameter.to_proposal_space(value)
                for parameter, value in zip(
                    self._parameters,
                    physical,
                    strict=True,
                )
            ],
            dtype=float,
        )

    def from_proposal_space(
        self,
        values: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Transform a proposal-space vector into physical coordinates."""
        proposal = self._validate_vector(
            values,
            space_name="proposal",
        )

        return np.asarray(
            [
                parameter.from_proposal_space(value)
                for parameter, value in zip(
                    self._parameters,
                    proposal,
                    strict=True,
                )
            ],
            dtype=float,
        )

    def transform_forward(
        self,
        values: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Backward-compatible alias for ``to_proposal_space``."""
        return self.to_proposal_space(values)

    def transform_inverse(
        self,
        values: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Backward-compatible alias for ``from_proposal_space``."""
        return self.from_proposal_space(values)

    def wrap_physical(
        self,
        values: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Wrap periodic physical coordinates into canonical intervals."""
        physical = self._validate_vector(
            values,
            space_name="physical",
        )

        return np.asarray(
            [
                parameter.wrap_physical(value)
                for parameter, value in zip(
                    self._parameters,
                    physical,
                    strict=True,
                )
            ],
            dtype=float,
        )

    def validate_parameter_names(
        self,
        names: Sequence[str],
    ) -> None:
        """Validate external parameter names against geometry ordering."""
        external_names = list(names)

        if external_names != self.names:
            raise ValueError(
                "Geometry and ParameterSet parameter ordering must match: "
                f"geometry={self.names!r}, "
                f"parameter_set={external_names!r}."
            )

    def block_indices(
        self,
        block_name: str,
    ) -> list[int]:
        """Return ordered parameter indices belonging to one block."""
        try:
            parameter_names = self._blocks[block_name]
        except KeyError as error:
            raise KeyError(
                f"Unknown proposal block {block_name!r}."
            ) from error

        return [
            self._index[name]
            for name in parameter_names
        ]

    def _derive_blocks(
        self,
    ) -> OrderedDict[str, tuple[str, ...]]:
        grouped: OrderedDict[str, list[str]] = OrderedDict()

        for parameter in self._parameters:
            grouped.setdefault(
                parameter.block,
                [],
            ).append(parameter.name)

        return OrderedDict(
            (
                block_name,
                tuple(parameter_names),
            )
            for block_name, parameter_names in grouped.items()
        )

    def _validate_vector(
        self,
        values: Sequence[float] | np.ndarray,
        *,
        space_name: str,
    ) -> np.ndarray:
        array = np.asarray(values, dtype=float)

        if array.ndim != 1:
            raise ValueError(
                f"{space_name.capitalize()} values must be one-dimensional."
            )

        if array.shape != (self.ndim,):
            raise ValueError(
                f"Expected {self.ndim} {space_name} values, "
                f"received {array.size}."
            )

        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"{space_name.capitalize()} values must be finite."
            )

        return array
