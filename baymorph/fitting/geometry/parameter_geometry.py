"""Per-parameter geometry metadata."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .transforms import IdentityTransform, Transform


@dataclass(frozen=True)
class ParameterGeometry:
    """Geometry metadata for one model parameter.

    Parameters
    ----------
    name : str
        Parameter name.
    transform : Transform
        Invertible transformation between physical and proposal space.
    block : str
        Name of the proposal block containing the parameter.
    period : float or None
        Physical period for periodic parameters. Non-periodic parameters
        must use ``None``.
    """

    name: str
    transform: Transform = field(default_factory=IdentityTransform)
    block: str = "global"
    period: float | None = None

    def __post_init__(self) -> None:
        """Validate the parameter-geometry definition."""
        if not self.name:
            raise ValueError("Parameter name cannot be empty.")

        if not self.block:
            raise ValueError(
                f"Proposal block for parameter {self.name!r} cannot be empty."
            )

        if not isinstance(self.transform, Transform):
            raise TypeError(
                f"Transform for parameter {self.name!r} must inherit from Transform."
            )

        if self.period is not None:
            period = float(self.period)
            if not np.isfinite(period) or period <= 0.0:
                raise ValueError(
                    f"Period for parameter {self.name!r} must be positive and finite."
                )
            object.__setattr__(self, "period", period)

    @property
    def is_periodic(self) -> bool:
        """Whether the parameter is periodic."""
        return self.period is not None

    def to_proposal_space(self, value: float) -> float:
        """Transform one physical value into proposal space."""
        return self.transform.forward(value)

    def from_proposal_space(self, value: float) -> float:
        """Transform one proposal-space value into physical space."""
        physical_value = self.transform.inverse(value)
        return self.wrap_physical(physical_value)

    def wrap_physical(self, value: float) -> float:
        """Wrap a physical value into the canonical periodic interval.

        Non-periodic parameters are returned unchanged. Periodic parameters
        use the canonical interval ``[0, period)``.
        """
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(
                f"Parameter {self.name!r} requires a finite physical value."
            )

        if self.period is None:
            return value

        return float(np.mod(value, self.period))

    def physical_difference(
        self,
        first: float,
        second: float,
    ) -> float:
        """Return the signed shortest physical-space difference.

        For non-periodic parameters this is ``first - second``. For periodic
        parameters the result lies in ``[-period / 2, period / 2)``.
        """
        first = float(first)
        second = float(second)

        if not np.isfinite(first) or not np.isfinite(second):
            raise ValueError(
                f"Parameter {self.name!r} requires finite values."
            )

        difference = first - second

        if self.period is None:
            return difference

        half_period = 0.5 * self.period
        return float(
            np.mod(difference + half_period, self.period) - half_period
        )
