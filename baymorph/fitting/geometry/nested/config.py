"""Configuration for native geometry-aware nested sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NestedSamplerConfig:
    """Immutable configuration for one nested-sampling run.

    Parameters
    ----------
    nlive : int
        Number of active live points.
    dlogz : float
        Evidence-based termination tolerance.
    max_iterations : int or None
        Maximum number of active nested-sampling iterations.
    max_likelihood_calls : int or None
        Maximum number of likelihood evaluations.
    seed : int or None
        Seed used to initialize the sampler random-number generator.
    """

    nlive: int = 500
    dlogz: float = 0.01
    max_iterations: int | None = None
    max_likelihood_calls: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if isinstance(self.nlive, bool) or not isinstance(
            self.nlive,
            int,
        ):
            raise TypeError(
                "nlive must be an integer."
            )

        if self.nlive <= 0:
            raise ValueError(
                "nlive must be positive."
            )

        dlogz = float(self.dlogz)

        if not np.isfinite(dlogz) or dlogz <= 0.0:
            raise ValueError(
                "dlogz must be positive and finite."
            )

        object.__setattr__(
            self,
            "dlogz",
            dlogz,
        )

        self._validate_optional_positive_integer(
            name="max_iterations",
            value=self.max_iterations,
        )
        self._validate_optional_positive_integer(
            name="max_likelihood_calls",
            value=self.max_likelihood_calls,
        )

        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(
                self.seed,
                int,
            ):
                raise TypeError(
                    "seed must be an integer or None."
                )

            if self.seed < 0:
                raise ValueError(
                    "seed must be non-negative."
                )

    @staticmethod
    def _validate_optional_positive_integer(
        *,
        name: str,
        value: int | None,
    ) -> None:
        if value is None:
            return

        if isinstance(value, bool) or not isinstance(
            value,
            int,
        ):
            raise TypeError(
                f"{name} must be an integer or None."
            )

        if value <= 0:
            raise ValueError(
                f"{name} must be positive when provided."
            )
