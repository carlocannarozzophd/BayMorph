"""Composition of deterministic observation effects."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from baymorph.observation.protocols import ObservationEffect

FloatArray = NDArray[np.float64]


class ObservationModel:
    """Ordered composition of deterministic observation effects.

    Parameters
    ----------
    effects : iterable of ObservationEffect, optional
        Effects applied sequentially to the expected image.

    Notes
    -----
    ``ObservationModel`` contains deterministic transformations only.
    Stochastic noise belongs to the likelihood or simulation layer.
    """

    def __init__(
        self,
        effects: Iterable[ObservationEffect] = (),
    ) -> None:
        self.effects = tuple(effects)

        for effect in self.effects:
            if not isinstance(effect, ObservationEffect):
                raise TypeError(
                    "ObservationModel effects must implement apply()."
                )

    def observe(
        self,
        image: ArrayLike,
    ) -> FloatArray:
        """Apply all observation effects in their declared order."""
        observed = np.asarray(
            image,
            dtype=float,
        )

        if observed.ndim != 2:
            raise ValueError(
                "ObservationModel input image must be 2D."
            )

        if not np.all(np.isfinite(observed)):
            raise ValueError(
                "ObservationModel input image must contain only "
                "finite values."
            )

        observed = observed.copy()

        for effect in self.effects:
            observed = np.asarray(
                effect.apply(observed),
                dtype=float,
            )

            if observed.ndim != 2:
                raise ValueError(
                    "Observation effects must return 2D images."
                )

            if not np.all(np.isfinite(observed)):
                raise ValueError(
                    "Observation effects must return only finite values."
                )

        return observed
