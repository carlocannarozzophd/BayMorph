"""Structural protocols for intrinsic morphological models."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@runtime_checkable
class MorphologicalComponent(Protocol):
    """Protocol implemented by intrinsic morphological components.

    A morphological component represents an intrinsic contribution to the
    source surface brightness. It owns neither the PSF nor the likelihood.

    Implementations must support rendering on a requested detector coordinate
    domain and must expose their named intrinsic contributions.

    Notes
    -----
    The protocol is structural. Implementations do not need to inherit from
    this class.
    """

    def render(
        self,
        shape: tuple[int, int],
        *,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> FloatArray:
        """Render the total intrinsic component image."""

    def render_components(
        self,
        shape: tuple[int, int],
        *,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> dict[str, FloatArray]:
        """Render the named intrinsic contributions."""
