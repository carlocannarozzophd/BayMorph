"""Structural protocols for intrinsic morphological models."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@runtime_checkable
class MorphologicalComponent(Protocol):
    """Structural contract for one intrinsic morphological contribution.

    A morphological component represents one additive contribution to the
    intrinsic source surface brightness. It owns neither the PSF nor the
    likelihood.

    Implementations satisfy this protocol structurally and do not need to
    inherit from it.
    """

    def render(
        self,
        shape: tuple[int, int],
        *,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> FloatArray:
        """Render the intrinsic component on the requested domain."""


@runtime_checkable
class DecomposableMorphology(
    MorphologicalComponent,
    Protocol,
):
    """Structural contract for a model exposing named components."""

    def render_components(
        self,
        shape: tuple[int, int],
        *,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> dict[str, FloatArray]:
        """Render the named intrinsic model contributions."""
