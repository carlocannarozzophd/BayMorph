"""Structural protocols for observation effects."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@runtime_checkable
class ObservationEffect(Protocol):
    """Structural contract for one deterministic observation effect.

    An observation effect transforms an expected image while remaining
    independent of the likelihood and inference backend.

    Examples include:

    - PSF convolution;
    - additive background;
    - detector response;
    - geometric distortion;
    - resampling.

    Notes
    -----
    This initial protocol applies to deterministic transformations of a
    rendered image. Intrinsic morphology and stochastic detector noise remain
    outside this contract.

    Implementations satisfy the protocol structurally and do not need to
    inherit from this class.
    """

    def apply(
        self,
        image: FloatArray,
    ) -> FloatArray:
        """Apply the deterministic observation effect."""
