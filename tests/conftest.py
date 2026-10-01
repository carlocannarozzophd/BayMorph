"""Fixtures shared by several test modules."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.rendering.effective_psf import SUBPIXEL_FACTOR
from baymorph.simulation.psf import EmpiricalPSF


def gaussian_subpixel_psf(sigma: float = 1.1, half: int = 7) -> EmpiricalPSF:
    """Return a smooth empirical PSF with kernels at every sub-pixel phase.

    The kernels sample a circular Gaussian at the pixel centres displaced by
    each phase, normalized to unit sum: a valid sub-pixel product for tests
    of placement and rendering, not a pixel-integrated PSF.
    """
    factor = SUBPIXEL_FACTOR
    centre = (factor - 1) // 2
    offsets = np.arange(2 * half + 1, dtype=float) - half
    kernels = np.empty((factor, factor, 2 * half + 1, 2 * half + 1))

    for j in range(factor):
        for i in range(factor):
            x = offsets + (i - centre) / factor
            y = offsets + (j - centre) / factor
            kernel = np.exp(
                -(x[None, :] ** 2 + y[:, None] ** 2) / (2.0 * sigma**2)
            )
            kernels[j, i] = kernel / kernel.sum()

    return EmpiricalPSF(kernels[centre, centre], subpixel_kernels=kernels)


@pytest.fixture
def subpixel_gaussian_psf() -> EmpiricalPSF:
    """Return :func:`gaussian_subpixel_psf` with its default width."""
    return gaussian_subpixel_psf()
