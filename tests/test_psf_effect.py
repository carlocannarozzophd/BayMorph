"""Tests for deterministic PSF observation effects."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.observation import (
    ObservationEffect,
    ObservationModel,
    PSFEffect,
)
from baymorph.simulation.psf import (
    EmpiricalPSF,
    GaussianPSF,
    convolve_with_psf,
)


def test_psf_effect_satisfies_observation_effect_protocol() -> None:
    """PSF convolution must satisfy the structural effect contract."""
    effect = PSFEffect(
        GaussianPSF(fwhm=2.0),
    )

    assert isinstance(effect, ObservationEffect)


def test_psf_effect_matches_existing_convolution_utility() -> None:
    """The effect must reuse the established convolution semantics."""
    image = np.zeros((21, 23), dtype=float)
    image[10, 11] = 5.0

    psf = GaussianPSF(fwhm=2.5)
    effect = PSFEffect(
        psf,
        kernel_size=9,
    )

    expected = convolve_with_psf(
        image,
        psf,
        kernel_size=9,
    )
    actual = effect.apply(image)

    assert np.allclose(actual, expected)


def test_psf_effect_supports_empirical_psf() -> None:
    """Empirical kernels must use the same observation-effect API."""
    kernel = np.ones((5, 5), dtype=float)
    psf = EmpiricalPSF(kernel)

    image = np.zeros((17, 19), dtype=float)
    image[8, 9] = 1.0

    result = PSFEffect(psf).apply(image)

    assert result.shape == image.shape
    assert np.all(np.isfinite(result))


def test_psf_effect_composes_inside_observation_model() -> None:
    """PSF convolution must participate in ordered effect composition."""
    image = np.zeros((17, 17), dtype=float)
    image[8, 8] = 1.0

    model = ObservationModel(
        [
            PSFEffect(
                GaussianPSF(fwhm=2.0),
                kernel_size=7,
            )
        ]
    )

    observed = model.observe(image)

    assert observed.shape == image.shape
    assert observed[8, 8] < 1.0
    assert observed.sum() == pytest.approx(
        1.0,
        rel=1.0e-12,
        abs=1.0e-12,
    )


def test_psf_effect_rejects_negative_sidelobes() -> None:
    """A kernel with negative lobes must fail closed, not be renormalized."""
    kernel = np.ones((5, 5), dtype=float)
    kernel[0, 0] = -0.5

    with pytest.raises(ValueError, match="non-negative values"):
        PSFEffect(EmpiricalPSF(kernel))


def test_psf_effect_rejects_object_without_kernel() -> None:
    """Invalid PSF objects must fail during construction."""
    with pytest.raises(
        TypeError,
        match=r"exposing kernel",
    ):
        PSFEffect(object())


@pytest.mark.parametrize(
    "kernel_size",
    [
        0,
        -1,
        4,
        2.5,
    ],
)
def test_psf_effect_rejects_invalid_kernel_size(
    kernel_size,
) -> None:
    """Gaussian kernel sizes must remain positive odd integers."""
    with pytest.raises(
        ValueError,
        match=r"positive odd integer",
    ):
        PSFEffect(
            GaussianPSF(fwhm=2.0),
            kernel_size=kernel_size,
        )
