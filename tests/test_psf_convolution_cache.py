"""The cached PSF transform must reproduce the direct convolution exactly."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import fftconvolve

from baymorph.models import SersicModel
from baymorph.simulation.psf import (
    _CONVOLUTION_CACHE_ATTRIBUTE,
    EmpiricalPSF,
    GaussianPSF,
    convolve_with_psf,
)


class _SlottedPSF:
    """One PSF-like object that cannot carry a cache attribute."""

    __slots__ = ("kernel_array",)

    def __init__(self, kernel_array: np.ndarray) -> None:
        self.kernel_array = kernel_array

    @property
    def includes_pixel_response(self) -> bool:
        """Report that this kernel includes the detector pixel response."""
        return True

    def kernel(self, size: int | None = None) -> np.ndarray:
        """Return the unit-normalized kernel."""
        del size

        return self.kernel_array / self.kernel_array.sum()


def _empirical_psf(seed: int = 20260920) -> EmpiricalPSF:
    """Build one asymmetric, strictly positive empirical kernel."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:25, 0:25]

    kernel = np.exp(
        -0.5 * ((yy - 12.3) ** 2 + (xx - 11.7) ** 2) / 2.1**2
    )
    kernel += 1e-3 * rng.random(kernel.shape)

    return EmpiricalPSF(kernel_array=kernel / kernel.sum())


def _reference(image: np.ndarray, psf) -> np.ndarray:
    """Convolve directly, as the implementation did before the cache."""
    kernel = psf.kernel()

    return np.asarray(
        fftconvolve(
            image,
            kernel / kernel.sum(),
            mode="same",
        ),
        dtype=float,
    )


ENVELOPE = [
    {"flux": 1.0e4, "re": 1.5, "n": 0.6, "q": 0.95, "theta": 0.0},
    {"flux": 5.0e5, "re": 12.0, "n": 2.5, "q": 0.70, "theta": 0.8},
    {"flux": 2.0e6, "re": 40.0, "n": 5.9, "q": 0.41, "theta": -1.2},
    {"flux": 7.5e3, "re": 3.0, "n": 4.0, "q": 0.55, "theta": 1.5},
]


@pytest.mark.parametrize("parameters", ENVELOPE)
def test_cached_convolution_is_bit_identical(parameters: dict) -> None:
    """Across the prior envelope the cached path must not differ at all."""
    psf = _empirical_psf()
    shape = (129, 129)

    model = SersicModel(
        x0=64.3,
        y0=63.8,
        **parameters,
    )
    image = np.asarray(
        model.render(shape=shape, renderer="point"),
        dtype=float,
    )

    assert np.array_equal(
        convolve_with_psf(image, psf),
        _reference(image, psf),
    )


def test_cached_convolution_matches_a_gaussian_psf() -> None:
    """The analytic PSF path is cached on the same terms."""
    psf = GaussianPSF(fwhm=3.2)
    rng = np.random.default_rng(11)
    image = rng.random((97, 103))

    assert np.array_equal(
        convolve_with_psf(image, psf),
        _reference(image, psf),
    )


def test_transform_is_reused_across_calls() -> None:
    """A fit evaluates one fixed kernel millions of times; build it once."""
    psf = _empirical_psf()
    rng = np.random.default_rng(3)
    image = rng.random((64, 64))

    convolve_with_psf(image, psf)
    cache = getattr(psf, _CONVOLUTION_CACHE_ATTRIBUTE)

    assert len(cache) == 1
    transform = next(iter(cache.values()))

    convolve_with_psf(rng.random((64, 64)), psf)

    assert next(iter(cache.values())) is transform


def test_a_new_image_shape_gets_its_own_transform() -> None:
    """The padded transform depends on the image shape, so it is keyed by it."""
    psf = _empirical_psf()
    rng = np.random.default_rng(5)

    convolve_with_psf(rng.random((64, 64)), psf)
    convolve_with_psf(rng.random((80, 72)), psf)

    assert len(getattr(psf, _CONVOLUTION_CACHE_ATTRIBUTE)) == 2


def test_the_cache_is_bounded() -> None:
    """Repeated shapes must not accumulate transforms without limit."""
    psf = _empirical_psf()
    rng = np.random.default_rng(9)

    for size in range(40, 60):
        convolve_with_psf(rng.random((size, size)), psf)

    assert len(getattr(psf, _CONVOLUTION_CACHE_ATTRIBUTE)) <= 4


def test_a_psf_without_attributes_still_convolves_identically() -> None:
    """An uncachable PSF loses the speed, never the result."""
    kernel = _empirical_psf().kernel_array
    psf = _SlottedPSF(kernel)
    rng = np.random.default_rng(13)
    image = rng.random((70, 70))

    assert np.array_equal(
        convolve_with_psf(image, psf),
        _reference(image, psf),
    )
