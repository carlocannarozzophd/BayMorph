"""Synthetic observation generation."""

from dataclasses import dataclass

import numpy as np

from .noise import (
    add_gaussian_noise,
    add_poisson_noise,
)
from .psf import (
    GaussianPSF,
    convolve_with_psf,
)
from .sersic import SyntheticSersic


@dataclass
class SyntheticObservation:
    """
    Synthetic astronomical observation.
    """

    image: np.ndarray

    @property
    def shape(self):
        return self.image.shape


def make_observation(
    galaxy: SyntheticSersic,
    shape: tuple[int, int],
    psf_fwhm: float | None = None,
    poisson_noise: bool = True,
    read_noise_sigma: float | None = None,
    seed: int | None = None,
) -> SyntheticObservation:
    """
    Generate a synthetic observation.
    """

    image = galaxy.render(shape)

    if psf_fwhm is not None:
        image = convolve_with_psf(
            image,
            GaussianPSF(psf_fwhm),
        )

    if poisson_noise:
        image = add_poisson_noise(
            image,
            seed=seed,
        )

    if read_noise_sigma is not None:
        image = add_gaussian_noise(
            image,
            sigma=read_noise_sigma,
            seed=seed,
        )

    return SyntheticObservation(
        image=image
    )