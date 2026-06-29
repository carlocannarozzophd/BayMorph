"""Synthetic observation generation."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

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
    Complete synthetic observation produced by BayMorph.

    All intermediate products are retained to support diagnostics,
    validation notebooks, and report generation.
    """

    image: np.ndarray

    intrinsic_image: np.ndarray
    psf_kernel: np.ndarray | None
    convolved_image: np.ndarray
    sky_image: np.ndarray
    noise_realization: np.ndarray

    truth: dict[str, float]

    @property
    def shape(self):
        return self.image.shape

    @property
    def has_psf(self) -> bool:
        return self.psf_kernel is not None

    @property
    def sky_level(self) -> float:
        return float(np.nanmedian(self.sky_image - self.convolved_image))

    def to_fits(
        self,
        filename: str | Path,
        overwrite: bool = True,
    ) -> None:
        """Write synthetic observation to a FITS file."""
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)

        hdu = fits.PrimaryHDU(data=self.image)
        hdu.writeto(filename, overwrite=overwrite)


def make_observation(
    galaxy: SyntheticSersic,
    shape: tuple[int, int],
    sky: float = 0.0,
    psf_fwhm: float | None = None,
    poisson_noise: bool = True,
    read_noise_sigma: float | None = None,
    seed: int | None = None,
) -> SyntheticObservation:
    """Generate a synthetic observation."""

    intrinsic_image = galaxy.render(shape)

    psf_kernel = None

    if psf_fwhm is not None:
        psf = GaussianPSF(psf_fwhm)
        psf_kernel = psf.kernel()

        convolved_image = convolve_with_psf(
            intrinsic_image,
            psf,
        )
    else:
        convolved_image = intrinsic_image.copy()

    sky_image = convolved_image + sky

    image = sky_image.copy()

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

    noise_realization = image - sky_image

    truth = {
        "flux": galaxy.flux,
        "re": galaxy.re,
        "n": galaxy.n,
        "q": galaxy.q,
        "theta": galaxy.theta,
        "x0": galaxy.x0,
        "y0": galaxy.y0,
        "sky": sky,
    }

    return SyntheticObservation(
        image=image,
        intrinsic_image=intrinsic_image,
        psf_kernel=psf_kernel,
        convolved_image=convolved_image,
        sky_image=sky_image,
        noise_realization=noise_realization,
        truth=truth,
    )


def make_observation_from_config(config) -> SyntheticObservation:
    """Generate a synthetic observation from a resolved BayMorph config."""

    sim = config["simulation"]
    truth = sim["truth"]

    galaxy = SyntheticSersic(
        flux=truth["flux"],
        re=truth["re"],
        n=truth["n"],
        q=truth["q"],
        theta=np.deg2rad(truth["theta_deg"]),
        x0=truth["x0"],
        y0=truth["y0"],
    )

    return make_observation(
        galaxy=galaxy,
        shape=tuple(sim["shape"]),
        sky=sim.get("sky", 0.0),
        psf_fwhm=sim.get("psf_fwhm"),
        poisson_noise=sim.get("poisson_noise", True),
        read_noise_sigma=sim.get("read_noise_sigma"),
        seed=sim.get("seed"),
    )