"""Synthetic-observation builders for scientific validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from baymorph.simulation.observation import (
    SyntheticObservation,
    make_observation,
)
from baymorph.simulation.sersic import SyntheticSersic
from baymorph.validation.truth import GroundTruth


_REQUIRED_SERSIC_PARAMETERS = (
    "flux",
    "re",
    "n",
    "q",
    "theta",
    "x0",
    "y0",
)


def simulated_effective_gain(
    noise_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the effective-gain declaration of a simulation (AD-047).

    The simulations count photo-electrons with a gain of one, so their
    Poisson noise has the source variance of a declared gain of one; without
    Poisson noise the source variance is absent.
    """
    if bool(noise_metadata.get("poisson_noise", True)):
        return {
            "source": "declared",
            "gain": 1.0,
        }

    return {
        "source": "absent",
    }


def build_synthetic_observation(
    truth: GroundTruth,
) -> SyntheticObservation:
    """Build a synthetic observation from a ground-truth description.

    The initial implementation supports single-component Sérsic simulations.
    ``theta`` must be expressed in radians, consistently with
    :class:`baymorph.simulation.sersic.SyntheticSersic`.

    Parameters
    ----------
    truth : GroundTruth
        Ground-truth description of the simulated target.

    Returns
    -------
    SyntheticObservation
        Complete synthetic observation and its intermediate products.

    Raises
    ------
    ValueError
        If the requested model is unsupported or required parameters are
        missing.
    """
    if truth.model_name not in {"sersic", "single_sersic"}:
        raise ValueError(
            "Synthetic-observation building currently supports only "
            "'sersic' and 'single_sersic' models."
        )

    missing = [
        name
        for name in _REQUIRED_SERSIC_PARAMETERS
        if name not in truth.parameters
    ]

    if missing:
        names = ", ".join(missing)
        raise ValueError(
            f"Missing ground-truth Sérsic parameters: {names}."
        )

    galaxy = SyntheticSersic(
        flux=truth.parameter("flux"),
        re=truth.parameter("re"),
        n=truth.parameter("n"),
        q=truth.parameter("q"),
        theta=truth.parameter("theta"),
        x0=truth.parameter("x0"),
        y0=truth.parameter("y0"),
    )

    psf_fwhm = truth.psf_metadata.get("fwhm")
    if psf_fwhm is not None:
        psf_fwhm = float(psf_fwhm)

    poisson_noise = truth.noise_metadata.get(
        "poisson_noise",
        True,
    )
    read_noise_sigma = truth.noise_metadata.get(
        "read_noise_sigma",
    )

    if read_noise_sigma is not None:
        read_noise_sigma = float(read_noise_sigma)

    return make_observation(
        galaxy=galaxy,
        shape=truth.image_shape,
        sky=0.0 if truth.sky_level is None else truth.sky_level,
        psf_fwhm=psf_fwhm,
        poisson_noise=bool(poisson_noise),
        read_noise_sigma=read_noise_sigma,
        seed=truth.random_seed,
    )