from __future__ import annotations

import numpy as np

from baymorph.models import SersicModel
from baymorph.simulation.psf import convolve_with_psf

from .radial_profiles import radial_profile_from_image


def _render_components_from_values(
    values,
    shape,
    *,
    psf=None,
):
    values = np.asarray(values, dtype=float)

    if values.size != 11:
        raise ValueError(
            "Expected 11 parameters: "
            "flux_bulge, re_bulge, n_bulge, "
            "flux_disk, re_disk, n_disk, "
            "q, theta, x0, y0, sky."
        )

    (
        flux_bulge,
        re_bulge,
        n_bulge,
        flux_disk,
        re_disk,
        n_disk,
        q,
        theta,
        x0,
        y0,
        sky,
    ) = values

    inner = SersicModel(
        flux=flux_bulge,
        re=re_bulge,
        n=n_bulge,
        q=q,
        theta=theta,
        x0=x0,
        y0=y0,
    ).render(shape=shape)

    outer = SersicModel(
        flux=flux_disk,
        re=re_disk,
        n=n_disk,
        q=q,
        theta=theta,
        x0=x0,
        y0=y0,
    ).render(shape=shape)

    if psf is not None:
        inner = convolve_with_psf(inner, psf=psf)
        outer = convolve_with_psf(outer, psf=psf)

    total = inner + outer

    # ``sky`` is part of the fitted posterior vector, but posterior
    # morphology profiles are source-light profiles.  We therefore validate
    # and unpack it, but do not add it to inner/outer/total.
    _ = sky

    return inner, outer, total


def compute_posterior_profile_bands(
    posterior_samples,
    shape,
    radii,
    *,
    mask=None,
    psf=None,
    percentiles=(16.0, 50.0, 84.0),
):
    posterior_samples = _validate_posterior_samples(posterior_samples)
    radii = _validate_radii(radii)
    _validate_shape(shape)

    percentiles = np.asarray(percentiles, dtype=float)

    if percentiles.shape != (3,):
        raise ValueError("percentiles must contain exactly three values.")

    profiles = {
        "inner": [],
        "outer": [],
        "total": [],
    }

    for values in posterior_samples:
        inner, outer, total = _render_components_from_values(
            values=values,
            shape=shape,
            psf=psf,
        )

        center_x = values[8]
        center_y = values[9]

        images = {
            "inner": inner,
            "outer": outer,
            "total": total,
        }

        for component, image in images.items():
            profiles[component].append(
                radial_profile_from_image(
                    image,
                    center_x=center_x,
                    center_y=center_y,
                    radii=radii,
                    mask=mask,
                )
            )

    radius = 0.5 * (
        radii[:-1]
        + radii[1:]
    )

    return {
        "radii": radii,
        "radius": radius,
        **{
            component: _profile_percentile_bands(
                component_profiles,
                percentiles=percentiles,
            )
            for component, component_profiles in profiles.items()
        },
    }


def _profile_percentile_bands(
    profiles,
    *,
    percentiles,
):
    profiles = np.asarray(profiles, dtype=float)

    p16, p50, p84 = np.nanpercentile(
        profiles,
        percentiles,
        axis=0,
    )

    return {
        "p16": p16,
        "p50": p50,
        "p84": p84,
    }


def _validate_posterior_samples(posterior_samples):
    posterior_samples = np.asarray(posterior_samples, dtype=float)

    if posterior_samples.ndim != 2:
        raise ValueError("posterior_samples must be a 2D array.")

    if posterior_samples.shape[1] != 11:
        raise ValueError(
            "posterior_samples must have shape (n_samples, 11)."
        )

    if posterior_samples.shape[0] < 1:
        raise ValueError("posterior_samples must contain at least one sample.")

    return posterior_samples


def _validate_shape(shape):
    if len(shape) != 2:
        raise ValueError("shape must be a 2-element tuple.")


def _validate_radii(radii):
    radii = np.asarray(radii, dtype=float)

    if radii.ndim != 1:
        raise ValueError("radii must be 1D.")

    if radii.size < 2:
        raise ValueError("radii must contain at least two edges.")

    if not np.all(np.isfinite(radii)):
        raise ValueError("radii must contain only finite values.")

    if np.any(np.diff(radii) <= 0):
        raise ValueError("radii must be strictly increasing.")

    return radii