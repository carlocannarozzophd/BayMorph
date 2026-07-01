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
    include_sky=False,
):
    """Render inner, outer, and total model images from one posterior sample.

    Parameters
    ----------
    values : array-like
        One posterior sample with eleven parameters ordered as
        ``flux_bulge, re_bulge, n_bulge, flux_disk, re_disk, n_disk,
        q, theta, x0, y0, sky``.
    shape : tuple of int
        Output image shape as ``(ny, nx)``.
    psf : array-like, optional
        Point-spread function used to convolve each Sérsic component.
    include_sky : bool, default=False
        If True, add the fitted sky level to the total image.

    Returns
    -------
    inner : ndarray
        Rendered inner Sérsic component.
    outer : ndarray
        Rendered outer Sérsic component.
    total : ndarray
        Sum of the rendered components, optionally including sky.

    Raises
    ------
    ValueError
        If ``values`` does not contain exactly eleven parameters.
    """
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

    if include_sky:
        total = total + sky

    return inner, outer, total


def compute_posterior_profile_bands(
    posterior_samples,
    shape,
    radii,
    *,
    mask=None,
    psf=None,
    percentiles=(16.0, 50.0, 84.0),
    include_sky=False,
    profile_center=None,
):
    """Compute posterior radial-profile bands for a two-component Sérsic model.

    Parameters
    ----------
    posterior_samples : array-like
        Posterior samples with shape ``(n_samples, 11)``. Parameters must be
        ordered as ``flux_bulge, re_bulge, n_bulge, flux_disk, re_disk,
        n_disk, q, theta, x0, y0, sky``.
    shape : tuple of int
        Shape of the rendered model images as ``(ny, nx)``.
    radii : array-like
        Radial bin edges used to compute radial profiles.
    mask : array-like of bool, optional
        Boolean mask passed to ``radial_profile_from_image``.
    psf : array-like, optional
        Point-spread function used to convolve each rendered component.
    percentiles : sequence of float, default=(16.0, 50.0, 84.0)
        Three percentiles used to summarize the posterior profiles.
    include_sky : bool, default=False
        If True, include the fitted sky level in the total model profile.
    profile_center : tuple of float, optional
        Fixed profile center ``(center_x, center_y)``. If None, the center
        is taken from each posterior sample.

    Returns
    -------
    dict
        Dictionary containing the input bin edges, bin centers, and percentile
        bands for the ``inner``, ``outer``, and ``total`` components. Each
        component stores ``p16``, ``p50``, and ``p84`` arrays.

    Raises
    ------
    ValueError
        If the posterior samples, radii, shape, or percentiles are invalid.
    """
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
            include_sky=include_sky,
        )

        if profile_center is None:
            center_x = values[8]
            center_y = values[9]
        else:
            center_x, center_y = profile_center

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

    radius = 0.5 * (radii[:-1] + radii[1:])

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
    """Compute percentile bands from a stack of radial profiles.

    Parameters
    ----------
    profiles : array-like
        Collection of radial profiles with shape ``(n_samples, n_bins)``.
    percentiles : array-like
        Three percentiles to compute along the sample axis.

    Returns
    -------
    dict
        Dictionary with ``p16``, ``p50``, and ``p84`` percentile arrays.
    """
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
    """Validate posterior samples for two-component Sérsic profile bands."""
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
    """Validate an image shape tuple."""
    if len(shape) != 2:
        raise ValueError("shape must be a 2-element tuple.")


def _validate_radii(radii):
    """Validate radial-bin edges."""
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