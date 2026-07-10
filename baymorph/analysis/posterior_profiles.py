"""Posterior radial-profile summaries for model-agnostic fit results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from baymorph.analysis.radial_profiles import radial_profile_from_image


@dataclass(frozen=True)
class ProfileBand:
    """Percentile bands for one radial profile.

    Parameters
    ----------
    p16 : ndarray
        Sixteenth-percentile radial profile.
    p50 : ndarray
        Median radial profile.
    p84 : ndarray
        Eighty-fourth-percentile radial profile.
    """

    p16: np.ndarray
    p50: np.ndarray
    p84: np.ndarray


@dataclass(frozen=True)
class PosteriorProfileResult:
    """Posterior radial-profile summaries.

    Parameters
    ----------
    radius : ndarray
        Radial-bin centers.
    total : ProfileBand
        Posterior percentile bands for the total galaxy profile, excluding
        any additive sky component.
    components : dict of str to ProfileBand
        Posterior percentile bands for each named galaxy component.
    """

    radius: np.ndarray
    total: ProfileBand
    components: dict[str, ProfileBand]


def _as_samples_array(posterior) -> np.ndarray:
    """Return posterior samples as a two-dimensional array.

    Parameters
    ----------
    posterior : object or array-like
        Posterior container exposing a ``samples`` attribute, or a raw
        two-dimensional sample array.

    Returns
    -------
    ndarray
        Posterior samples with shape ``(n_samples, n_parameters)``.

    Raises
    ------
    ValueError
        If the samples are not two-dimensional or contain no rows.
    """
    if hasattr(posterior, "samples"):
        samples = posterior.samples
    else:
        samples = posterior

    samples = np.asarray(
        samples,
        dtype=float,
    )

    if samples.ndim != 2:
        raise ValueError(
            "posterior samples must be a two-dimensional array"
        )

    if samples.shape[0] == 0:
        raise ValueError(
            "posterior samples must contain at least one sample"
        )

    return samples


def _select_sample_indices(
    n_samples: int,
    *,
    max_samples: int | None,
    random_state: int | np.random.Generator | None,
) -> np.ndarray:
    """Return posterior-sample indices used for profile estimation.

    Parameters
    ----------
    n_samples : int
        Total number of posterior samples.
    max_samples : int, optional
        Maximum number of samples to retain. If None, use all samples.
    random_state : int or numpy.random.Generator, optional
        Random-number state used for reproducible subsampling.

    Returns
    -------
    ndarray
        Sorted integer indices of selected posterior samples.

    Raises
    ------
    ValueError
        If ``max_samples`` is non-positive.
    """
    indices = np.arange(
        n_samples
    )

    if max_samples is None or max_samples >= n_samples:
        return indices

    if max_samples <= 0:
        raise ValueError(
            "max_samples must be positive when provided"
        )

    if isinstance(
        random_state,
        np.random.Generator,
    ):
        rng = random_state
    else:
        rng = np.random.default_rng(
            random_state
        )

    return np.sort(
        rng.choice(
            indices,
            size=max_samples,
            replace=False,
        )
    )


def _profile_band(
    profile_stack: np.ndarray,
) -> ProfileBand:
    """Return 16th, 50th, and 84th percentile profile bands.

    Parameters
    ----------
    profile_stack : ndarray
        Stack of radial profiles with shape
        ``(n_samples, n_radial_bins)``.

    Returns
    -------
    ProfileBand
        Posterior percentile bands.
    """
    p16, p50, p84 = np.nanpercentile(
        profile_stack,
        [16.0, 50.0, 84.0],
        axis=0,
    )

    return ProfileBand(
        p16=p16,
        p50=p50,
        p84=p84,
    )


def _radial_profile(
    image: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    radii: np.ndarray,
    mask: np.ndarray | None,
) -> np.ndarray:
    """Return a median radial profile for one image.

    Parameters
    ----------
    image : ndarray
        Input image.
    center_x : float
        X-coordinate of the radial-profile center.
    center_y : float
        Y-coordinate of the radial-profile center.
    radii : ndarray
        Radial-bin edges.
    mask : ndarray of bool, optional
        Pixels excluded from the radial-profile calculation.

    Returns
    -------
    ndarray
        Median radial-profile values.
    """
    return radial_profile_from_image(
        image,
        center_x=center_x,
        center_y=center_y,
        radii=radii,
        mask=mask,
    )


def compute_posterior_profile_bands(
    posterior,
    model_factory: Callable[[np.ndarray], object],
    image_shape: tuple[int, int],
    radii: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    mask: np.ndarray | None = None,
    max_samples: int | None = 500,
    random_state: int | np.random.Generator | None = 0,
) -> PosteriorProfileResult:
    """Compute posterior radial-profile percentile bands.

    For each selected posterior sample, the model components are rendered
    individually. The total galaxy image is then constructed explicitly as
    the sum of those components. Any additive sky term included by a wrapper
    model is therefore excluded from both the total galaxy profile and the
    component profiles.

    Parameters
    ----------
    posterior : object or array-like
        Posterior samples or a posterior container exposing ``samples``.
    model_factory : callable
        Callable that accepts one posterior sample and returns a model object
        exposing ``render_components``.
    image_shape : tuple of int
        Output image shape as ``(ny, nx)``.
    radii : ndarray
        Radial-bin edges.
    center_x : float
        X-coordinate of the radial-profile center.
    center_y : float
        Y-coordinate of the radial-profile center.
    mask : ndarray of bool, optional
        Pixels excluded from profile measurements.
    max_samples : int, optional
        Maximum number of posterior samples used to estimate profile bands.
    random_state : int or numpy.random.Generator, optional
        Random-number state used for reproducible posterior subsampling.

    Returns
    -------
    PosteriorProfileResult
        Posterior percentile bands for the total galaxy profile and each
        named component.

    Raises
    ------
    ValueError
        If model component names change between posterior samples, or if a
        model returns no components.
    """
    samples = _as_samples_array(
        posterior
    )

    radii = np.asarray(
        radii,
        dtype=float,
    )

    sample_indices = _select_sample_indices(
        samples.shape[0],
        max_samples=max_samples,
        random_state=random_state,
    )

    radius = 0.5 * (
        radii[:-1]
        + radii[1:]
    )

    total_profiles = []
    component_profiles: dict[str, list[np.ndarray]] = {}
    component_names: tuple[str, ...] | None = None

    for sample_index in sample_indices:
        model = model_factory(
            samples[sample_index]
        )

        components = model.render_components(
            image_shape
        )

        if not components:
            raise ValueError(
                "model returned no components"
            )

        current_names = tuple(
            components
        )

        if component_names is None:
            component_names = current_names
            component_profiles = {
                name: []
                for name in component_names
            }
        elif current_names != component_names:
            raise ValueError(
                "model component names changed between posterior samples"
            )

        component_images = {
            name: np.asarray(
                image,
                dtype=float,
            )
            for name, image in components.items()
        }

        total_image = np.sum(
            np.stack(
                list(
                    component_images.values()
                ),
                axis=0,
            ),
            axis=0,
        )

        total_profiles.append(
            _radial_profile(
                total_image,
                center_x=center_x,
                center_y=center_y,
                radii=radii,
                mask=mask,
            )
        )

        for name, image in component_images.items():
            component_profiles[name].append(
                _radial_profile(
                    image,
                    center_x=center_x,
                    center_y=center_y,
                    radii=radii,
                    mask=mask,
                )
            )

    return PosteriorProfileResult(
        radius=radius,
        total=_profile_band(
            np.asarray(
                total_profiles,
                dtype=float,
            )
        ),
        components={
            name: _profile_band(
                np.asarray(
                    profiles,
                    dtype=float,
                )
            )
            for name, profiles in component_profiles.items()
        },
    )