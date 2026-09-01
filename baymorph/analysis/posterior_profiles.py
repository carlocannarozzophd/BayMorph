"""Posterior radial-profile summaries for model-agnostic fit results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from baymorph.analysis.radial_profiles import elliptical_radius_map


@dataclass(frozen=True)
class ProfileBand:
    """Percentile bands for one radial profile."""

    p16: np.ndarray
    p50: np.ndarray
    p84: np.ndarray


@dataclass(frozen=True)
class PosteriorProfileResult:
    """Posterior radial-profile summaries."""

    radius: np.ndarray
    total: ProfileBand
    components: dict[str, ProfileBand]


def _as_samples_array(
    posterior,
) -> np.ndarray:
    """Return a two-dimensional equal-weight posterior sample array.

    Posterior containers exposing an equal-weight representation are
    preferred over backend-native weighted samples. Raw arrays remain
    supported and are interpreted as already equal-weight samples.
    """
    nested_posterior = getattr(
        posterior,
        "posterior",
        None,
    )

    if (
        nested_posterior is not None
        and hasattr(
            nested_posterior,
            "equal_weight_samples",
        )
    ):
        posterior = nested_posterior.equal_weight_samples

    elif hasattr(
        posterior,
        "equal_weight_samples",
    ):
        posterior = posterior.equal_weight_samples

    elif hasattr(
        posterior,
        "posterior_samples",
    ):
        posterior = posterior.posterior_samples

    elif hasattr(
        posterior,
        "samples",
    ):
        posterior = posterior.samples

    samples = np.asarray(
        posterior,
        dtype=float,
    )

    if samples.ndim != 2:
        raise ValueError(
            "posterior samples must be a two-dimensional array"
        )

    if samples.shape[0] == 0:
        raise ValueError(
            "posterior samples must not be empty"
        )

    if not np.all(
        np.isfinite(
            samples
        )
    ):
        raise ValueError(
            "posterior samples must contain only finite values"
        )

    return samples

def _select_sample_indices(
    n_samples: int,
    *,
    max_samples: int | None,
    random_state: int | np.random.Generator | None,
) -> np.ndarray:
    """Return posterior-sample indices used for profile estimation."""
    indices = np.arange(
        n_samples
    )

    if (
        max_samples is None
        or max_samples >= n_samples
    ):
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
    """Return 16th, 50th, and 84th percentile profile bands."""
    p16, p50, p84 = np.nanpercentile(
        profile_stack,
        [
            16.0,
            50.0,
            84.0,
        ],
        axis=0,
    )

    return ProfileBand(
        p16=p16,
        p50=p50,
        p84=p84,
    )


def _prepare_annulus_indices(
    image_shape: tuple[int, int],
    radii: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    mask: np.ndarray | None,
    axis_ratio: float,
    position_angle: float,
) -> tuple[np.ndarray, ...]:
    """Precompute flattened pixel indices for fixed elliptical annuli."""
    radii = np.asarray(
        radii,
        dtype=float,
    )

    if radii.ndim != 1:
        raise ValueError(
            "radii must be 1D."
        )

    if radii.size < 2:
        raise ValueError(
            "radii must contain at least two edges."
        )

    if not np.all(
        np.isfinite(
            radii
        )
    ):
        raise ValueError(
            "radii must contain only finite values."
        )

    if np.any(
        np.diff(
            radii
        )
        <= 0.0
    ):
        raise ValueError(
            "radii must be strictly increasing."
        )

    radius_map = elliptical_radius_map(
        image_shape,
        center_x=center_x,
        center_y=center_y,
        axis_ratio=axis_ratio,
        position_angle=position_angle,
    )

    valid_geometry = np.ones(
        image_shape,
        dtype=bool,
    )

    if mask is not None:
        mask = np.asarray(
            mask,
            dtype=bool,
        )

        if mask.shape != image_shape:
            raise ValueError(
                "mask must have the same shape as image."
            )

        valid_geometry &= ~mask

    return tuple(
        np.flatnonzero(
            (
                valid_geometry
                & (
                    radius_map
                    >= radii[index]
                )
                & (
                    radius_map
                    < radii[index + 1]
                )
            ).ravel()
        )
        for index in range(
            radii.size - 1
        )
    )


def _radial_profile_from_annuli(
    image: np.ndarray,
    *,
    image_shape: tuple[int, int],
    annulus_indices: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Return one median profile using precomputed annulus membership."""
    image = np.asarray(
        image,
        dtype=float,
    )

    if image.shape != image_shape:
        raise ValueError(
            "rendered image shape does not match image_shape."
        )

    flat_image = image.ravel()

    profile = np.full(
        len(
            annulus_indices
        ),
        np.nan,
        dtype=float,
    )

    for index, pixel_indices in enumerate(
        annulus_indices
    ):
        values = flat_image[
            pixel_indices
        ]

        values = values[
            np.isfinite(
                values
            )
        ]

        if values.size:
            profile[
                index
            ] = np.median(
                values
            )

    return profile


def compute_posterior_profile_bands(
    posterior,
    model_factory: Callable[[np.ndarray], object],
    image_shape: tuple[int, int],
    radii: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    axis_ratio: float = 1.0,
    position_angle: float = 0.0,
    mask: np.ndarray | None = None,
    max_samples: int | None = 500,
    random_state: int | np.random.Generator | None = 0,
) -> PosteriorProfileResult:
    """Compute posterior percentile bands on fixed elliptical annuli.

    For every selected posterior sample, galaxy components are rendered
    individually and summed explicitly. Additive sky terms provided by model
    wrappers are therefore excluded.

    All posterior samples are measured on the same reference annuli defined
    by ``center_x``, ``center_y``, ``axis_ratio``, and ``position_angle``.
    This keeps the radial coordinate physically comparable between samples.

    Parameters
    ----------
    posterior : object or array-like
        Posterior samples or a posterior container exposing ``samples``.
    model_factory : callable
        Callable accepting one posterior sample and returning a model exposing
        ``render_components``.
    image_shape : tuple of int
        Output image shape as ``(ny, nx)``.
    radii : ndarray
        Semi-major-axis radial-bin edges in pixels.
    center_x, center_y : float
        Detector-space centre of the reference annuli.
    axis_ratio : float, default=1
        Minor-to-major axis ratio of the fixed reference annuli.
    position_angle : float, default=0
        Counter-clockwise major-axis angle of the fixed reference annuli from
        the positive detector x-axis, in radians.
    mask : ndarray of bool, optional
        Pixels excluded from profile measurements.
    max_samples : int, optional
        Maximum number of posterior samples used for profile estimation.
    random_state : int or numpy.random.Generator, optional
        Random-number state used for reproducible posterior subsampling.

    Returns
    -------
    PosteriorProfileResult
        Posterior percentile bands for the total galaxy profile and each
        named component.
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

    annulus_indices = _prepare_annulus_indices(
        image_shape,
        radii,
        center_x=center_x,
        center_y=center_y,
        mask=mask,
        axis_ratio=axis_ratio,
        position_angle=position_angle,
    )

    total_profiles = []
    component_profiles: dict[
        str,
        list[np.ndarray],
    ] = {}

    component_names: tuple[
        str,
        ...,
    ] | None = None

    for sample_index in sample_indices:
        model = model_factory(
            samples[
                sample_index
            ]
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

        sample_component_profiles = {}

        for name, image in component_images.items():
            profile = _radial_profile_from_annuli(
                image,
                image_shape=image_shape,
                annulus_indices=annulus_indices,
            )

            component_profiles[
                name
            ].append(
                profile
            )

            sample_component_profiles[
                name
            ] = profile

        if len(
            component_images
        ) == 1:
            total_profiles.append(
                next(
                    iter(
                        sample_component_profiles.values()
                    )
                )
            )
        else:
            total_image = np.zeros(
                image_shape,
                dtype=float,
            )

            for image in component_images.values():
                total_image += image

            total_profiles.append(
                _radial_profile_from_annuli(
                    total_image,
                    image_shape=image_shape,
                    annulus_indices=annulus_indices,
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
