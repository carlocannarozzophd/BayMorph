"""Posterior radial-profile summaries for model-agnostic fit results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from baymorph.analysis.radial_profiles import radial_profile_from_image


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


def _as_samples_array(posterior) -> np.ndarray:
    """Return posterior samples as a two-dimensional array."""
    if hasattr(posterior, "samples"):
        samples = posterior.samples
    else:
        samples = posterior

    samples = np.asarray(samples, dtype=float)

    if samples.ndim != 2:
        raise ValueError("posterior samples must be a two-dimensional array")

    if samples.shape[0] == 0:
        raise ValueError("posterior samples must contain at least one sample")

    return samples


def _select_sample_indices(
    n_samples: int,
    *,
    max_samples: int | None,
    random_state: int | np.random.Generator | None,
) -> np.ndarray:
    """Return posterior-sample indices used for profile estimation."""
    indices = np.arange(n_samples)

    if max_samples is None or max_samples >= n_samples:
        return indices

    if max_samples <= 0:
        raise ValueError("max_samples must be positive when provided")

    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    return np.sort(
        rng.choice(
            indices,
            size=max_samples,
            replace=False,
        )
    )


def _profile_band(profile_stack: np.ndarray) -> ProfileBand:
    """Return 16th, 50th, and 84th percentile bands."""
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
    """Return radial-profile values."""
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
    """Compute posterior radial-profile percentile bands."""
    samples = _as_samples_array(posterior)
    radii = np.asarray(radii, dtype=float)

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

        total_image = np.asarray(
            model.render(image_shape),
            dtype=float,
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

        components = model.render_components(
            image_shape,
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

        for name, image in components.items():
            component_profiles[name].append(
                _radial_profile(
                    np.asarray(image, dtype=float),
                    center_x=center_x,
                    center_y=center_y,
                    radii=radii,
                    mask=mask,
                )
            )

    return PosteriorProfileResult(
        radius=radius,
        total=_profile_band(
            np.asarray(total_profiles, dtype=float)
        ),
        components={
            name: _profile_band(
                np.asarray(profiles, dtype=float)
            )
            for name, profiles in component_profiles.items()
        },
    )