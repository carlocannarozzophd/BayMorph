"""Joint likelihood for conditionally independent photometric bands."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from baymorph.likelihood.base import ImageLikelihood


class MultiBandLikelihood:
    """Combine one configured image likelihood per photometric band."""

    def __init__(
        self,
        likelihoods: Mapping[str, ImageLikelihood],
    ) -> None:
        if not likelihoods:
            raise ValueError(
                "MultiBandLikelihood requires at least one band."
            )

        normalized = {}

        for band, likelihood in likelihoods.items():
            if not isinstance(band, str) or not band:
                raise TypeError(
                    "Band names must be non-empty strings."
                )

            if not isinstance(likelihood, ImageLikelihood):
                raise TypeError(
                    "All values must be ImageLikelihood instances."
                )

            normalized[band] = likelihood

        self._likelihoods = normalized

    @property
    def bands(self) -> tuple[str, ...]:
        """Return bands in deterministic evaluation order."""
        return tuple(self._likelihoods)

    @property
    def image_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return the model-image shape required by each band."""
        return {
            band: likelihood.image_shape
            for band, likelihood in self._likelihoods.items()
        }

    def __call__(
        self,
        model_images: Mapping[str, np.ndarray],
    ) -> float:
        """Evaluate and sum all band-specific log-likelihood factors."""
        supplied = set(model_images)
        expected = set(self._likelihoods)

        if supplied != expected:
            raise ValueError(
                "Model images must define exactly the configured bands."
            )

        total = 0.0

        for band, likelihood in self._likelihoods.items():
            value = float(
                likelihood(
                    model_images[band]
                )
            )

            if not np.isfinite(value):
                return -np.inf

            total += value

        return float(total)
