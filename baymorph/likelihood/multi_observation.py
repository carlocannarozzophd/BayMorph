"""Joint likelihood for conditionally independent astronomical observations."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from baymorph.likelihood.base import ImageLikelihood


class MultiObservationLikelihood:
    """Combine one configured image likelihood per observation.

    Notes
    -----
    This factorization assumes observations are conditionally independent
    given the complete forward model and their observation-specific noise
    models. Shared inferential parameters do not violate this assumption.
    """

    def __init__(
        self,
        likelihoods: Mapping[str, ImageLikelihood],
    ) -> None:
        if not likelihoods:
            raise ValueError(
                "MultiObservationLikelihood requires at least one observation."
            )

        normalized = {}

        for observation, likelihood in likelihoods.items():
            if (
                not isinstance(observation, str)
                or not observation
            ):
                raise TypeError(
                    "Observation names must be non-empty strings."
                )

            if not isinstance(
                likelihood,
                ImageLikelihood,
            ):
                raise TypeError(
                    "All values must be ImageLikelihood instances."
                )

            normalized[
                observation
            ] = likelihood

        self._likelihoods = normalized

    @property
    def observations(self) -> tuple[str, ...]:
        """Return observations in deterministic evaluation order."""
        return tuple(
            self._likelihoods
        )

    @property
    def image_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return the model-image shape required by each observation."""
        return {
            observation: likelihood.image_shape
            for observation, likelihood in self._likelihoods.items()
        }

    @property
    def requires_source(self) -> bool:
        """Whether some observation needs the source part (AD-047)."""
        return any(
            likelihood.requires_source
            for likelihood in self._likelihoods.values()
        )

    def __call__(
        self,
        model_images: Mapping[str, np.ndarray],
        source_images: Mapping[str, np.ndarray] | None = None,
    ) -> float:
        """Evaluate and sum all observation-specific likelihood factors.

        ``source_images`` holds the source part of each model, without its
        residual background; observations whose likelihood depends on it
        (AD-047) require it.
        """
        supplied = set(
            model_images
        )
        expected = set(
            self._likelihoods
        )

        if supplied != expected:
            raise ValueError(
                "Model images must define exactly the configured "
                "observations."
            )

        if source_images is not None and set(source_images) != expected:
            raise ValueError(
                "Source images must define exactly the configured "
                "observations."
            )

        total = 0.0

        for observation, likelihood in self._likelihoods.items():
            value = float(
                likelihood(
                    model_images[
                        observation
                    ],
                    source=(
                        None
                        if source_images is None
                        else source_images[observation]
                    ),
                )
            )

            if not np.isfinite(
                value
            ):
                return -np.inf

            total += value

        return float(
            total
        )
