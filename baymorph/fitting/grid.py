"""Simple grid-search fitting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from baymorph.likelihood import ImageLikelihood


@dataclass(frozen=True)
class GridSearchResult:
    """Result of a one-dimensional grid search.

    Parameters
    ----------
    values : ndarray
        Grid of sampled parameter values.
    loglikes : ndarray
        Log-likelihood evaluated at each parameter value.
    """

    values: np.ndarray
    loglikes: np.ndarray

    def __post_init__(self) -> None:
        """Validate and normalize stored grid-search arrays."""
        values = np.asarray(
            self.values,
            dtype=float,
        )
        loglikes = np.asarray(
            self.loglikes,
            dtype=float,
        )

        if values.ndim != 1:
            raise ValueError(
                "values must be a one-dimensional array."
            )

        if loglikes.ndim != 1:
            raise ValueError(
                "loglikes must be a one-dimensional array."
            )

        if values.shape != loglikes.shape:
            raise ValueError(
                "values and loglikes must have the same shape."
            )

        if values.size == 0:
            raise ValueError(
                "Grid-search results require at least one sampled value."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Grid-search values must be finite."
            )

        if np.any(np.isnan(loglikes)):
            raise ValueError(
                "Grid-search log-likelihood values must not contain NaN."
            )

        values.setflags(write=False)
        loglikes.setflags(write=False)

        object.__setattr__(
            self,
            "values",
            values,
        )
        object.__setattr__(
            self,
            "loglikes",
            loglikes,
        )

    @property
    def best_index(self) -> int:
        """Index of the maximum log-likelihood."""
        return int(
            np.argmax(self.loglikes)
        )

    @property
    def best_value(self) -> float:
        """Parameter value corresponding to the maximum log-likelihood."""
        return float(
            self.values[self.best_index]
        )

    @property
    def best_loglike(self) -> float:
        """Maximum log-likelihood value."""
        return float(
            self.loglikes[self.best_index]
        )


def grid_search_1d(
    values: np.ndarray,
    model_factory,
    likelihood: ImageLikelihood,
) -> GridSearchResult:
    """Perform a one-dimensional grid search.

    Parameters
    ----------
    values : array-like
        One-dimensional grid of parameter values to evaluate.
    model_factory : callable
        Callable receiving one parameter value and returning a model object
        exposing a ``render(shape=...)`` method.
    likelihood : ImageLikelihood
        Fully configured image likelihood. The likelihood owns the observed
        image, mask, and all fixed statistical quantities.

    Returns
    -------
    GridSearchResult
        Parameter grid and corresponding log-likelihood values.

    Raises
    ------
    TypeError
        If ``likelihood`` is not an :class:`ImageLikelihood`.
    ValueError
        If ``values`` is not a finite, non-empty one-dimensional array or if
        a rendered model does not match the expected image shape.
    """
    if not isinstance(
        likelihood,
        ImageLikelihood,
    ):
        raise TypeError(
            "likelihood must be an ImageLikelihood instance."
        )

    values_array = np.asarray(
        values,
        dtype=float,
    )

    if values_array.ndim != 1:
        raise ValueError(
            "values must be a one-dimensional array."
        )

    if values_array.size == 0:
        raise ValueError(
            "values must contain at least one grid point."
        )

    if not np.all(np.isfinite(values_array)):
        raise ValueError(
            "values must contain only finite values."
        )

    loglikes = np.empty(
        values_array.shape,
        dtype=float,
    )

    for index, value in enumerate(values_array):
        model = model_factory(
            float(value)
        )

        model_image = np.asarray(
            model.render(
                shape=likelihood.image_shape,
            ),
            dtype=float,
        )

        if model_image.shape != likelihood.image_shape:
            raise ValueError(
                "Rendered model must have the same shape as the likelihood "
                "data."
            )

        loglikes[index] = likelihood(
            model_image
        )

    return GridSearchResult(
        values=values_array,
        loglikes=loglikes,
    )