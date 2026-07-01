"""Simple grid-search fitting utilities."""

from dataclasses import dataclass

import numpy as np

from baymorph.likelihood import gaussian_log_likelihood


@dataclass
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

    @property
    def best_index(self) -> int:
        """int : Index of the maximum log-likelihood."""
        return int(np.argmax(self.loglikes))

    @property
    def best_value(self) -> float:
        """float : Parameter value corresponding to the maximum log-likelihood."""
        return float(self.values[self.best_index])

    @property
    def best_loglike(self) -> float:
        """float : Maximum log-likelihood value."""
        return float(self.loglikes[self.best_index])


def grid_search_1d(
    values: np.ndarray,
    model_factory,
    data: np.ndarray,
    noise: np.ndarray,
    mask: np.ndarray | None = None,
) -> GridSearchResult:
    """Perform a one-dimensional grid search.

    Parameters
    ----------
    values : ndarray
        Grid of parameter values to evaluate.
    model_factory : callable
        Callable that receives a parameter value and returns a model object
        exposing a ``render(shape=...)`` method.
    data : ndarray
        Observed image.
    noise : ndarray
        Noise map associated with ``data``.
    mask : ndarray of bool, optional
        Boolean mask identifying pixels excluded from the likelihood
        evaluation.

    Returns
    -------
    GridSearchResult
        Parameter grid together with the corresponding Gaussian
        log-likelihood values.
    """
    values = np.asarray(values, dtype=float)

    loglikes = []

    for value in values:
        model = model_factory(value)
        model_image = model.render(shape=data.shape)

        loglike = gaussian_log_likelihood(
            data=data,
            model=model_image,
            noise=noise,
            mask=mask,
        )

        loglikes.append(loglike)

    return GridSearchResult(
        values=values,
        loglikes=np.asarray(loglikes),
    )