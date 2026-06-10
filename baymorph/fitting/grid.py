"""Simple grid-search fitting utilities."""

from dataclasses import dataclass

import numpy as np

from baymorph.likelihood import gaussian_log_likelihood


@dataclass
class GridSearchResult:
    """Result of a one-dimensional grid search."""

    values: np.ndarray
    loglikes: np.ndarray

    @property
    def best_index(self) -> int:
        """Index of the best likelihood value."""
        return int(np.argmax(self.loglikes))

    @property
    def best_value(self) -> float:
        """Best-fit parameter value."""
        return float(self.values[self.best_index])

    @property
    def best_loglike(self) -> float:
        """Best log-likelihood value."""
        return float(self.loglikes[self.best_index])


def grid_search_1d(
    values: np.ndarray,
    model_factory,
    data: np.ndarray,
    noise: np.ndarray,
    mask: np.ndarray | None = None,
) -> GridSearchResult:
    """Run a one-dimensional grid search."""
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