"""Live-point containers for geometry-aware nested sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from ..geometry_model import GeometryModel


@dataclass(frozen=True)
class LivePoint:
    """One live point in physical parameter space.

    Parameters
    ----------
    values : ndarray
        One-dimensional physical parameter vector.
    logl : float
        Log-likelihood value associated with the point.
    """

    values: np.ndarray
    logl: float

    def __post_init__(self) -> None:
        """Validate and freeze the stored values."""
        values = np.asarray(self.values, dtype=float)

        if values.ndim != 1:
            raise ValueError(
                "LivePoint values must be one-dimensional."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "LivePoint values must be finite."
            )

        logl = float(self.logl)

        if np.isnan(logl):
            raise ValueError(
                "LivePoint log-likelihood must not be NaN."
            )

        immutable_values = values.copy()
        immutable_values.setflags(write=False)

        object.__setattr__(self, "values", immutable_values)
        object.__setattr__(self, "logl", logl)


class LivePoints:
    """Fixed-dimensional collection of live points."""

    def __init__(
        self,
        geometry: GeometryModel,
    ) -> None:
        if not isinstance(geometry, GeometryModel):
            raise TypeError(
                "geometry must be a GeometryModel instance."
            )

        self._geometry = geometry
        self._points: list[LivePoint] = []

    @property
    def geometry(self) -> GeometryModel:
        """Geometry associated with the live-point population."""
        return self._geometry

    @property
    def ndim(self) -> int:
        """Parameter dimensionality."""
        return self._geometry.ndim

    @property
    def size(self) -> int:
        """Number of live points."""
        return len(self._points)

    @property
    def values(self) -> np.ndarray:
        """Physical live-point matrix with shape ``(nlive, ndim)``."""
        if not self._points:
            return np.empty((0, self.ndim), dtype=float)

        return np.stack(
            [point.values for point in self._points],
            axis=0,
        )

    @property
    def proposal_values(self) -> np.ndarray:
        """Live-point matrix transformed into proposal space."""
        if not self._points:
            return np.empty((0, self.ndim), dtype=float)

        return np.stack(
            [
                self._geometry.to_proposal_space(point.values)
                for point in self._points
            ],
            axis=0,
        )

    @property
    def logl(self) -> np.ndarray:
        """Log-likelihood values for all live points."""
        return np.asarray(
            [point.logl for point in self._points],
            dtype=float,
        )

    @property
    def worst_index(self) -> int:
        """Index of the live point with the lowest log-likelihood."""
        self._require_nonempty()
        return int(np.argmin(self.logl))

    @property
    def best_index(self) -> int:
        """Index of the live point with the highest log-likelihood."""
        self._require_nonempty()
        return int(np.argmax(self.logl))

    @property
    def worst(self) -> LivePoint:
        """Live point with the lowest log-likelihood."""
        return self._points[self.worst_index]

    @property
    def best(self) -> LivePoint:
        """Live point with the highest log-likelihood."""
        return self._points[self.best_index]

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[LivePoint]:
        return iter(self._points)

    def __getitem__(self, index: int) -> LivePoint:
        return self._points[index]

    def append(
        self,
        values: Sequence[float] | np.ndarray,
        logl: float,
    ) -> None:
        """Append one validated live point."""
        point = self._make_point(values, logl)
        self._points.append(point)

    def replace(
        self,
        index: int,
        values: Sequence[float] | np.ndarray,
        logl: float,
    ) -> LivePoint:
        """Replace one live point and return the removed point."""
        point = self._make_point(values, logl)
        removed = self._points[index]
        self._points[index] = point
        return removed

    def replace_worst(
        self,
        values: Sequence[float] | np.ndarray,
        logl: float,
        *,
        require_improvement: bool = True,
    ) -> LivePoint:
        """Replace the current worst point."""
        point = self._make_point(values, logl)
        index = self.worst_index
        removed = self._points[index]

        if require_improvement and not point.logl > removed.logl:
            raise ValueError(
                "Replacement live point must have log-likelihood "
                "strictly above the current worst point."
            )

        self._points[index] = point
        return removed

    def sort(self) -> None:
        """Sort live points by increasing log-likelihood."""
        self._points.sort(
            key=lambda point: point.logl
        )

    def covariance(
        self,
        *,
        proposal_space: bool = True,
        ddof: int = 1,
    ) -> np.ndarray:
        """Estimate the live-point covariance matrix."""
        if self.size < 2:
            raise ValueError(
                "At least two live points are required for covariance."
            )

        if ddof < 0:
            raise ValueError(
                "ddof must be non-negative."
            )

        matrix = (
            self.proposal_values
            if proposal_space
            else self.values
        )

        covariance = np.cov(
            matrix,
            rowvar=False,
            ddof=ddof,
        )

        return np.atleast_2d(
            np.asarray(covariance, dtype=float)
        )

    def _make_point(
        self,
        values: Sequence[float] | np.ndarray,
        logl: float,
    ) -> LivePoint:
        array = np.asarray(values, dtype=float)

        if array.shape != (self.ndim,):
            raise ValueError(
                f"Expected live-point shape {(self.ndim,)!r}, "
                f"received {array.shape!r}."
            )

        return LivePoint(
            values=array,
            logl=logl,
        )

    def _require_nonempty(self) -> None:
        if not self._points:
            raise ValueError(
                "LivePoints is empty."
            )
