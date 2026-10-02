"""Base class of the profile components of AD-048.

A profile component is an intrinsic surface brightness defined on the whole
plane by a closed-form expression. It renders through the generic cell rules
of :mod:`baymorph.rendering.profile_integration`, with the renderer names of
:class:`baymorph.models.SersicModel`, so the observation model renders it
under an analytic PSF and under the effective-PSF rendering of AD-046
without knowing its type.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from baymorph.rendering.profile_integration import (
    DEFAULT_INTEGRATION_ORDER,
    DEFAULT_MAX_DEPTH,
    DEFAULT_RTOL,
    FLUX_ATOL,
    render_cells,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def require_positive(value: float, name: str) -> None:
    """Require a finite, strictly positive parameter."""
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")


def require_axis_ratio(value: float, name: str = "q") -> None:
    """Require an axis ratio in ``(0, 1]``."""
    if not np.isfinite(value) or value <= 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and satisfy 0 < {name} <= 1.")


def require_finite(value: float, name: str) -> None:
    """Require a finite parameter."""
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")


@dataclass(frozen=True)
class ProfileComponent:
    """Intrinsic component given by a surface brightness on the plane.

    Subclasses are frozen dataclasses with the fields ``flux``, ``x0`` and
    ``y0``. They name the fields that are lengths in ``LENGTH_FIELDS``, so
    that :meth:`scaled` can express the component on a finer grid, and
    implement :meth:`intensity`. A component whose profile has a kink or an
    edge away from its centre returns those cells from :meth:`nonsmooth_cells`
    so that the hybrid rules integrate them adaptively; one whose support
    ends at an edge also returns, from :meth:`edge_cells`, the cells the edge
    crosses at every level of that integration. An oscillating profile
    returns its local wavelength from :meth:`local_wavelength`.

    ``flux`` is the total flux of the continuous profile over the whole
    plane, the analytic convention of the Sérsic profile.
    """

    LENGTH_FIELDS: ClassVar[tuple[str, ...]] = ()
    NAME: ClassVar[str] = "component"

    def centre(self) -> tuple[float, float]:
        """Return the centre ``(x0, y0)`` in pixel coordinates."""
        return float(self.x0), float(self.y0)

    def scaled(self, factor: int) -> "ProfileComponent":
        """Return this component on a pixel grid finer by ``factor``.

        Cell ``J`` of the finer grid is centred at the coordinate
        ``(J + 0.5) / factor - 0.5`` of this grid, so every length is
        multiplied by ``factor`` and the centre mapped accordingly. The
        total flux is unchanged.
        """
        if not isinstance(factor, (int, np.integer)) or factor < 1:
            raise ValueError("factor must be a positive integer.")

        changes = {
            name: getattr(self, name) * factor
            for name in self.LENGTH_FIELDS
        }
        changes["x0"] = factor * (self.x0 + 0.5) - 0.5
        changes["y0"] = factor * (self.y0 + 0.5) - 0.5

        return replace(self, **changes)

    def intensity(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> FloatArray:
        """Return the surface brightness at global coordinates ``(x, y)``."""
        raise NotImplementedError

    def nonsmooth_cells(
        self,
        x: FloatArray,
        y: FloatArray,
    ) -> BoolArray | None:
        """Return the cells, centred at ``(x, y)``, where the profile is not smooth.

        The centre needs no entry: the hybrid rules always integrate about it
        adaptively.
        """
        return None

    def edge_cells(
        self,
        x: FloatArray,
        y: FloatArray,
        half: float,
    ) -> BoolArray | None:
        """Return the cells of half-width ``half`` that an edge may cross.

        A profile that ends at the edge of a finite support returns the
        cells, centred at ``(x, y)``, that the edge may cross, until they are
        small enough; adaptive integration splits them whatever its test of
        agreement says. A profile without such an edge returns ``None``.
        """
        return None

    def local_wavelength(
        self,
        x: FloatArray,
        y: FloatArray,
    ) -> FloatArray | None:
        """Return a lower bound of the local wavelength at ``(x, y)``, in cells.

        The hybrid rules integrate adaptively the cells where it is shorter
        than their rule resolves. A profile that does not oscillate returns
        ``None``.
        """
        return None

    def render(
        self,
        shape: tuple[int, int],
        *,
        renderer: str = "hybrid",
        integration_order: int = DEFAULT_INTEGRATION_ORDER,
        rtol: float = DEFAULT_RTOL,
        atol: float | None = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> FloatArray:
        """Render the component on the requested grid.

        ``hybrid`` is the default for PSFs without the pixel response;
        ``point`` and ``hybrid_cusp`` are the outer and fine-cell rules of
        the effective-PSF rendering of AD-046; ``fixed`` and ``adaptive``
        serve validation. See :mod:`baymorph.rendering.profile_integration`.

        ``atol`` is the absolute tolerance of adaptive integration per cell;
        by default ``FLUX_ATOL`` times the total flux, so that a cusp
        converges at a depth independent of the flux scale.
        """
        if atol is None:
            atol = FLUX_ATOL * float(self.flux)

        return render_cells(
            self.intensity,
            shape,
            origin,
            renderer=renderer,
            centre=self.centre(),
            nonsmooth=self.nonsmooth_cells,
            edges=self.edge_cells,
            wavelength=self.local_wavelength,
            integration_order=integration_order,
            rtol=rtol,
            atol=atol,
            max_depth=max_depth,
        )

    def render_components(
        self,
        shape: tuple[int, int],
        **kwargs,
    ) -> dict[str, FloatArray]:
        """Render the component as a single named contribution."""
        return {
            self.NAME: self.render(
                shape,
                **kwargs,
            )
        }
