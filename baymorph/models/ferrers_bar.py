r"""Ferrers bar on generalized ellipses (AD-048).

The surface brightness of the Ferrers profile, in the form of GALFIT (Peng et
al. 2010) and Imfit (Erwin 2015) with its central slope fixed to zero, is

.. math::

    I(m) = I_0 \left[1 - \left(\frac{m}{a}\right)^{2}\right]^{\alpha}
    \quad (m < a), \qquad I(m) = 0 \quad (m \geq a),

on generalized ellipses of shape ``c0`` (see :mod:`baymorph.models.isophote`);
``c0 > 0`` gives the boxy isophotes of bars (Athanassoula et al. 1990).
``a`` is the semi-major length at which the light of the bar ends and
``alpha`` the sharpness of its edge: the slope at the edge is infinite for
``alpha < 1``, finite for ``alpha = 1`` and zero beyond. The area within ``m``
is ``K(c) q m**2``, so the total flux is

.. math::

    F = \frac{K(c)\, q\, I_0\, a^{2}}{\alpha + 1}.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from baymorph.models.generalized_sersic import axis_strips
from baymorph.models.isophote import (
    area_factor,
    generalized_radius,
    major_minor_offsets,
    radius_reach,
    validate_c0,
)
from baymorph.models.profile import (
    ProfileComponent,
    require_axis_ratio,
    require_finite,
    require_positive,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

EDGE_CELL_MARGIN = 2.0
"""Cells whose centre lies within this many cells of the edge, along ``x`` or
``y``, are integrated adaptively.

Four-point Gauss-Legendre quadrature of a cell whose centre lies two cells
inside an edge with ``alpha = 0.5``, the steepest of the default prior, is
accurate to about 1e-9 of the profile there.
"""

EDGE_RESOLUTION = 512
"""Cells the edge crosses are split until their side is at most ``a`` divided
by this number."""


@dataclass(frozen=True)
class FerrersBarComponent(ProfileComponent):
    """Ferrers bar with boxy or discy isophotes.

    Parameters
    ----------
    flux : float
        Total flux of the bar.
    a : float
        Semi-major length of the bar, where its light ends, in pixels.
    alpha : float
        Sharpness of the edge, positive.
    c0 : float
        Isophote shape: 0 for ellipses, positive for boxy and negative for
        discy isophotes, at least -1 so that they stay convex.
    q : float
        Axis ratio of the isophotes.
    theta : float
        Position angle of the major axis, in radians, from ``+x`` toward
        ``+y``.
    x0, y0 : float
        Centre in pixel coordinates.
    """

    LENGTH_FIELDS = ("a",)
    NAME = "bar"

    flux: float
    a: float
    alpha: float
    c0: float
    q: float
    theta: float
    x0: float
    y0: float

    def __post_init__(self) -> None:
        """Validate the parameters."""
        require_positive(self.flux, "flux")
        require_positive(self.a, "a")
        require_positive(self.alpha, "alpha")
        validate_c0(self.c0)
        require_axis_ratio(self.q)
        require_finite(self.theta, "theta")
        require_finite(self.x0, "x0")
        require_finite(self.y0, "y0")

    @property
    def log_central_intensity(self) -> float:
        """Return the logarithm of the central surface brightness ``I0``."""
        return float(
            np.log(self.flux)
            + np.log(self.alpha + 1.0)
            - np.log(area_factor(self.c0))
            - np.log(self.q)
            - 2.0 * np.log(self.a)
        )

    def _offsets(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> tuple[FloatArray, FloatArray]:
        return major_minor_offsets(
            x,
            y,
            x0=self.x0,
            y0=self.y0,
            theta=self.theta,
        )

    def _radius(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        major, minor = self._offsets(x, y)

        return (
            major,
            minor,
            generalized_radius(major, minor, q=self.q, c0=self.c0),
        )

    def intensity(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> FloatArray:
        """Return the surface brightness at global coordinates ``(x, y)``."""
        _, _, radius = self._radius(x, y)
        inside = np.maximum(1.0 - (radius / self.a) ** 2, 0.0)

        return np.exp(self.log_central_intensity) * inside**self.alpha

    def nonsmooth_cells(
        self,
        x: FloatArray,
        y: FloatArray,
    ) -> BoolArray:
        """Return the cells near the edge, and along the axes within the bar.

        The edge, where the profile ends, is not smooth for any ``alpha``.
        Non-elliptical isophotes are not smooth along their axes either
        (see :func:`baymorph.models.generalized_sersic.axis_strips`); only
        the part of the axes inside the bar matters, since the profile is
        zero beyond it.
        """
        major, minor, radius = self._radius(x, y)
        reach = radius_reach(
            EDGE_CELL_MARGIN,
            q=self.q,
            c0=self.c0,
            theta=self.theta,
        )
        cells = np.abs(radius - self.a) <= reach
        strips = axis_strips(major, minor, self.c0)

        if strips is not None:
            cells = cells | (strips & (radius < self.a))

        return cells

    def edge_cells(
        self,
        x: FloatArray,
        y: FloatArray,
        half: float,
    ) -> BoolArray | None:
        """Return the cells of half-width ``half`` that the edge may cross.

        Cells are split while their side exceeds ``a / EDGE_RESOLUTION``, a
        fraction of the bar, so that the rule is the same on every grid.
        """
        if 2.0 * half <= self.a / EDGE_RESOLUTION:
            return None

        _, _, radius = self._radius(x, y)

        return np.abs(radius - self.a) <= radius_reach(
            half,
            q=self.q,
            c0=self.c0,
            theta=self.theta,
        )
