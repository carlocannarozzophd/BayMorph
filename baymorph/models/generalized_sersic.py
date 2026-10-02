r"""Sérsic profile on generalized ellipses: boxy and discy isophotes (AD-048).

The surface brightness is the Sérsic law of the generalized radius

.. math::

    I(m) = I_0 \exp\left[-b_n \left(\frac{m}{R_e}\right)^{1/n}\right],
    \qquad
    m = \left(|x'|^{c} + |y'/q|^{c}\right)^{1/c}, \quad c = 2 + c_0,

with ``c0 > 0`` for boxy and ``c0 < 0`` for discy isophotes, the convention
of GALFIT (Peng et al. 2010) and Imfit (Erwin 2015) for the generalized
ellipses of Athanassoula et al. (1990). The area within ``m`` is
``K(c) q m**2``, ``pi`` over the area ratio of Peng et al. (2002), so the
fraction of light within an isophote depends on ``m`` alone: ``re`` is the
semi-major radius of the isophote that holds half the light for every
``c0``, and the total flux is

.. math::

    F = 2 K(c) q I_0 R_e^2 n b_n^{-2n} \Gamma(2n),

which for ellipses, ``K = pi``, is the flux of :class:`SersicModel`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaincinv, gammaln

from baymorph.models.isophote import (
    area_factor,
    generalized_radius,
    major_minor_offsets,
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

AXIS_CELL_MARGIN = 0.75
"""Half-width, in cells, of the strips along the axes integrated adaptively."""


def axis_strips(
    major: FloatArray,
    minor: FloatArray,
    c0: float,
) -> BoolArray | None:
    """Return the cells near the axes, where ``|x'|**c`` is not smooth.

    For ``c`` other than an even integer the generalized radius has a
    singular derivative on the axes of the isophotes, a kink at ``c = 1``;
    the cells whose centre lies within ``AXIS_CELL_MARGIN`` of an axis are
    integrated adaptively. Ellipses and ``c = 4, 6, ...`` need none.
    """
    c = 2.0 + c0

    if c0 == 0.0 or (c == np.round(c) and int(round(c)) % 2 == 0):
        return None

    return (np.abs(major) < AXIS_CELL_MARGIN) | (
        np.abs(minor) < AXIS_CELL_MARGIN
    )


@dataclass(frozen=True)
class GeneralizedSersicComponent(ProfileComponent):
    """Sérsic profile with boxy or discy isophotes.

    Parameters
    ----------
    flux : float
        Total flux of the continuous profile over the whole plane.
    re : float
        Semi-major radius of the half-light isophote, in pixels.
    n : float
        Sérsic index.
    q : float
        Axis ratio of the isophotes.
    c0 : float
        Isophote shape: 0 for ellipses, positive for boxy and negative for
        discy isophotes, at least -1 so that they stay convex.
    theta : float
        Position angle of the major axis, in radians, from ``+x`` toward
        ``+y``.
    x0, y0 : float
        Centre in pixel coordinates.
    """

    LENGTH_FIELDS = ("re",)
    NAME = "boxy_sersic"

    flux: float
    re: float
    n: float
    q: float
    c0: float
    theta: float
    x0: float
    y0: float

    def __post_init__(self) -> None:
        """Validate the parameters."""
        require_positive(self.flux, "flux")
        require_positive(self.re, "re")
        require_positive(self.n, "n")
        require_axis_ratio(self.q)
        validate_c0(self.c0)
        require_finite(self.theta, "theta")
        require_finite(self.x0, "x0")
        require_finite(self.y0, "y0")

    @property
    def bn(self) -> float:
        """Return the Sérsic constant enclosing half the light."""
        return float(gammaincinv(2.0 * self.n, 0.5))

    @property
    def log_central_intensity(self) -> float:
        """Return the logarithm of the central surface brightness ``I0``."""
        bn = self.bn
        log_normalization = (
            np.log(2.0)
            + np.log(area_factor(self.c0))
            + np.log(self.q)
            + 2.0 * np.log(self.re)
            + np.log(self.n)
            + gammaln(2.0 * self.n)
            - 2.0 * self.n * np.log(bn)
        )

        return float(np.log(self.flux) - log_normalization)

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

    def intensity(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> FloatArray:
        """Return the surface brightness at global coordinates ``(x, y)``."""
        major, minor = self._offsets(x, y)
        radius = generalized_radius(major, minor, q=self.q, c0=self.c0)

        return np.exp(
            self.log_central_intensity
            - self.bn * (radius / self.re) ** (1.0 / self.n)
        )

    def nonsmooth_cells(
        self,
        x: FloatArray,
        y: FloatArray,
    ) -> BoolArray | None:
        """Return the cells along the axes of non-elliptical isophotes."""
        major, minor = self._offsets(x, y)

        return axis_strips(major, minor, self.c0)
