r"""Elliptical ring with a Gaussian radial profile (AD-048).

The surface brightness is

.. math::

    I(m) = A \exp\left[-\frac{(m - r)^{2}}{2 \sigma^{2}}\right],

where ``m`` is the elliptical radius of axis ratio ``q``, ``r`` the
semi-major radius of the crest of the ring and ``sigma`` its width along the
major axis, the Gaussian ring of Imfit (Erwin 2015). The area within ``m`` is
``pi q m**2``, so the total flux is

.. math::

    F = 2 \pi q A \left\{\sigma^{2} e^{-r^{2} / (2 \sigma^{2})}
    + \sqrt{\frac{\pi}{2}}\, r \sigma
    \left[1 + \operatorname{erf}\left(\frac{r}{\sqrt{2}\, \sigma}\right)
    \right]\right\},

which tends to the circumference times the area of the Gaussian, ``2 pi r q``
times ``sqrt(2 pi) sigma A``, for a ring much wider than its width.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import erf

from baymorph.models.isophote import (
    generalized_radius,
    major_minor_offsets,
    radius_reach,
)
from baymorph.models.profile import (
    ProfileComponent,
    require_axis_ratio,
    require_finite,
    require_positive,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

RESOLVED_WIDTH = 1.0
"""Narrowest width ``q * sigma``, in cells, that the fixed cell rules resolve.

Four-point Gauss-Legendre quadrature integrates a Gaussian of unit width over
a cell to 6e-8 of its peak, and one of half that width only to 1.4e-5.
"""

BAND_WIDTHS = 7.0
"""Half-width, in units of ``sigma``, of the band of a narrow ring integrated
adaptively; beyond it the profile, and so the error of any cell rule, stays
below 3e-11 of the crest."""


@dataclass(frozen=True)
class GaussianRingComponent(ProfileComponent):
    """Elliptical ring with a Gaussian radial profile.

    Parameters
    ----------
    flux : float
        Total flux of the ring.
    r : float
        Semi-major radius of the crest of the ring, in pixels.
    sigma : float
        Width of the ring along the major axis, in pixels.
    q : float
        Axis ratio.
    theta : float
        Position angle of the major axis, in radians, from ``+x`` toward
        ``+y``.
    x0, y0 : float
        Centre in pixel coordinates.
    """

    LENGTH_FIELDS = ("r", "sigma")
    NAME = "ring"

    flux: float
    r: float
    sigma: float
    q: float
    theta: float
    x0: float
    y0: float

    def __post_init__(self) -> None:
        """Validate the parameters."""
        require_positive(self.flux, "flux")
        require_positive(self.r, "r")
        require_positive(self.sigma, "sigma")
        require_axis_ratio(self.q)
        require_finite(self.theta, "theta")
        require_finite(self.x0, "x0")
        require_finite(self.y0, "y0")

    @property
    def log_crest_intensity(self) -> float:
        """Return the logarithm of the surface brightness ``A`` on the crest."""
        ratio = self.r / self.sigma
        radial = self.sigma**2 * np.exp(-0.5 * ratio**2) + (
            np.sqrt(0.5 * np.pi)
            * self.r
            * self.sigma
            * (1.0 + erf(ratio / np.sqrt(2.0)))
        )

        return float(
            np.log(self.flux) - np.log(2.0 * np.pi * self.q * radial)
        )

    def _radius(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> FloatArray:
        major, minor = major_minor_offsets(
            x,
            y,
            x0=self.x0,
            y0=self.y0,
            theta=self.theta,
        )

        return generalized_radius(major, minor, q=self.q)

    def intensity(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> FloatArray:
        """Return the surface brightness at global coordinates ``(x, y)``."""
        offset = (self._radius(x, y) - self.r) / self.sigma

        return np.exp(self.log_crest_intensity - 0.5 * offset**2)

    def nonsmooth_cells(
        self,
        x: FloatArray,
        y: FloatArray,
    ) -> BoolArray | None:
        """Return the band about the crest of a ring narrower than a cell.

        Across its crest the ring is narrowest on the minor axis, where its
        width is ``q * sigma``. A ring at least ``RESOLVED_WIDTH`` cells wide
        there needs no adaptive cells; a narrower one is integrated
        adaptively within ``BAND_WIDTHS`` widths of its crest, widened by the
        reach of a cell.
        """
        if self.q * self.sigma >= RESOLVED_WIDTH:
            return None

        reach = radius_reach(0.5, q=self.q, theta=self.theta)

        return np.abs(self._radius(x, y) - self.r) <= (
            BAND_WIDTHS * self.sigma + reach
        )
