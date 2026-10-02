r"""Core-Sérsic profile: a Sérsic body with a depleted core (AD-048).

The surface brightness of Graham et al. (2003) and Trujillo et al. (2004) is

.. math::

    I(R) = I' \left[1 + \left(\frac{R_b}{R}\right)^{\alpha}\right]^{\gamma/\alpha}
    \exp\left\{-b_n \left[\frac{R^{\alpha} + R_b^{\alpha}}
    {R_e^{\alpha}}\right]^{1/(\alpha n)}\right\},

an inner power law of slope ``gamma`` within the break radius ``rb`` and a
Sérsic profile outside it, joined by a transition whose sharpness grows with
``alpha``. ``b_n`` is the Sérsic constant of index ``n``, so ``re`` is the
half-light radius of the outer Sérsic profile; the half-light radius of the
whole profile is a derived quantity, smaller by the light the core lacks.
``flux`` is the total flux over the whole plane; ``I'`` follows from it by
a radial integral computed numerically.

Baes (2026) finds that the deprojected density is monotonic only for
``alpha`` up to a critical value that equals 2 for ``gamma = 0`` and is never
below 2, so ``alpha <= 2`` is admissible for every core.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad
from scipy.special import gammaincinv

from baymorph.models.isophote import generalized_radius, major_minor_offsets
from baymorph.models.profile import (
    ProfileComponent,
    require_axis_ratio,
    require_finite,
    require_positive,
)

FloatArray = NDArray[np.float64]

MAX_GAMMA = 2.0
"""Inner slopes of 2 or more hold an infinite flux at the centre."""

_LOWEST_RADIUS_FRACTION = 1.0e-10
"""Smallest radius, as a fraction of ``rb``, of the normalization integral."""


@dataclass(frozen=True)
class CoreSersicComponent(ProfileComponent):
    """Core-Sérsic profile on ellipses.

    Parameters
    ----------
    flux : float
        Total flux of the continuous profile over the whole plane.
    re : float
        Half-light radius of the outer Sérsic profile, in pixels.
    n : float
        Sérsic index of the outer profile.
    rb : float
        Break radius between the core and the Sérsic profile, in pixels.
    gamma : float
        Logarithmic slope of the inner power law, below 2.
    alpha : float
        Sharpness of the transition, positive.
    q : float
        Axis ratio.
    theta : float
        Position angle of the major axis, in radians, from ``+x`` toward
        ``+y``.
    x0, y0 : float
        Centre in pixel coordinates.
    """

    LENGTH_FIELDS = ("re", "rb")
    NAME = "core_sersic"

    flux: float
    re: float
    n: float
    rb: float
    gamma: float
    alpha: float
    q: float
    theta: float
    x0: float
    y0: float

    def __post_init__(self) -> None:
        """Validate the parameters."""
        require_positive(self.flux, "flux")
        require_positive(self.re, "re")
        require_positive(self.n, "n")
        require_positive(self.rb, "rb")
        require_positive(self.alpha, "alpha")
        require_finite(self.gamma, "gamma")

        if self.gamma >= MAX_GAMMA:
            raise ValueError(
                "gamma must be below 2: a steeper core holds an infinite flux."
            )

        require_axis_ratio(self.q)
        require_finite(self.theta, "theta")
        require_finite(self.x0, "x0")
        require_finite(self.y0, "y0")

    @cached_property
    def bn(self) -> float:
        """Return the Sérsic constant of the outer profile."""
        return float(gammaincinv(2.0 * self.n, 0.5))

    def _log_shape(self, radius: FloatArray) -> FloatArray:
        """Return the logarithm of the profile with ``I' = 1``.

        Both terms are formed with ``logaddexp``, so neither overflows at
        small radii or large ``alpha``.
        """
        log_radius = np.log(radius)
        log_rb = np.log(self.rb)
        core = (self.gamma / self.alpha) * np.logaddexp(
            0.0,
            self.alpha * (log_rb - log_radius),
        )
        outer = np.exp(
            (
                np.logaddexp(self.alpha * log_radius, self.alpha * log_rb)
                - self.alpha * np.log(self.re)
            )
            / (self.alpha * self.n)
        )

        return core - self.bn * outer

    @cached_property
    def log_intensity_scale(self) -> float:
        """Return ``log I'`` such that the total flux is ``flux``.

        The flux is ``2 pi q`` times the integral of ``I(R) R dR``, computed
        in ``ln R`` from ``1e-10 rb`` until the profile is negligible. Below
        ``1e-10 rb`` the profile is the power law ``R**-gamma``, whose
        integral ``I(R) R**2 / (2 - gamma)`` at the lower limit is added.
        """
        log_low = np.log(_LOWEST_RADIUS_FRACTION * self.rb)
        log_high = np.log(self.re) + self.n * np.log(800.0 / self.bn)
        breaks = sorted(
            value
            for value in (np.log(self.rb), np.log(self.re))
            if log_low < value < log_high
        )

        def integrand(log_radius: float) -> float:
            radius = np.exp(log_radius)

            return float(
                np.exp(self._log_shape(np.asarray(radius)) + 2.0 * log_radius)
            )

        integral, _ = quad(
            integrand,
            log_low,
            log_high,
            points=breaks,
            limit=400,
            epsabs=0.0,
            epsrel=1.0e-11,
        )
        integral += integrand(log_low) / (MAX_GAMMA - self.gamma)

        return float(
            np.log(self.flux) - np.log(2.0 * np.pi * self.q * integral)
        )

    def surface_brightness(self, radius: ArrayLike) -> FloatArray:
        """Return the surface brightness at elliptical radius ``radius``."""
        radius = np.maximum(
            np.asarray(radius, dtype=float),
            _LOWEST_RADIUS_FRACTION * self.rb,
        )

        return np.exp(self.log_intensity_scale + self._log_shape(radius))

    def intensity(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> FloatArray:
        """Return the surface brightness at global coordinates ``(x, y)``."""
        major, minor = major_minor_offsets(
            x,
            y,
            x0=self.x0,
            y0=self.y0,
            theta=self.theta,
        )

        return self.surface_brightness(
            generalized_radius(major, minor, q=self.q)
        )
