r"""Exponential disc with a break in its scale length (AD-048).

The surface brightness of Erwin, Pohlen & Beckman (2008) is

.. math::

    I(r) = I_0 \exp\left(-\frac{r}{h_{\mathrm{in}}}\right)
    \left[\frac{1 + e^{(r - r_b)/w_b}}{1 + e^{-r_b/w_b}}\right]^{
    w_b (1/h_{\mathrm{in}} - 1/h_{\mathrm{out}})},

an exponential of scale length ``h_in`` inside the break radius ``rb`` and
of scale length ``h_out`` outside it, joined over a width ``wb``. Erwin,
Pohlen & Beckman write the sharpness ``alpha = 1 / wb``; the width is used
here so that every parameter but the flux is a length. ``h_out < h_in``
gives a down-bending profile (type II, Freeman 1970), ``h_out > h_in`` an
up-bending one (type III, Erwin, Beckman & Pohlen 2005), and ``h_out = h_in``
a single exponential (type I). ``flux`` is the total flux over the whole
plane, and ``I0`` follows from it by a radial integral computed
numerically.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad

from baymorph.models.isophote import generalized_radius, major_minor_offsets
from baymorph.models.profile import (
    ProfileComponent,
    require_axis_ratio,
    require_finite,
    require_positive,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BrokenExponentialComponent(ProfileComponent):
    """Exponential disc whose scale length changes at a break radius.

    Parameters
    ----------
    flux : float
        Total flux of the continuous profile over the whole plane.
    h_in : float
        Scale length inside the break, in pixels.
    h_out : float
        Scale length outside the break, in pixels.
    rb : float
        Break radius, in pixels.
    wb : float
        Width of the transition at the break, in pixels.
    q : float
        Axis ratio.
    theta : float
        Position angle of the major axis, in radians, from ``+x`` toward
        ``+y``.
    x0, y0 : float
        Centre in pixel coordinates.
    """

    LENGTH_FIELDS = ("h_in", "h_out", "rb", "wb")
    NAME = "disk"

    flux: float
    h_in: float
    h_out: float
    rb: float
    wb: float
    q: float
    theta: float
    x0: float
    y0: float

    def __post_init__(self) -> None:
        """Validate the parameters."""
        require_positive(self.flux, "flux")
        require_positive(self.h_in, "h_in")
        require_positive(self.h_out, "h_out")
        require_positive(self.rb, "rb")
        require_positive(self.wb, "wb")
        require_axis_ratio(self.q)
        require_finite(self.theta, "theta")
        require_finite(self.x0, "x0")
        require_finite(self.y0, "y0")

    def _log_shape(self, radius: FloatArray) -> FloatArray:
        """Return the logarithm of the profile with ``I0 = 1``."""
        exponent = self.wb * (1.0 / self.h_in - 1.0 / self.h_out)
        transition = np.logaddexp(0.0, (radius - self.rb) / self.wb) - (
            np.logaddexp(0.0, -self.rb / self.wb)
        )

        return -radius / self.h_in + exponent * transition

    @cached_property
    def log_central_intensity(self) -> float:
        """Return ``log I0`` such that the total flux is ``flux``.

        The flux is ``2 pi q`` times the integral of ``I(r) r dr``, split at
        the break.
        """

        def integrand(radius: float) -> float:
            return float(np.exp(self._log_shape(np.asarray(radius)))) * radius

        inner, _ = quad(
            integrand, 0.0, self.rb, limit=400, epsabs=0.0, epsrel=1.0e-12
        )
        outer, _ = quad(
            integrand, self.rb, np.inf, limit=400, epsabs=0.0, epsrel=1.0e-12
        )

        return float(
            np.log(self.flux)
            - np.log(2.0 * np.pi * self.q * (inner + outer))
        )

    def surface_brightness(self, radius: ArrayLike) -> FloatArray:
        """Return the surface brightness at elliptical radius ``radius``."""
        radius = np.asarray(radius, dtype=float)

        return np.exp(self.log_central_intensity + self._log_shape(radius))

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
