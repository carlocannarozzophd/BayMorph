r"""Exponential disc with two logarithmic spiral arms (AD-048).

The surface brightness is that of the exponential disc modulated by two
arms,

.. math::

    I(R, \varphi) = I_d(R) \left\{1 + A(R) \cos\left[2(\varphi - \varphi_0)
    - 2 w \ln\left(\frac{R}{R_e}\right)\right]\right\},
    \qquad
    A(R) = A_0 \left[1 - e^{-(R / r_{\mathrm{in}})^{2}}\right],

where ``R`` and ``phi`` are the radius and the azimuth in the plane of the
disc, deprojected with its axis ratio and position angle, ``I_d`` is the
exponential disc of half-light radius ``R_e``
(:class:`baymorph.models.ExponentialComponent`), and ``w = cot psi`` for the
pitch angle ``psi`` of the arms. The crests of the arms follow
``phi = phi0 + w ln(R / R_e)`` modulo ``pi``, so ``phi0`` is the azimuth of
an arm at ``R_e``, where the arms are best measured. The amplitude vanishes
at the centre, where the phase diverges, over the scale ``r_in``.

The azimuth runs from the major axis toward the ``+minor`` direction, which
is counter-clockwise on the detector with ``x`` to the right and ``y`` up.
For ``w > 0`` the arms therefore wind counter-clockwise outward and draw a
Z; for ``w < 0`` they draw an S.

The cosine integrates to zero over every ellipse, so the flux is the flux of
the exponential disc, exactly, and the profile is non-negative for
``0 <= A0 <= 1``. The disc is rendered by the production rules of the Sérsic
profile and the zero-mean modulation by the generic rules of AD-048.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from functools import cached_property
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from baymorph.models.exponential import ExponentialComponent
from baymorph.models.isophote import disc_azimuth, major_minor_offsets
from baymorph.models.profile import (
    ProfileComponent,
    require_axis_ratio,
    require_finite,
    require_positive,
)

FloatArray = NDArray[np.float64]

ARMS = 2
"""Number of arms; each number of arms would be a separate model."""


def _validate(component) -> None:
    """Validate the parameters shared by the disc and its modulation."""
    require_positive(component.flux, "flux")
    require_positive(component.re, "re")
    require_axis_ratio(component.q)
    require_finite(component.theta, "theta")

    if (
        not np.isfinite(component.amplitude)
        or component.amplitude < 0.0
        or component.amplitude > 1.0
    ):
        raise ValueError(
            "amplitude must lie between 0 and 1, so that the disc stays "
            "non-negative."
        )

    require_finite(component.winding, "winding")
    require_finite(component.phase, "phase")
    require_positive(component.r_in, "r_in")
    require_finite(component.x0, "x0")
    require_finite(component.y0, "y0")


@dataclass(frozen=True)
class LogSpiralModulation(ProfileComponent):
    """The zero-mean modulation that two spiral arms add to an exponential disc.

    It has the parameters of :class:`LogSpiralDiskComponent`. ``flux`` is the
    flux of the disc it modulates; its own integral vanishes. It is rendered
    by the generic cell rules and is part of a disc, not a component of a
    model by itself.
    """

    LENGTH_FIELDS = ("re", "r_in")
    NAME = "arms"

    flux: float
    re: float
    q: float
    theta: float
    amplitude: float
    winding: float
    phase: float
    r_in: float
    x0: float
    y0: float

    def __post_init__(self) -> None:
        """Validate the parameters."""
        _validate(self)

    @cached_property
    def disc(self) -> ExponentialComponent:
        """Return the exponential disc that the arms modulate."""
        return ExponentialComponent(
            flux=self.flux,
            re=self.re,
            q=self.q,
            theta=self.theta,
            x0=self.x0,
            y0=self.y0,
        )

    def _disc_coordinates(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> tuple[FloatArray, FloatArray]:
        """Return the offsets along the major axis and in the plane of the disc."""
        major, minor = major_minor_offsets(
            x,
            y,
            x0=self.x0,
            y0=self.y0,
            theta=self.theta,
        )

        return major, minor / self.q

    def intensity(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> FloatArray:
        """Return the modulation at global coordinates ``(x, y)``."""
        major, minor = major_minor_offsets(
            x,
            y,
            x0=self.x0,
            y0=self.y0,
            theta=self.theta,
        )
        radius = np.hypot(major, minor / self.q)
        inside = radius > 0.0
        safe = np.where(inside, radius, self.re)
        envelope = self.amplitude * -np.expm1(-((radius / self.r_in) ** 2))
        angle = ARMS * (
            disc_azimuth(major, minor, q=self.q)
            - self.phase
            - self.winding * np.log(safe / self.re)
        )

        return np.where(
            inside,
            self.disc.surface_brightness(radius) * envelope * np.cos(angle),
            0.0,
        )

    def local_wavelength(
        self,
        x: FloatArray,
        y: FloatArray,
    ) -> FloatArray:
        """Return the local wavelength of the arms on the detector, in cells.

        In the plane of the disc the phase has the gradient
        ``(2 / R**2) (-(Y + w X), X - w Y)``; on the detector its second
        component is divided by ``q``. The wavelength is ``2 pi`` over the
        norm, and vanishes at the centre.
        """
        major, minor = self._disc_coordinates(x, y)
        gradient = np.hypot(
            minor + self.winding * major,
            (major - self.winding * minor) / self.q,
        )
        squared = major * major + minor * minor

        return np.where(
            gradient > 0.0,
            np.pi * squared / np.where(gradient > 0.0, gradient, 1.0),
            0.0,
        )


@dataclass(frozen=True)
class LogSpiralDiskComponent:
    """Exponential disc with two logarithmic spiral arms.

    Parameters
    ----------
    flux : float
        Total flux of the disc, arms included.
    re : float
        Half-light radius of the exponential disc, in pixels, and the radius
        at which ``phase`` is measured.
    q : float
        Axis ratio of the disc.
    theta : float
        Position angle of the major axis, in radians, from ``+x`` toward
        ``+y``.
    amplitude : float
        Relative amplitude ``A0`` of the arms, from 0 to 1.
    winding : float
        ``w = cot psi`` for the pitch angle ``psi``; positive for arms that
        draw a Z on the detector, negative for an S.
    phase : float
        Azimuth of an arm at ``re``, in radians, in the plane of the disc.
    r_in : float
        Radius, in pixels, over which the arms fade toward the centre.
    x0, y0 : float
        Centre in pixel coordinates.
    """

    NAME: ClassVar[str] = "disk"

    flux: float
    re: float
    q: float
    theta: float
    amplitude: float
    winding: float
    phase: float
    r_in: float
    x0: float
    y0: float

    def __post_init__(self) -> None:
        """Validate the parameters."""
        _validate(self)

    @property
    def exponential(self) -> ExponentialComponent:
        """Return the exponential disc, rendered by the Sérsic rules."""
        return ExponentialComponent(
            flux=self.flux,
            re=self.re,
            q=self.q,
            theta=self.theta,
            x0=self.x0,
            y0=self.y0,
        )

    @property
    def modulation(self) -> LogSpiralModulation:
        """Return the zero-mean modulation of the arms."""
        return LogSpiralModulation(
            **{field.name: getattr(self, field.name) for field in fields(self)}
        )

    def centre(self) -> tuple[float, float]:
        """Return the centre ``(x0, y0)`` in pixel coordinates."""
        return float(self.x0), float(self.y0)

    def scaled(self, factor: int) -> "LogSpiralDiskComponent":
        """Return this disc on a pixel grid finer by ``factor``.

        See :meth:`baymorph.models.profile.ProfileComponent.scaled`; ``re``
        and ``r_in`` are lengths, and the winding and the phase do not
        change.
        """
        if not isinstance(factor, (int, np.integer)) or factor < 1:
            raise ValueError("factor must be a positive integer.")

        return replace(
            self,
            re=self.re * factor,
            r_in=self.r_in * factor,
            x0=factor * (self.x0 + 0.5) - 0.5,
            y0=factor * (self.y0 + 0.5) - 0.5,
        )

    def intensity(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> FloatArray:
        """Return the surface brightness at global coordinates ``(x, y)``."""
        modulation = self.modulation
        major, minor = modulation._disc_coordinates(x, y)
        disc = self.exponential.surface_brightness(np.hypot(major, minor))

        return disc + modulation.intensity(x, y)

    def render(
        self,
        shape: tuple[int, int],
        *,
        renderer: str = "hybrid",
        origin: tuple[float, float] = (0.0, 0.0),
        **controls,
    ) -> FloatArray:
        """Render the disc and its arms on the requested grid.

        The exponential disc is rendered by the Sérsic rules with
        ``renderer``; the modulation by the generic rules with the same
        renderer and ``controls``.
        """
        return self.exponential.render(
            shape,
            renderer=renderer,
            origin=origin,
        ) + self.modulation.render(
            shape,
            renderer=renderer,
            origin=origin,
            **controls,
        )

    def render_components(
        self,
        shape: tuple[int, int],
        **kwargs,
    ) -> dict[str, FloatArray]:
        """Render the disc, arms included, as a single named contribution."""
        return {self.NAME: self.render(shape, **kwargs)}
