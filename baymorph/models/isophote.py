r"""Isophote coordinates of the profile components of AD-048.

A component on generalized ellipses measures its radius as

.. math::

    m = \left(|x'|^{c} + |y'/q|^{c}\right)^{1/c}, \qquad c = 2 + c_0,

where ``(x', y')`` are the offsets from the centre along the major and the
minor axis. ``c0 = 0`` gives ellipses, ``c0 > 0`` boxy and ``c0 < 0`` discy
isophotes, the convention of GALFIT and Imfit. The area within ``m`` is
``K(c) q m**2``, so a profile of ``m`` keeps its radial meaning on every
isophote shape.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaln

FloatArray = NDArray[np.float64]

MIN_C0 = -1.0
"""Smallest ``c0``: ``c = 1`` is the last convex isophote, a rhombus."""


def validate_c0(c0: float, *, name: str = "c0") -> None:
    """Require a finite isophote exponent with convex isophotes."""
    if not np.isfinite(c0) or c0 < MIN_C0:
        raise ValueError(
            f"{name} must be finite and at least {MIN_C0:g}, so that the "
            "isophotes are convex."
        )


def major_minor_offsets(
    x: ArrayLike,
    y: ArrayLike,
    *,
    x0: float,
    y0: float,
    theta: float,
) -> tuple[FloatArray, FloatArray]:
    """Return the offsets from ``(x0, y0)`` along the major and minor axis.

    ``theta`` is the angle of the major axis from the ``+x`` axis toward
    ``+y``, the convention of :class:`baymorph.models.SersicModel`.
    """
    dx = np.asarray(x, dtype=float) - x0
    dy = np.asarray(y, dtype=float) - y0
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return (
        dx * cos_theta + dy * sin_theta,
        -dx * sin_theta + dy * cos_theta,
    )


def generalized_radius(
    major: FloatArray,
    minor: FloatArray,
    *,
    q: float,
    c0: float = 0.0,
) -> FloatArray:
    """Return the radius ``m`` of the generalized ellipse through a point."""
    if c0 == 0.0:
        return np.sqrt(major * major + (minor / q) ** 2)

    c = 2.0 + c0

    return (np.abs(major) ** c + np.abs(minor / q) ** c) ** (1.0 / c)


def disc_azimuth(
    major: FloatArray,
    minor: FloatArray,
    *,
    q: float,
) -> FloatArray:
    """Return the azimuth in the plane of a thin disc.

    The disc is seen with axis ratio ``q``; the azimuth is measured in its
    own plane from the major axis toward the ``+minor`` direction, in
    ``(-pi, pi]``.
    """
    return np.arctan2(minor / q, major)


def area_factor(c0: float = 0.0) -> float:
    """Return ``K(c) = 4 Gamma(1 + 1/c)**2 / Gamma(1 + 2/c)``.

    The area within the generalized ellipse of radius ``m`` and axis ratio
    ``q`` is ``K(c) q m**2``; ``K = pi`` for ellipses.
    """
    validate_c0(c0)

    if c0 == 0.0:
        return float(np.pi)

    c = 2.0 + c0

    return float(
        4.0 * np.exp(2.0 * gammaln(1.0 + 1.0 / c) - gammaln(1.0 + 2.0 / c))
    )
