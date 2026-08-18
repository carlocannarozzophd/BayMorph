"""Experimental hybrid rendering utilities for Sérsic models."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.special import gammainc

if TYPE_CHECKING:
    from baymorph.models.sersic import SersicModel

FloatArray = NDArray[np.float64]

_TWO_PI = 2.0 * np.pi
_DIRECTION_EPSILON = 1.0e-15
_ANGULAR_BREAKPOINT_ATOL = (
    1024.0 * np.finfo(float).eps * _TWO_PI
)


@lru_cache(maxsize=None)
def _legendre_rule(
    order: int,
) -> tuple[FloatArray, FloatArray]:
    """Return a cached Gauss-Legendre quadrature rule."""
    if (
        not isinstance(order, (int, np.integer))
        or order <= 0
    ):
        raise ValueError(
            "integration_order must be a positive integer."
        )

    nodes, weights = np.polynomial.legendre.leggauss(order)

    nodes = np.asarray(nodes, dtype=float)
    weights = np.asarray(weights, dtype=float)

    nodes.setflags(write=False)
    weights.setflags(write=False)

    return nodes, weights


def _split_central_pixel_integral(
    model: SersicModel,
    *,
    pixel_x: float,
    pixel_y: float,
    integration_order: int,
) -> float:
    """Return a split high-order numerical central-pixel integral.

    This routine is retained only as an independent numerical validation
    reference. Hybrid rendering uses the semi-analytic pixel integral.
    """
    nodes, weights = _legendre_rule(
        integration_order
    )

    xmin = pixel_x - 0.5
    xmax = pixel_x + 0.5
    ymin = pixel_y - 0.5
    ymax = pixel_y + 0.5

    x_edges = [xmin]
    y_edges = [ymin]

    if xmin < model.x0 < xmax:
        x_edges.append(model.x0)

    if ymin < model.y0 < ymax:
        y_edges.append(model.y0)

    x_edges.append(xmax)
    y_edges.append(ymax)

    weight_2d = (
        weights[:, None]
        * weights[None, :]
    )

    cos_theta = np.cos(model.theta)
    sin_theta = np.sin(model.theta)

    total = 0.0

    for xa, xb in zip(
        x_edges[:-1],
        x_edges[1:],
        strict=True,
    ):
        for ya, yb in zip(
            y_edges[:-1],
            y_edges[1:],
            strict=True,
        ):
            half_width = 0.5 * (xb - xa)
            half_height = 0.5 * (yb - ya)

            x_center = 0.5 * (xa + xb)
            y_center = 0.5 * (ya + yb)

            x = (
                x_center
                + half_width * nodes[:, None]
            )
            y = (
                y_center
                + half_height * nodes[None, :]
            )

            dx = x - model.x0
            dy = y - model.y0

            x_rotated = (
                dx * cos_theta
                + dy * sin_theta
            )
            y_rotated = (
                -dx * sin_theta
                + dy * cos_theta
            )

            radius = np.sqrt(
                x_rotated**2
                + (y_rotated / model.q) ** 2
            )

            intensity = np.exp(
                model.log_central_intensity
                - model.bn
                * (radius / model.re)
                ** (1.0 / model.n)
            )

            total += (
                half_width
                * half_height
                * float(
                    np.sum(
                        weight_2d * intensity
                    )
                )
            )

    return float(total)


def _merge_periodic_angles(
    angles: list[float],
    *,
    atol: float = _ANGULAR_BREAKPOINT_ATOL,
) -> tuple[float, ...]:
    """Merge numerically coincident angular directions.

    Detector-pixel vertices can be exactly collinear with the Sérsic centre
    while floating-point coordinate arithmetic produces slightly different
    ``atan2`` values. Such duplicate directions would create artificial,
    extremely narrow integration intervals.

    Angles within ``atol`` are therefore treated as one geometric direction.
    The periodic equivalence between zero and two pi is handled explicitly.
    """
    if not angles:
        return ()

    normalized: list[float] = []

    for angle in angles:
        value = float(
            np.mod(
                angle,
                _TWO_PI,
            )
        )

        if (
            value <= atol
            or _TWO_PI - value <= atol
        ):
            value = 0.0

        normalized.append(value)

    normalized.sort()

    clusters: list[list[float]] = [
        [normalized[0]]
    ]

    for angle in normalized[1:]:
        cluster = clusters[-1]

        if angle - cluster[0] <= atol:
            cluster.append(angle)
        else:
            clusters.append(
                [angle]
            )

    merged = [
        float(np.mean(cluster))
        for cluster in clusters
    ]

    if merged and merged[0] <= atol:
        merged[0] = 0.0

    return tuple(merged)


def _pixel_angular_breakpoints(
    model: SersicModel,
    *,
    pixel_x: float,
    pixel_y: float,
) -> tuple[float, ...]:
    """Return distinct angular directions through transformed pixel vertices."""
    cos_theta = np.cos(model.theta)
    sin_theta = np.sin(model.theta)

    angles: list[float] = []

    for x in (
        pixel_x - 0.5,
        pixel_x + 0.5,
    ):
        for y in (
            pixel_y - 0.5,
            pixel_y + 0.5,
        ):
            dx = x - model.x0
            dy = y - model.y0

            x_rotated = (
                dx * cos_theta
                + dy * sin_theta
            )
            y_rotated = (
                -dx * sin_theta
                + dy * cos_theta
            ) / model.q

            phi = float(
                np.arctan2(
                    y_rotated,
                    x_rotated,
                )
            )

            if phi < 0.0:
                phi += _TWO_PI

            angles.append(phi)

    return _merge_periodic_angles(
        angles
    )


def _ray_pixel_interval(
    model: SersicModel,
    phi: float,
    *,
    pixel_x: float,
    pixel_y: float,
) -> tuple[float, float] | None:
    """Return the elliptical-radius interval intersecting one pixel."""
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    cos_theta = np.cos(model.theta)
    sin_theta = np.sin(model.theta)

    dx_per_radius = (
        cos_phi * cos_theta
        - model.q
        * sin_phi
        * sin_theta
    )
    dy_per_radius = (
        cos_phi * sin_theta
        + model.q
        * sin_phi
        * cos_theta
    )

    dx_min = (
        pixel_x - 0.5 - model.x0
    )
    dx_max = (
        pixel_x + 0.5 - model.x0
    )
    dy_min = (
        pixel_y - 0.5 - model.y0
    )
    dy_max = (
        pixel_y + 0.5 - model.y0
    )

    lower = 0.0
    upper = np.inf

    for coordinate, minimum, maximum in (
        (
            dx_per_radius,
            dx_min,
            dx_max,
        ),
        (
            dy_per_radius,
            dy_min,
            dy_max,
        ),
    ):
        if (
            abs(coordinate)
            < _DIRECTION_EPSILON
        ):
            if not (
                minimum
                <= 0.0
                <= maximum
            ):
                return None

            continue

        first = minimum / coordinate
        second = maximum / coordinate

        entry = min(
            first,
            second,
        )
        exit_ = max(
            first,
            second,
        )

        lower = max(
            lower,
            entry,
        )
        upper = min(
            upper,
            exit_,
        )

        if upper <= lower:
            return None

    lower = max(
        lower,
        0.0,
    )

    if (
        not np.isfinite(upper)
        or upper <= lower
    ):
        return None

    return (
        float(lower),
        float(upper),
    )


def _semianalytic_pixel_integral(
    model: SersicModel,
    *,
    pixel_x: float,
    pixel_y: float,
    epsabs: float = 1.0e-13,
    epsrel: float = 1.0e-12,
) -> float:
    r"""Return the semi-analytic Sérsic integral over any detector pixel.

    Elliptical polar coordinates reduce the two-dimensional Sérsic integral
    to a one-dimensional angular quadrature. For each angular direction,
    the detector pixel intersects the ray over a radial interval
    ``[R_in(phi), R_out(phi)]``. The radial Sérsic integral is evaluated
    analytically with the regularized lower incomplete gamma function.
    """
    if (
        not np.isfinite(epsabs)
        or epsabs < 0.0
    ):
        raise ValueError(
            "epsabs must be finite and non-negative."
        )

    if (
        not np.isfinite(epsrel)
        or epsrel < 0.0
    ):
        raise ValueError(
            "epsrel must be finite and non-negative."
        )

    if epsabs == 0.0 and epsrel == 0.0:
        raise ValueError(
            "At least one integration tolerance must be positive."
        )

    bn = model.bn
    gamma_shape = 2.0 * model.n
    inverse_n = 1.0 / model.n

    def cumulative_flux_fraction(
        radius: float,
    ) -> float:
        if radius <= 0.0:
            return 0.0

        argument = (
            bn
            * (radius / model.re)
            ** inverse_n
        )

        return float(
            gammainc(
                gamma_shape,
                argument,
            )
        )

    def angular_integrand(
        phi: float,
    ) -> float:
        interval = _ray_pixel_interval(
            model,
            phi,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
        )

        if interval is None:
            return 0.0

        radius_in, radius_out = interval

        return (
            cumulative_flux_fraction(
                radius_out
            )
            - cumulative_flux_fraction(
                radius_in
            )
        )

    boundaries = [
        0.0,
        *[
            phi
            for phi in _pixel_angular_breakpoints(
                model,
                pixel_x=pixel_x,
                pixel_y=pixel_y,
            )
            if 0.0 < phi < _TWO_PI
        ],
        _TWO_PI,
    ]

    boundaries = sorted(
        set(boundaries)
    )

    angular_integral = 0.0

    for lower, upper in zip(
        boundaries[:-1],
        boundaries[1:],
        strict=True,
    ):
        midpoint = (
            0.5 * (lower + upper)
        )

        if _ray_pixel_interval(
            model,
            midpoint,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
        ) is None:
            continue

        value, _ = quad(
            angular_integrand,
            lower,
            upper,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=100,
        )

        angular_integral += value

    return float(
        model.flux
        * angular_integral
        / _TWO_PI
    )


def _semianalytic_central_pixel_integral(
    model: SersicModel,
    *,
    pixel_x: float,
    pixel_y: float,
    epsabs: float = 1.0e-13,
    epsrel: float = 1.0e-12,
) -> float:
    """Return the semi-analytic integral of the pixel containing the centre."""
    tolerance = (
        8.0
        * np.finfo(float).eps
    )

    if not (
        pixel_x - 0.5 - tolerance
        <= model.x0
        <= pixel_x + 0.5 + tolerance
        and pixel_y - 0.5 - tolerance
        <= model.y0
        <= pixel_y + 0.5 + tolerance
    ):
        raise ValueError(
            "Semi-analytic central integration requires a pixel "
            "containing the Sersic centre."
        )

    return _semianalytic_pixel_integral(
        model,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        epsabs=epsabs,
        epsrel=epsrel,
    )


def _ring_pixels(
    shape: tuple[int, int],
    *,
    central_x: int,
    central_y: int,
    half_width: int,
) -> tuple[tuple[int, int], ...]:
    """Return in-bounds pixels on one Chebyshev ring."""
    xmin = max(
        0,
        central_x - half_width,
    )
    xmax = min(
        shape[1] - 1,
        central_x + half_width,
    )
    ymin = max(
        0,
        central_y - half_width,
    )
    ymax = min(
        shape[0] - 1,
        central_y + half_width,
    )

    if xmin > xmax or ymin > ymax:
        return ()

    pixels: list[tuple[int, int]] = []

    for local_y in range(
        ymin,
        ymax + 1,
    ):
        for local_x in range(
            xmin,
            xmax + 1,
        ):
            distance = max(
                abs(local_x - central_x),
                abs(local_y - central_y),
            )

            if distance == half_width:
                pixels.append(
                    (
                        local_y,
                        local_x,
                    )
                )

    return tuple(pixels)


def _apply_automatic_semianalytic_region(
    model: SersicModel,
    image: FloatArray,
    *,
    origin: tuple[float, float],
    unit_flux_pixel_tolerance: float,
    guard_rings: int,
    max_half_width: int,
    epsabs: float,
    epsrel: float,
) -> tuple[
    FloatArray,
    int,
    int,
    tuple[float, ...],
]:
    """Replace numerically difficult inner rings by semi-analytic integrals."""
    if (
        not np.isfinite(
            unit_flux_pixel_tolerance
        )
        or unit_flux_pixel_tolerance <= 0.0
    ):
        raise ValueError(
            "unit_flux_pixel_tolerance must be finite and positive."
        )

    if (
        not isinstance(
            guard_rings,
            (int, np.integer),
        )
        or guard_rings <= 0
    ):
        raise ValueError(
            "guard_rings must be a positive integer."
        )

    if (
        not isinstance(
            max_half_width,
            (int, np.integer),
        )
        or max_half_width < 0
    ):
        raise ValueError(
            "max_half_width must be a non-negative integer."
        )

    flux_scale = abs(
        float(model.flux)
    )

    if (
        not np.isfinite(flux_scale)
        or flux_scale <= 0.0
    ):
        raise ValueError(
            "Automatic Sérsic refinement requires finite positive flux."
        )

    y_origin, x_origin = origin

    central_x = int(
        np.floor(
            model.x0
            - x_origin
            + 0.5
        )
    )
    central_y = int(
        np.floor(
            model.y0
            - y_origin
            + 0.5
        )
    )

    output = np.asarray(
        image,
        dtype=float,
    ).copy()

    consecutive_converged_rings = 0
    evaluated_pixels = 0
    evaluated_ring_errors: list[float] = []

    for half_width in range(
        max_half_width + 1
    ):
        pixels = _ring_pixels(
            output.shape,
            central_x=central_x,
            central_y=central_y,
            half_width=half_width,
        )

        if not pixels:
            continue

        ring_max_error = 0.0

        for local_y, local_x in pixels:
            pixel_x = (
                x_origin
                + local_x
            )
            pixel_y = (
                y_origin
                + local_y
            )

            semi = _semianalytic_pixel_integral(
                model,
                pixel_x=float(pixel_x),
                pixel_y=float(pixel_y),
                epsabs=epsabs,
                epsrel=epsrel,
            )

            midpoint = float(
                output[
                    local_y,
                    local_x,
                ]
            )

            normalized_error = (
                abs(semi - midpoint)
                / flux_scale
            )

            ring_max_error = max(
                ring_max_error,
                normalized_error,
            )

            output[
                local_y,
                local_x,
            ] = semi

        evaluated_pixels += len(pixels)
        evaluated_ring_errors.append(
            ring_max_error
        )

        if (
            ring_max_error
            <= unit_flux_pixel_tolerance
        ):
            consecutive_converged_rings += 1
        else:
            consecutive_converged_rings = 0

        if (
            consecutive_converged_rings
            >= guard_rings
        ):
            return (
                output,
                half_width,
                evaluated_pixels,
                tuple(
                    evaluated_ring_errors
                ),
            )

        if (
            evaluated_pixels
            == output.size
        ):
            return (
                output,
                half_width,
                evaluated_pixels,
                tuple(
                    evaluated_ring_errors
                ),
            )

    raise RuntimeError(
        "Automatic Sérsic semi-analytic refinement did not converge "
        f"within max_half_width={max_half_width}."
    )


def render_sersic_hybrid(
    model: SersicModel,
    shape: tuple[int, int],
    *,
    unit_flux_pixel_tolerance: float,
    origin: tuple[float, float] = (0.0, 0.0),
    guard_rings: int = 2,
    max_half_width: int = 16,
    epsabs: float = 1.0e-13,
    epsrel: float = 1.0e-12,
) -> FloatArray:
    """Render a Sérsic profile with automatic semi-analytic refinement.

    The complete image is initially evaluated by midpoint integration.
    Successive Chebyshev rings around the Sérsic centre are then evaluated
    semi-analytically. Refinement stops after ``guard_rings`` consecutive
    rings for which the maximum midpoint error, normalized by total model
    flux, is no larger than ``unit_flux_pixel_tolerance``.

    The tolerance is intentionally explicit while this renderer remains
    experimental. It is an intrinsic rendering criterion and does not by
    itself guarantee a particular likelihood or posterior error.
    """
    model._validate_shape(shape)

    image = model.render(
        shape=shape,
        origin=origin,
        renderer="fixed",
        integration_order=1,
    )

    (
        image,
        _,
        _,
        _,
    ) = _apply_automatic_semianalytic_region(
        model,
        image,
        origin=origin,
        unit_flux_pixel_tolerance=(
            unit_flux_pixel_tolerance
        ),
        guard_rings=guard_rings,
        max_half_width=max_half_width,
        epsabs=epsabs,
        epsrel=epsrel,
    )

    return image
