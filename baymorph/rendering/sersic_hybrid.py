"""Hybrid semi-analytic rendering utilities for Sérsic models."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.special import gammainc, gammaln

if TYPE_CHECKING:
    from baymorph.models.sersic import SersicModel

FloatArray = NDArray[np.float64]

_TWO_PI = 2.0 * np.pi
_DIRECTION_EPSILON = 1.0e-15
_ANGULAR_BREAKPOINT_ATOL = (
    1024.0 * np.finfo(float).eps * _TWO_PI
)

DEFAULT_OUTER_INTEGRATION_ORDER = 4
DEFAULT_UNIT_FLUX_PIXEL_TOLERANCE = 1.0e-4
DEFAULT_GUARD_RINGS = 2
DEFAULT_MAX_HALF_WIDTH = 16
DEFAULT_EPSABS = 1.0e-13
DEFAULT_EPSREL = 1.0e-12
DEFAULT_CUSP_HALF_WIDTH = 22


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

    This routine is retained as an independent numerical validation
    reference. Production hybrid rendering uses the semi-analytic
    pixel integral.
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
    while floating-point arithmetic produces slightly different ``atan2``
    values. Such duplicate directions would create artificial, extremely
    narrow angular integration intervals.

    Angles within ``atol`` are treated as one geometric direction. The
    periodic equivalence between zero and two pi is handled explicitly.
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
    epsabs: float = DEFAULT_EPSABS,
    epsrel: float = DEFAULT_EPSREL,
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
    epsabs: float = DEFAULT_EPSABS,
    epsrel: float = DEFAULT_EPSREL,
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

            numerical = float(
                output[
                    local_y,
                    local_x,
                ]
            )

            normalized_error = (
                abs(semi - numerical)
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


def _render_outer_fixed_fast(
    model: SersicModel,
    shape: tuple[int, int],
    *,
    origin: tuple[float, float],
    integration_order: int,
) -> FloatArray:
    """Render the smooth outer image with fixed quadrature.

    This production fast path preserves the same Gauss-Legendre pixel
    integration rule used by the generic fixed renderer while avoiding
    repeated full-image geometry and Sérsic normalization work.

    The generic fixed renderer remains independent and serves as the
    numerical validation reference.
    """
    nodes, weights = _legendre_rule(
        integration_order
    )

    ny, nx = shape
    y_origin, x_origin = origin

    pixel_x = (
        np.arange(
            nx,
            dtype=float,
        )[None, :]
        + x_origin
    )
    pixel_y = (
        np.arange(
            ny,
            dtype=float,
        )[:, None]
        + y_origin
    )

    n = float(model.n)
    re = float(model.re)
    q = float(model.q)
    bn = float(model.bn)

    log_normalization = (
        np.log(2.0 * np.pi)
        + np.log(q)
        + 2.0 * np.log(re)
        + np.log(n)
        + gammaln(2.0 * n)
        - 2.0 * n * np.log(bn)
    )

    log_central_intensity = float(
        np.log(model.flux)
        - log_normalization
    )

    cos_theta = float(
        np.cos(model.theta)
    )
    sin_theta = float(
        np.sin(model.theta)
    )

    dx = pixel_x - model.x0
    dy = pixel_y - model.y0

    x_rotated_center = (
        dx * cos_theta
        + dy * sin_theta
    )
    y_rotated_center = (
        -dx * sin_theta
        + dy * cos_theta
    )

    inverse_q_squared = 1.0 / (q * q)
    inverse_re_squared = 1.0 / (re * re)
    radial_exponent = 0.5 / n

    image = np.zeros(
        shape,
        dtype=float,
    )

    for xi, wx in zip(
        nodes,
        weights,
        strict=True,
    ):
        for yi, wy in zip(
            nodes,
            weights,
            strict=True,
        ):
            x_rotated = (
                x_rotated_center
                + 0.5
                * (
                    xi * cos_theta
                    + yi * sin_theta
                )
            )

            y_rotated = (
                y_rotated_center
                + 0.5
                * (
                    -xi * sin_theta
                    + yi * cos_theta
                )
            )

            scaled_radius_squared = (
                x_rotated * x_rotated
                + y_rotated
                * y_rotated
                * inverse_q_squared
            ) * inverse_re_squared

            image += (
                wx
                * wy
                * np.exp(
                    log_central_intensity
                    - bn
                    * scaled_radius_squared
                    ** radial_exponent
                )
            )

    return 0.25 * image


def render_sersic_hybrid(
    model: SersicModel,
    shape: tuple[int, int],
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    outer_integration_order: int = DEFAULT_OUTER_INTEGRATION_ORDER,
    unit_flux_pixel_tolerance: float = DEFAULT_UNIT_FLUX_PIXEL_TOLERANCE,
    guard_rings: int = DEFAULT_GUARD_RINGS,
    max_half_width: int = DEFAULT_MAX_HALF_WIDTH,
    epsabs: float = DEFAULT_EPSABS,
    epsrel: float = DEFAULT_EPSREL,
) -> FloatArray:
    """Render a Sérsic profile with validated hybrid integration.

    The smooth outer image is evaluated with fixed Gauss-Legendre quadrature.
    Successive Chebyshev rings around the Sérsic centre are then compared with
    semi-analytic detector-pixel integrals. Evaluated inner pixels are replaced
    by their semi-analytic values and refinement stops after the requested
    number of consecutive converged guard rings.

    The default numerical controls correspond to the validated BayMorph
    Sérsic rendering rule.
    """
    model._validate_shape(shape)
    model._validate_integration_order(
        outer_integration_order
    )

    image = _render_outer_fixed_fast(
        model,
        shape,
        origin=origin,
        integration_order=outer_integration_order,
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


def _render_corrected_point_samples(
    model: SersicModel,
    shape: tuple[int, int],
    *,
    origin: tuple[float, float],
) -> FloatArray:
    """Return cell means from point samples corrected to sixth order.

    The profile is sampled at the cell centres of ``shape`` extended by two
    cells on every side, the midpoint rule of :func:`_render_outer_fixed_fast`,
    and filtered along each axis with ``[-17, 308, 5178, 308, -17] / 5760``.
    The filter adds the second- and fourth-order terms of the cell mean, so
    the result is exact for quintic profiles and leaves an error of order
    ``h**6`` in the cell size ``h``: accurate where the profile is smooth on
    the scale of a cell, away from its centre.
    """
    samples = _render_outer_fixed_fast(
        model,
        (shape[0] + 4, shape[1] + 4),
        origin=(origin[0] - 2.0, origin[1] - 2.0),
        integration_order=1,
    )
    weights = np.array([-17.0, 308.0, 5178.0, 308.0, -17.0]) / 5760.0
    rows = sum(
        weight * samples[offset:offset + shape[0]]
        for offset, weight in enumerate(weights)
    )

    return sum(
        weight * rows[:, offset:offset + shape[1]]
        for offset, weight in enumerate(weights)
    )


def render_sersic_cusp_hybrid(
    model: SersicModel,
    shape: tuple[int, int],
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    cusp_half_width: int = DEFAULT_CUSP_HALF_WIDTH,
    outer_integration_order: int = DEFAULT_OUTER_INTEGRATION_ORDER,
) -> FloatArray:
    """Render a Sérsic profile with the hybrid rule near its centre only.

    Cells within ``cusp_half_width`` cells of the cell containing the centre
    (Chebyshev distance) take the values of :func:`render_sersic_hybrid`,
    bit for bit when ``origin`` holds whole numbers; the other cells are the
    corrected point samples of :func:`_render_corrected_point_samples`. The
    half-width exceeds the rings the semi-analytic refinement may use, so
    its failure to converge still raises.

    This is the cell rule of the effective-PSF rendering of AD-046 on its
    finer grid, where every cell away from the centre is small against the
    scale on which the profile changes.
    """
    model._validate_shape(shape)

    if (
        not isinstance(cusp_half_width, (int, np.integer))
        or cusp_half_width <= DEFAULT_MAX_HALF_WIDTH
    ):
        raise ValueError(
            "cusp_half_width must be an integer larger than "
            f"{DEFAULT_MAX_HALF_WIDTH}, the rings of the semi-analytic "
            "refinement."
        )

    image = _render_corrected_point_samples(
        model,
        shape,
        origin=origin,
    )
    y_origin, x_origin = origin
    central_x = int(np.floor(model.x0 - x_origin + 0.5))
    central_y = int(np.floor(model.y0 - y_origin + 0.5))
    top = max(central_y - cusp_half_width, 0)
    bottom = min(central_y + cusp_half_width + 1, shape[0])
    left = max(central_x - cusp_half_width, 0)
    right = min(central_x + cusp_half_width + 1, shape[1])

    if top >= bottom or left >= right:
        return image

    image[top:bottom, left:right] = render_sersic_hybrid(
        model,
        (bottom - top, right - left),
        origin=(y_origin + top, x_origin + left),
        outer_integration_order=outer_integration_order,
    )

    return image
