"""Experimental high-throughput hybrid rendering for Sérsic models."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from baymorph.models.sersic import SersicModel

FloatArray = NDArray[np.float64]


@lru_cache(maxsize=None)
def _legendre_rule(
    order: int,
) -> tuple[FloatArray, FloatArray]:
    """Return a cached Gauss-Legendre quadrature rule."""
    nodes, weights = np.polynomial.legendre.leggauss(order)

    nodes = np.asarray(nodes, dtype=float)
    weights = np.asarray(weights, dtype=float)

    nodes.setflags(write=False)
    weights.setflags(write=False)

    return nodes, weights


def _validate_positive_integer(
    value: int,
    *,
    name: str,
) -> None:
    """Require a strictly positive integer."""
    if (
        not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError(
            f"{name} must be a positive integer."
        )


def _validate_configuration(
    *,
    base_order: int,
    refinement_order: int,
    refinement_radius: float,
    central_min_order: int,
    central_max_order: int,
    central_rtol: float,
    central_atol: float,
) -> None:
    """Validate hybrid-renderer numerical controls."""
    _validate_positive_integer(
        base_order,
        name="base_order",
    )
    _validate_positive_integer(
        refinement_order,
        name="refinement_order",
    )
    _validate_positive_integer(
        central_min_order,
        name="central_min_order",
    )
    _validate_positive_integer(
        central_max_order,
        name="central_max_order",
    )

    if central_max_order < central_min_order:
        raise ValueError(
            "central_max_order must be greater than or equal to "
            "central_min_order."
        )

    if (
        not np.isfinite(refinement_radius)
        or refinement_radius < 0.0
    ):
        raise ValueError(
            "refinement_radius must be finite and non-negative."
        )

    if (
        not np.isfinite(central_rtol)
        or central_rtol < 0.0
    ):
        raise ValueError(
            "central_rtol must be finite and non-negative."
        )

    if (
        not np.isfinite(central_atol)
        or central_atol < 0.0
    ):
        raise ValueError(
            "central_atol must be finite and non-negative."
        )


def _central_pixel(
    model: SersicModel,
    *,
    origin: tuple[float, float],
) -> tuple[int, int, float, float]:
    """Return the local index and global centre of the central pixel."""
    y_origin, x_origin = origin

    local_x = model.x0 - x_origin
    local_y = model.y0 - y_origin

    index_x = int(
        np.floor(local_x + 0.5)
    )
    index_y = int(
        np.floor(local_y + 0.5)
    )

    pixel_x = x_origin + index_x
    pixel_y = y_origin + index_y

    return (
        index_y,
        index_x,
        float(pixel_y),
        float(pixel_x),
    )


def _split_central_pixel_integral(
    model: SersicModel,
    *,
    pixel_x: float,
    pixel_y: float,
    integration_order: int,
) -> float:
    """Integrate the central pixel after splitting at the Sérsic centre."""
    _validate_positive_integer(
        integration_order,
        name="integration_order",
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

    nodes, weights = _legendre_rule(
        integration_order
    )

    weight_2d = (
        weights[:, None]
        * weights[None, :]
    )

    bn = model.bn
    log_i0 = model.log_central_intensity

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
                log_i0
                - bn
                * (radius / model.re) ** (1.0 / model.n)
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


def _converged_central_pixel_integral(
    model: SersicModel,
    *,
    pixel_x: float,
    pixel_y: float,
    minimum_order: int,
    maximum_order: int,
    rtol: float,
    atol: float,
) -> float:
    """Return an order-converged split central-pixel integral."""
    order = int(minimum_order)

    previous = _split_central_pixel_integral(
        model,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        integration_order=order,
    )

    if order == maximum_order:
        return previous

    while order < maximum_order:
        next_order = min(
            2 * order,
            maximum_order,
        )

        current = _split_central_pixel_integral(
            model,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            integration_order=next_order,
        )

        tolerance = (
            atol
            + rtol * abs(current)
        )

        if abs(current - previous) <= tolerance:
            return current

        previous = current
        order = next_order

    raise RuntimeError(
        "Split central-pixel Sérsic quadrature did not converge "
        f"by integration order {maximum_order}."
    )


def render_sersic_hybrid(
    model: SersicModel,
    shape: tuple[int, int],
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    base_order: int = 1,
    refinement_order: int = 8,
    refinement_radius: float = 16.0,
    central_min_order: int = 32,
    central_max_order: int = 512,
    central_rtol: float = 1.0e-6,
    central_atol: float = 0.0,
) -> FloatArray:
    """Render a Sérsic profile with localized numerical refinement.

    The renderer combines inexpensive fixed quadrature over the complete
    footprint with higher-order integration in a square region surrounding
    the Sérsic centre. The detector pixel containing the mathematical centre
    is treated separately by splitting its integration domain at ``x0`` and
    ``y0`` and increasing its quadrature order until convergence.

    This renderer is experimental. Its numerical controls are explicit and
    must be validated over the intended scientific parameter domain before it
    replaces the uniform fixed-order renderer in production inference.
    """
    _validate_configuration(
        base_order=base_order,
        refinement_order=refinement_order,
        refinement_radius=refinement_radius,
        central_min_order=central_min_order,
        central_max_order=central_max_order,
        central_rtol=central_rtol,
        central_atol=central_atol,
    )

    image = model.render(
        shape=shape,
        origin=origin,
        renderer="fixed",
        integration_order=base_order,
    )

    y_origin, x_origin = origin

    local_center_x = model.x0 - x_origin
    local_center_y = model.y0 - y_origin

    x_min = max(
        0,
        int(
            np.floor(
                local_center_x
                - refinement_radius
            )
        ),
    )
    x_max = min(
        shape[1],
        int(
            np.ceil(
                local_center_x
                + refinement_radius
            )
        )
        + 1,
    )
    y_min = max(
        0,
        int(
            np.floor(
                local_center_y
                - refinement_radius
            )
        ),
    )
    y_max = min(
        shape[0],
        int(
            np.ceil(
                local_center_y
                + refinement_radius
            )
        )
        + 1,
    )

    if (
        x_min < x_max
        and y_min < y_max
    ):
        local_y, local_x = np.mgrid[
            y_min:y_max,
            x_min:x_max,
        ]

        pixel_x = (
            np.asarray(
                local_x,
                dtype=float,
            )
            + x_origin
        )
        pixel_y = (
            np.asarray(
                local_y,
                dtype=float,
            )
            + y_origin
        )

        image[
            y_min:y_max,
            x_min:x_max,
        ] = model._integrate_pixel_fixed(
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            integration_order=refinement_order,
        )

    (
        central_y,
        central_x,
        pixel_y,
        pixel_x,
    ) = _central_pixel(
        model,
        origin=origin,
    )

    if (
        0 <= central_y < shape[0]
        and 0 <= central_x < shape[1]
    ):
        image[
            central_y,
            central_x,
        ] = _converged_central_pixel_integral(
            model,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            minimum_order=central_min_order,
            maximum_order=central_max_order,
            rtol=central_rtol,
            atol=central_atol,
        )

    return np.asarray(
        image,
        dtype=float,
    )
