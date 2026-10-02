"""Generic cell rules for the profile components of AD-048.

A profile component supplies its surface brightness as a vectorized function
``intensity(x, y)`` of global pixel coordinates. The functions here turn it
into an image on a grid whose cell ``[j, i]`` has unit size and is centred
at ``(origin[1] + i, origin[0] + j)``. The renderers mirror those of the
Sérsic profile:

``point``
    the profile at every cell centre: the outer image of the effective-PSF
    rendering of AD-046;
``fixed``
    Gauss-Legendre quadrature of a fixed order over every cell;
``hybrid``
    the fixed rule, with adaptive integration about the centre and on the
    cells a component declares non-smooth: the rule for PSFs without the
    pixel response;
``hybrid_cusp``
    point samples corrected to sixth order, with adaptive integration about
    the centre and on the declared cells: the fine-cell rule of the
    effective-PSF rendering;
``adaptive``
    adaptive integration of every cell, for validation.

Adaptive integration splits a cell into four until the sum of the four
estimates agrees with the parent estimate within the tolerance, and raises
if it does not converge. A cell that holds an integrable singularity, such as
a power-law cusp, keeps the same relative error at every level, because the
profile looks the same at every scale; its absolute error falls with the
cell size, so components state the absolute tolerance as a fraction of their
total flux, ``FLUX_ATOL``, and the depth allows for it. The adaptive block
about the centre widens until no cell of its outer ring differs from the
rule used outside it by more than the tolerance, so the boundary of the
block is itself checked.

The test of agreement cannot see a part of a cell that no node samples at
two successive levels. Where a profile ends at the edge of a finite support,
a thin sliver of it inside a cell, or of empty plane beyond it, can escape
every node, and the cell is accepted with that part missing. A component
with such an edge declares, through ``edges``, the cells of a given size the
edge may cross; those are split, whatever their agreement, until the
component deems them small enough.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation

from baymorph.rendering.cell_means import sixth_order_cell_means
from baymorph.rendering.sersic_hybrid import DEFAULT_CUSP_HALF_WIDTH

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
Intensity = Callable[[FloatArray, FloatArray], FloatArray]
NonSmoothCells = Callable[[FloatArray, FloatArray], BoolArray | None]
EdgeCells = Callable[[FloatArray, FloatArray, float], BoolArray | None]

RENDERERS = ("hybrid", "hybrid_cusp", "fixed", "adaptive", "point")
DEFAULT_INTEGRATION_ORDER = 4
DEFAULT_RTOL = 1.0e-8
DEFAULT_ATOL = 1.0e-12
FLUX_ATOL = 1.0e-10
DEFAULT_MAX_DEPTH = 30
DEFAULT_CORE_HALF_WIDTH = 2
MAX_CORE_HALF_WIDTH = 32
MAX_CUSP_HALF_WIDTH = 4 * DEFAULT_CUSP_HALF_WIDTH

STENCIL_REACH = 2
"""Cells on each side that the sixth-order filter of one cell reads."""

_CHILD_X = np.array([-0.5, 0.5, -0.5, 0.5])
_CHILD_Y = np.array([-0.5, -0.5, 0.5, 0.5])


def validate_shape(shape: tuple[int, int]) -> None:
    """Require a two-dimensional shape of positive integers."""
    if (
        len(shape) != 2
        or any(
            not isinstance(size, (int, np.integer)) or size <= 0
            for size in shape
        )
    ):
        raise ValueError(
            "Render shape must contain two positive integers."
        )


def _validate_controls(
    *,
    integration_order: int,
    rtol: float,
    atol: float,
    max_depth: int,
) -> None:
    """Validate the quadrature order, tolerances and refinement depth."""
    if (
        not isinstance(integration_order, (int, np.integer))
        or integration_order <= 0
    ):
        raise ValueError(
            "integration_order must be a positive integer."
        )

    if not np.isfinite(rtol) or rtol < 0.0:
        raise ValueError("rtol must be finite and non-negative.")

    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be finite and non-negative.")

    if rtol == 0.0 and atol == 0.0:
        raise ValueError("rtol and atol cannot both be zero.")

    if not isinstance(max_depth, (int, np.integer)) or max_depth < 0:
        raise ValueError("max_depth must be a non-negative integer.")


def _finite(values: FloatArray) -> FloatArray:
    """Return ``values`` as floats, requiring them to be finite."""
    values = np.asarray(values, dtype=float)

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "The profile is not finite where it was evaluated."
        )

    return values


def _axes(
    shape: tuple[int, int],
    origin: tuple[float, float],
) -> tuple[FloatArray, FloatArray]:
    """Return the cell-centre coordinates as a row and a column."""
    y_origin, x_origin = origin

    return (
        (np.arange(shape[1], dtype=float) + x_origin)[None, :],
        (np.arange(shape[0], dtype=float) + y_origin)[:, None],
    )


def point_samples(
    intensity: Intensity,
    shape: tuple[int, int],
    origin: tuple[float, float] = (0.0, 0.0),
) -> FloatArray:
    """Return the profile at every cell centre."""
    x, y = _axes(shape, origin)

    return np.broadcast_to(_finite(intensity(x, y)), shape).copy()


def gauss_legendre_cells(
    intensity: Intensity,
    x: FloatArray,
    y: FloatArray,
    half: float,
    integration_order: int,
) -> FloatArray:
    """Return the integrals over square cells of half-width ``half``.

    ``x`` and ``y`` are the cell centres; they broadcast against each other.
    """
    nodes, weights = np.polynomial.legendre.leggauss(integration_order)
    total = 0.0

    for xi, wx in zip(nodes, weights, strict=True):
        for yi, wy in zip(nodes, weights, strict=True):
            total = total + wx * wy * intensity(x + half * xi, y + half * yi)

    return _finite(total) * (half * half)


def fixed_cells(
    intensity: Intensity,
    shape: tuple[int, int],
    origin: tuple[float, float] = (0.0, 0.0),
    *,
    integration_order: int = DEFAULT_INTEGRATION_ORDER,
) -> FloatArray:
    """Return the cell integrals by fixed Gauss-Legendre quadrature."""
    x, y = _axes(shape, origin)

    return np.broadcast_to(
        gauss_legendre_cells(intensity, x, y, 0.5, integration_order),
        shape,
    ).copy()


def corrected_point_samples(
    intensity: Intensity,
    shape: tuple[int, int],
    origin: tuple[float, float] = (0.0, 0.0),
) -> FloatArray:
    """Return cell means from point samples corrected to sixth order."""
    samples = point_samples(
        intensity,
        (shape[0] + 4, shape[1] + 4),
        (origin[0] - 2.0, origin[1] - 2.0),
    )

    return np.asarray(sixth_order_cell_means(samples, shape), dtype=float)


def refine_cells(
    intensity: Intensity,
    x: FloatArray,
    y: FloatArray,
    estimate: FloatArray,
    *,
    integration_order: int = DEFAULT_INTEGRATION_ORDER,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    max_depth: int = DEFAULT_MAX_DEPTH,
    edges: EdgeCells | None = None,
) -> FloatArray:
    """Return adaptive integrals of the unit cells centred at ``x`` and ``y``.

    ``estimate`` holds a first estimate of each cell by the same quadrature.
    A cell is split into four children; it is accepted when the sum of their
    estimates agrees with its own within ``max(atol, rtol * |value|)``, and
    otherwise each child is split in turn, down to ``max_depth`` levels.

    ``edges``, when given, is called with the centres and the half-width of
    the children of every level and returns the children that the edge of a
    finite support may cross, or ``None`` once the children are small enough.
    A cell with such a child is not accepted, whatever the agreement.

    Raises
    ------
    ValueError
        If some cell has not converged at ``max_depth``.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    estimate = np.asarray(estimate, dtype=float).ravel()
    result = np.zeros(x.size)
    owner = np.arange(x.size)
    half = 0.5

    for depth in range(max_depth + 1):
        child_x = (x[:, None] + half * _CHILD_X).ravel()
        child_y = (y[:, None] + half * _CHILD_Y).ravel()
        half = 0.5 * half
        children = gauss_legendre_cells(
            intensity,
            child_x,
            child_y,
            half,
            integration_order,
        ).reshape(-1, 4)
        fine = children.sum(axis=1)
        accepted = np.abs(fine - estimate) <= np.maximum(
            atol,
            rtol * np.abs(fine),
        )

        if edges is not None:
            crossed = edges(child_x, child_y, half)

            if crossed is not None:
                accepted &= ~np.asarray(crossed, dtype=bool).reshape(-1, 4).any(
                    axis=1
                )

        np.add.at(result, owner[accepted], fine[accepted])

        if np.all(accepted):
            return result

        if depth == max_depth:
            raise ValueError(
                "Adaptive cell integration did not converge within "
                f"{max_depth} levels for {int(np.sum(~accepted))} cells; "
                "the profile is not resolved at the requested tolerance."
            )

        pending = ~accepted
        x = child_x.reshape(-1, 4)[pending].ravel()
        y = child_y.reshape(-1, 4)[pending].ravel()
        estimate = children[pending].ravel()
        owner = np.repeat(owner[pending], 4)

    return result


def _refine_mask(
    intensity: Intensity,
    image: FloatArray,
    mask: BoolArray,
    origin: tuple[float, float],
    *,
    integration_order: int,
    rtol: float,
    atol: float,
    max_depth: int,
    edges: EdgeCells | None = None,
) -> FloatArray:
    """Return adaptive integrals of the cells of ``image`` within ``mask``."""
    rows, columns = np.nonzero(mask)
    x = columns.astype(float) + origin[1]
    y = rows.astype(float) + origin[0]
    estimate = gauss_legendre_cells(intensity, x, y, 0.5, integration_order)

    return refine_cells(
        intensity,
        x,
        y,
        estimate,
        integration_order=integration_order,
        rtol=rtol,
        atol=atol,
        max_depth=max_depth,
        edges=edges,
    )


def refine_about_centre(
    intensity: Intensity,
    image: FloatArray,
    origin: tuple[float, float],
    centre: tuple[float, float],
    *,
    half_width: int,
    max_half_width: int,
    integration_order: int,
    rtol: float,
    atol: float,
    max_depth: int,
    exclude: BoolArray | None = None,
    edges: EdgeCells | None = None,
) -> FloatArray:
    """Replace the cells about ``centre`` by adaptive integrals, in place.

    The block holds the cells within ``half_width`` cells of the cell that
    contains the centre (Chebyshev distance). It widens by two cells while
    a cell of its outer ring inside the image differs from its value in
    ``image`` by more than the tolerance. Cells in ``exclude``, which the
    caller integrates adaptively anyway, do not enter that test. ``edges``
    is passed to :func:`refine_cells`.

    Raises
    ------
    ValueError
        If the block would exceed ``max_half_width``.
    """
    shape = image.shape
    x0, y0 = centre
    central_column = int(np.floor(x0 - origin[1] + 0.5))
    central_row = int(np.floor(y0 - origin[0] + 0.5))
    rows, columns = np.indices(shape)
    distance = np.maximum(
        np.abs(rows - central_row),
        np.abs(columns - central_column),
    )
    done = np.zeros(shape, dtype=bool)
    tested = (
        np.ones(shape, dtype=bool)
        if exclude is None
        else ~np.broadcast_to(np.asarray(exclude, dtype=bool), shape)
    )
    width = int(half_width)

    while True:
        block = (distance <= width) & ~done

        if np.any(block):
            refined = _refine_mask(
                intensity,
                image,
                block,
                origin,
                integration_order=integration_order,
                rtol=rtol,
                atol=atol,
                max_depth=max_depth,
                edges=edges,
            )
            ring = (distance[block] == width) & tested[block]
            initial = image[block]
            converged = np.all(
                np.abs(refined[ring] - initial[ring])
                <= np.maximum(atol, rtol * np.abs(refined[ring]))
            )
            image[block] = refined
            done |= block
        else:
            converged = True

        if converged:
            return image

        width += 2

        if width > max_half_width:
            raise ValueError(
                "The profile is not smooth enough for the cell rule beyond "
                f"{max_half_width} cells of its centre."
            )


def render_cells(
    intensity: Intensity,
    shape: tuple[int, int],
    origin: tuple[float, float] = (0.0, 0.0),
    *,
    renderer: str = "hybrid",
    centre: tuple[float, float],
    nonsmooth: NonSmoothCells | None = None,
    edges: EdgeCells | None = None,
    integration_order: int = DEFAULT_INTEGRATION_ORDER,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> FloatArray:
    """Render a profile on a grid of unit cells by one of the cell rules.

    Parameters
    ----------
    intensity : callable
        Surface brightness at global coordinates ``(x, y)``, vectorized.
    shape : tuple of int
        Image shape ``(ny, nx)``.
    origin : tuple of float, default=(0.0, 0.0)
        Global coordinate ``(y, x)`` of the centre of cell ``[0, 0]``.
    renderer : {"hybrid", "hybrid_cusp", "fixed", "adaptive", "point"}
        Cell rule; see the module documentation.
    centre : tuple of float
        Component centre ``(x0, y0)``, where the adaptive block is placed.
    nonsmooth : callable, optional
        Function of the cell centres returning the cells, as a boolean
        array, on which the profile is not smooth and which the hybrid
        rules integrate adaptively.
    edges : callable, optional
        Function of the centres and the half-width of cells returning those
        the edge of a finite support may cross, or ``None`` once they are
        small enough; see :func:`refine_cells`. It applies wherever cells are
        integrated adaptively.
    integration_order, rtol, atol, max_depth
        Gauss-Legendre order, tolerances and depth of adaptive integration.

    Returns
    -------
    ndarray
        Cell integrals, or point samples for ``renderer="point"``.
    """
    validate_shape(shape)
    _validate_controls(
        integration_order=integration_order,
        rtol=rtol,
        atol=atol,
        max_depth=max_depth,
    )

    if renderer not in RENDERERS:
        raise ValueError(
            "renderer must be one of " + ", ".join(repr(name) for name in RENDERERS)
            + "."
        )

    controls = dict(
        integration_order=integration_order,
        rtol=rtol,
        atol=atol,
        max_depth=max_depth,
        edges=edges,
    )

    if renderer == "point":
        return point_samples(intensity, shape, origin)

    if renderer == "fixed":
        return fixed_cells(
            intensity,
            shape,
            origin,
            integration_order=integration_order,
        )

    if renderer == "adaptive":
        image = fixed_cells(
            intensity,
            shape,
            origin,
            integration_order=integration_order,
        )
        image[...] = _refine_mask(
            intensity,
            image,
            np.ones(shape, dtype=bool),
            origin,
            **controls,
        ).reshape(shape)

        return image

    if renderer == "hybrid":
        image = fixed_cells(
            intensity,
            shape,
            origin,
            integration_order=integration_order,
        )
        half_width = DEFAULT_CORE_HALF_WIDTH
        max_half_width = MAX_CORE_HALF_WIDTH
    else:
        image = corrected_point_samples(intensity, shape, origin)
        half_width = DEFAULT_CUSP_HALF_WIDTH
        max_half_width = MAX_CUSP_HALF_WIDTH

    mask = None

    if nonsmooth is not None:
        x, y = _axes(shape, origin)
        declared = nonsmooth(x, y)

        if declared is not None:
            mask = np.broadcast_to(np.asarray(declared, dtype=bool), shape)

            if renderer == "hybrid_cusp":
                # A corrected point sample reads the profile two cells away,
                # so the cells near a declared one are not smooth either.
                mask = binary_dilation(
                    mask,
                    structure=np.ones((3, 3), dtype=bool),
                    iterations=STENCIL_REACH,
                )

    refine_about_centre(
        intensity,
        image,
        origin,
        centre,
        half_width=half_width,
        max_half_width=max_half_width,
        exclude=mask,
        **controls,
    )

    if mask is not None and np.any(mask):
        image[mask] = _refine_mask(
            intensity,
            image,
            mask,
            origin,
            **controls,
        )

    return image
