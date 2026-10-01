"""Rendering through an effective PSF sampled at sub-pixel offsets (AD-046).

An empirical PSF built from detector pixels samples the effective PSF, the
optical PSF convolved with the pixel. The expected value of a pixel is then
the continuous convolution of the intrinsic profile with the effective PSF,
evaluated at the pixel centre. Sampling the profile at pixel centres
approximates that integral badly on a cusp, so the profile is split by a
smooth window centred on the source: within the window it is integrated over
cells ``SUBPIXEL_FACTOR`` times finer and convolved with the effective PSF at
the same sub-pixel offsets, divided by the response of one fine cell;
outside it the caller samples it at pixel centres and convolves it with the
native kernel.

The constants are fixed by AD-046, because they were validated as a whole on
the test bench of the PSF and rendering workstream; they are recorded in
``RENDERING_CONVENTION``, which enters the signature of every PSF that uses
them.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.fft import next_fast_len
from scipy.signal import fftconvolve
from scipy.special import erfc

FloatArray = NDArray[np.float64]

SUBPIXEL_FACTOR = 11
WINDOW_RADIUS = 7.0
WINDOW_WIDTH = 1.0
WINDOW_HALF = 14
CELL_RESPONSE_MARGIN = 4
RENDERING_CONVENTION = (
    "AD-046 effective-PSF rendering: window erfc((r - 7) / sqrt(2)) / 2, "
    "zero beyond 14 px; 11 sub-pixel phases per axis; fine-cell response "
    "divided out"
)


def smooth_window(
    xx: FloatArray,
    yy: FloatArray,
    x: float,
    y: float,
) -> FloatArray:
    """Return the window of AD-046 about ``(x, y)``.

    The window is one minus a cumulative normal in radius, centred at
    ``WINDOW_RADIUS`` with width ``WINDOW_WIDTH``, and zero from
    ``WINDOW_HALF`` px on. One minus the window at the centre, and the
    window at the cut, are both 1.3e-12.
    """
    radius = np.hypot(
        np.asarray(xx, dtype=float) - x,
        np.asarray(yy, dtype=float) - y,
    )
    window = 0.5 * erfc(
        (radius - WINDOW_RADIUS) / (np.sqrt(2.0) * WINDOW_WIDTH)
    )

    return np.where(radius < WINDOW_HALF, window, 0.0)


def interleave(subpixel_kernels: FloatArray) -> FloatArray:
    """Return sub-pixel kernels as one grid finer by the number of phases.

    Element ``[j, i]`` of ``subpixel_kernels`` is the kernel at the phase
    ``((i - c) / f, (j - c) / f)``, ``c = (f - 1) / 2``: its pixel ``[b, a]``
    is the effective PSF at offset ``(a - h, b - h)`` plus the phase. Fine
    pixel ``[f * b + j, f * a + i]`` of the result holds it, so the zero
    offset is the middle pixel.
    """
    factor, _, size, _ = subpixel_kernels.shape

    return subpixel_kernels.transpose(2, 0, 3, 1).reshape(
        size * factor,
        size * factor,
    )


def divide_cell_response(
    fine: FloatArray,
    factor: int,
) -> FloatArray:
    """Return a sub-pixel kernel divided by the response of one fine cell.

    Cell integrals of the profile placed at the cell centres blur the image
    by one fine cell, an error of order ``1 / factor**2`` that does not
    cancel; dividing the kernel by the transfer function of the fine cell,
    ``sinc(u) sinc(v)`` in cycles per fine cell, removes it. The division is
    separable, done on a grid with a margin of ``CELL_RESPONSE_MARGIN``
    native pixels, and the result keeps the sum of the kernel.
    """
    side = fine.shape[0]
    margin = CELL_RESPONSE_MARGIN * factor
    size = next_fast_len(side + 2 * margin)
    transfer = np.sinc(np.fft.fftfreq(size))
    padded = np.zeros((size, size))
    padded[margin:margin + side, margin:margin + side] = fine
    corrected = np.fft.ifft(
        np.fft.fft(padded, axis=0) / transfer[:, None],
        axis=0,
    )
    corrected = np.real(
        np.fft.ifft(np.fft.fft(corrected, axis=1) / transfer[None, :], axis=1)
    )[margin:margin + side, margin:margin + side]

    return corrected * (fine.sum() / corrected.sum())


def window_box(
    x: float,
    y: float,
) -> tuple[int, int]:
    """Return the native pixel nearest ``(x, y)``, the centre of the box.

    The box spans ``2 * WINDOW_HALF + 1`` native pixels about it, so it
    contains the support of the window for any sub-pixel position.
    """
    return int(np.floor(y + 0.5)), int(np.floor(x + 0.5))


def apply_outer_weight(
    image: FloatArray,
    origin: tuple[float, float],
    x: float,
    y: float,
) -> None:
    """Multiply in place the part of ``image`` under the window by one minus it.

    ``image`` holds point samples on pixels whose index ``(0, 0)`` lies at the
    global coordinate ``origin``, ordered ``(y, x)``.
    """
    row, column = window_box(x, y)
    top = row - WINDOW_HALF - int(round(origin[0]))
    left = column - WINDOW_HALF - int(round(origin[1]))
    rows = slice(max(top, 0), max(min(top + 2 * WINDOW_HALF + 1, image.shape[0]), 0))
    columns = slice(
        max(left, 0),
        max(min(left + 2 * WINDOW_HALF + 1, image.shape[1]), 0),
    )

    if rows.start >= rows.stop or columns.start >= columns.stop:
        return

    yy, xx = np.mgrid[rows, columns].astype(float)
    image[rows, columns] *= 1.0 - smooth_window(
        xx + origin[1],
        yy + origin[0],
        x,
        y,
    )


def local_images(
    scaled_images: dict[str, FloatArray],
    fine_kernel: FloatArray,
    x: float,
    y: float,
    shape: tuple[int, int],
    origin: tuple[float, float],
    *,
    factor: int = SUBPIXEL_FACTOR,
) -> dict[str, FloatArray]:
    """Convolve windowed fine images and read them at native pixel centres.

    ``scaled_images`` hold, for each named component, the cell integrals of
    its profile on the fine box of :func:`fine_box_origin`. Each is
    multiplied by the window, convolved with ``fine_kernel``, whose zero
    offset is its middle pixel, and read at the native pixel centres of a
    ``shape`` image whose index ``(0, 0)`` lies at the global coordinate
    ``origin``.
    """
    row, column = window_box(x, y)
    first_row, first_column = fine_box_origin(x, y, factor=factor)
    box = (2 * WINDOW_HALF + 1) * factor
    fine_y, fine_x = np.indices((box, box), dtype=float)
    weight = smooth_window(
        (fine_x + first_column + 0.5) / factor - 0.5,
        (fine_y + first_row + 0.5) / factor - 0.5,
        x,
        y,
    )
    reference = (fine_kernel.shape[0] - 1) // 2 + (factor - 1) // 2
    first = reference % factor
    shift = (first - reference) // factor
    images = {}

    for name, fine in scaled_images.items():
        if fine.shape != (box, box):
            raise ValueError(
                "Fine images must cover the box of the effective-PSF window."
            )

        samples = fftconvolve(fine * weight, fine_kernel, mode="full")[
            first::factor,
            first::factor,
        ]
        count = samples.shape[0]
        top = row - WINDOW_HALF + shift - int(round(origin[0]))
        left = column - WINDOW_HALF + shift - int(round(origin[1]))
        image = np.zeros(shape)
        rows = slice(max(top, 0), min(top + count, shape[0]))
        columns = slice(max(left, 0), min(left + count, shape[1]))

        if rows.start < rows.stop and columns.start < columns.stop:
            image[rows, columns] = samples[
                rows.start - top:rows.stop - top,
                columns.start - left:columns.stop - left,
            ]

        images[name] = image

    return images


def fine_box_origin(
    x: float,
    y: float,
    *,
    factor: int = SUBPIXEL_FACTOR,
) -> tuple[int, int]:
    """Return the global fine index ``(row, column)`` of the box's first cell.

    Fine cell ``J`` of the global fine grid is centred at the native
    coordinate ``(J + 0.5) / factor - 0.5``.
    """
    row, column = window_box(x, y)

    return factor * (row - WINDOW_HALF), factor * (column - WINDOW_HALF)
