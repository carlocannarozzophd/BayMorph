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

The evaluation follows the third milestone of AD-046: the fine cells away
from the centre are point samples corrected to sixth order, and the fine
convolution is computed exactly but only at the native pixel centres. It
agrees with the full evaluation within 1e-9 of the image peak.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.fft import fft, irfft2, next_fast_len, rfft, rfft2
from scipy.special import erfc

from baymorph.rendering.sersic_hybrid import DEFAULT_CUSP_HALF_WIDTH

FloatArray = NDArray[np.float64]

SUBPIXEL_FACTOR = 11
WINDOW_RADIUS = 7.0
WINDOW_WIDTH = 1.0
WINDOW_HALF = 14
CELL_RESPONSE_MARGIN = 4
RENDERING_CONVENTION = (
    "AD-046 effective-PSF rendering: window erfc((r - 7) / sqrt(2)) / 2, "
    "zero beyond 14 px; 11 sub-pixel phases per axis; fine-cell response "
    "divided out; fine cells by the hybrid rule within "
    f"{DEFAULT_CUSP_HALF_WIDTH} cells of the centre and by point samples "
    "filtered [-17, 308, 5178, 308, -17] / 5760 beyond; fine convolution "
    "evaluated at the native centres only"
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


class PolyphaseConvolution:
    """Convolve a fine box with a fine kernel, read at native pixel centres.

    ``fine_kernel`` is the corrected fine kernel of an empirical PSF,
    ``factor`` cells per native pixel on a square of an odd number of native
    pixels, its zero offset at the middle cell. Fine cell ``a`` of a native
    box pixel pairs with the sub-pixel kernel at the phase opposite to the
    cell's position, so the samples of the fine convolution at the native
    centres are exactly a sum, over the ``factor**2`` cell positions, of
    native convolutions of the box pixels with the sub-pixel kernels. The
    kernel transforms are computed once, here; the transforms of the box
    skip its empty rows.
    """

    def __init__(
        self,
        fine_kernel: FloatArray,
        *,
        factor: int = SUBPIXEL_FACTOR,
        box: int = 2 * WINDOW_HALF + 1,
    ) -> None:
        kernel = np.asarray(fine_kernel, dtype=float)
        side = kernel.shape[0]

        if (
            kernel.ndim != 2
            or kernel.shape[1] != side
            or side % factor != 0
            or (side // factor) % 2 != 1
        ):
            raise ValueError(
                "The fine kernel must be square, with an odd number of "
                "native pixels of factor cells each."
            )

        size = side // factor
        # Element [ay, ax, by, bx]: the kernel paired with cell (ay, ax) of a
        # box pixel, at native offset (bx - size // 2, by - size // 2).
        by_cell = kernel.reshape(size, factor, size, factor).transpose(
            1,
            3,
            0,
            2,
        )[::-1, ::-1]

        self.factor = factor
        self.box = box
        self.size = size
        self.samples = box + size - 1
        self._length = next_fast_len(self.samples, real=True)
        self._spectra = rfft2(
            by_cell,
            s=(self._length, self._length),
            axes=(-2, -1),
        ).reshape(factor * factor, -1)

    def __call__(self, fine: FloatArray) -> FloatArray:
        """Return the native samples of the convolution of ``fine``.

        ``fine`` covers ``box`` native pixels of ``factor`` cells per axis.
        Sample ``[i, j]`` lies ``size // 2`` native pixels before the first
        box pixel plus ``(i, j)``.
        """
        factor = self.factor
        box = self.box
        length = self._length

        if fine.shape != (box * factor, box * factor):
            raise ValueError(
                "Fine images must cover the box of the effective-PSF window."
            )

        cells = fine.reshape(box, factor, box, factor).transpose(1, 3, 0, 2)
        spectra = fft(
            rfft(cells, n=length, axis=-1),
            n=length,
            axis=-2,
        ).reshape(factor * factor, -1)
        samples = irfft2(
            np.einsum("ij,ij->j", spectra, self._spectra).reshape(
                length,
                length // 2 + 1,
            ),
            s=(length, length),
        )

        return samples[:self.samples, :self.samples]


def local_images(
    scaled_images: dict[str, FloatArray],
    convolution: PolyphaseConvolution,
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
    multiplied by the window, convolved by ``convolution`` with the
    sub-pixel kernels, and placed on a ``shape`` image whose index ``(0, 0)``
    lies at the global coordinate ``origin``.
    """
    if convolution.factor != factor:
        raise ValueError(
            "The convolution must use the fine factor of the box."
        )

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
    shift = -(convolution.size // 2)
    images = {}

    for name, fine in scaled_images.items():
        if fine.shape != (box, box):
            raise ValueError(
                "Fine images must cover the box of the effective-PSF window."
            )

        samples = convolution(fine * weight)
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
