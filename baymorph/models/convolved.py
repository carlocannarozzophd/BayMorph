"""PSF-convolved model wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np

from baymorph.simulation.psf import GaussianPSF, convolve_with_psf


class ConvolvedModel:
    """Apply PSF convolution to a model rendered on an extended domain.

    The intrinsic model is rendered beyond the requested output footprint by
    the half-width of the PSF kernel. This retains light originating outside
    the observed cutout that is scattered into it by the PSF.

    Parameters
    ----------
    model : object
        Model exposing ``render(shape=..., origin=...)`` and optionally
        ``render_components(shape=..., origin=...)``.
    psf : object
        PSF object exposing a ``kernel()`` method.
    kernel_size : int, default=25
        Kernel size used for Gaussian PSF models.
    """

    def __init__(
        self,
        model: Any,
        psf: Any,
        kernel_size: int = 25,
    ) -> None:
        self.model = model
        self.psf = psf
        self.kernel_size = kernel_size

    def _kernel(self) -> np.ndarray:
        """Return the normalized PSF kernel used by this wrapper."""
        if isinstance(self.psf, GaussianPSF):
            kernel = self.psf.kernel(size=self.kernel_size)
        else:
            kernel = self.psf.kernel()

        kernel_array = np.asarray(kernel, dtype=float)

        if kernel_array.ndim != 2:
            raise ValueError("PSF kernel must be 2D.")

        if not np.all(np.isfinite(kernel_array)):
            raise ValueError(
                "PSF kernel must contain only finite values."
            )

        total = kernel_array.sum()

        if total <= 0.0:
            raise ValueError("PSF kernel must have positive sum.")

        return kernel_array / total

    def _render_extended(
        self,
        shape: tuple[int, int],
    ) -> tuple[np.ndarray, int, int]:
        """Render the intrinsic model on the PSF-extended domain."""
        kernel = self._kernel()
        pad_y = kernel.shape[0] // 2
        pad_x = kernel.shape[1] // 2

        extended_shape = (
            shape[0] + 2 * pad_y,
            shape[1] + 2 * pad_x,
        )
        extended_origin = (
            -float(pad_y),
            -float(pad_x),
        )

        try:
            image = self.model.render(
                shape=extended_shape,
                origin=extended_origin,
            )
        except TypeError as exc:
            raise TypeError(
                "PSF-convolved models must support "
                "render(shape=..., origin=...)."
            ) from exc

        return np.asarray(image, dtype=float), pad_y, pad_x

    def render(
        self,
        shape: tuple[int, int],
    ) -> np.ndarray:
        """Render, PSF-convolve, and crop to the requested footprint."""
        extended, pad_y, pad_x = self._render_extended(shape)

        convolved = convolve_with_psf(
            extended,
            self.psf,
            kernel_size=self.kernel_size,
        )

        return convolved[
            pad_y : pad_y + shape[0],
            pad_x : pad_x + shape[1],
        ]

    def render_components(
        self,
        shape: tuple[int, int],
    ) -> dict[str, np.ndarray]:
        """Render and convolve every component on the extended domain."""
        kernel = self._kernel()
        pad_y = kernel.shape[0] // 2
        pad_x = kernel.shape[1] // 2

        extended_shape = (
            shape[0] + 2 * pad_y,
            shape[1] + 2 * pad_x,
        )
        extended_origin = (
            -float(pad_y),
            -float(pad_x),
        )

        try:
            components = self.model.render_components(
                shape=extended_shape,
                origin=extended_origin,
            )
        except TypeError as exc:
            raise TypeError(
                "PSF-convolved component models must support "
                "render_components(shape=..., origin=...)."
            ) from exc

        return {
            name: convolve_with_psf(
                image,
                self.psf,
                kernel_size=self.kernel_size,
            )[
                pad_y : pad_y + shape[0],
                pad_x : pad_x + shape[1],
            ]
            for name, image in components.items()
        }
