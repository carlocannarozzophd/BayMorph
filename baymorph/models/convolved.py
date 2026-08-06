"""PSF-convolved model wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np

from baymorph.observation.effects import PSFEffect


class ConvolvedModel:
    """Apply a PSF effect after rendering on an extended domain.

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
        self.psf_effect = PSFEffect(
            psf,
            kernel_size=kernel_size,
        )

    @property
    def psf(self) -> Any:
        """Return the configured PSF object."""
        return self.psf_effect.psf

    @property
    def kernel_size(self) -> int:
        """Return the configured analytic-kernel size."""
        return self.psf_effect.kernel_size

    def _extended_domain(
        self,
        shape: tuple[int, int],
    ) -> tuple[
        tuple[int, int],
        tuple[float, float],
        int,
        int,
    ]:
        """Return the PSF-extended shape, origin, and crop widths."""
        kernel = self.psf_effect.normalized_kernel()

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

        return (
            extended_shape,
            extended_origin,
            pad_y,
            pad_x,
        )

    @staticmethod
    def _crop(
        image: np.ndarray,
        *,
        shape: tuple[int, int],
        pad_y: int,
        pad_x: int,
    ) -> np.ndarray:
        """Crop an extended image to the requested footprint."""
        return np.asarray(
            image[
                pad_y : pad_y + shape[0],
                pad_x : pad_x + shape[1],
            ],
            dtype=float,
        )

    def render(
        self,
        shape: tuple[int, int],
    ) -> np.ndarray:
        """Render, PSF-convolve, and crop the complete model."""
        (
            extended_shape,
            extended_origin,
            pad_y,
            pad_x,
        ) = self._extended_domain(shape)

        try:
            intrinsic = self.model.render(
                shape=extended_shape,
                origin=extended_origin,
            )
        except TypeError as exc:
            raise TypeError(
                "PSF-convolved models must support "
                "render(shape=..., origin=...)."
            ) from exc

        convolved = self.psf_effect.apply(
            intrinsic,
        )

        return self._crop(
            convolved,
            shape=shape,
            pad_y=pad_y,
            pad_x=pad_x,
        )

    def render_components(
        self,
        shape: tuple[int, int],
    ) -> dict[str, np.ndarray]:
        """Render, convolve, and crop every named component."""
        (
            extended_shape,
            extended_origin,
            pad_y,
            pad_x,
        ) = self._extended_domain(shape)

        try:
            components = self.model.render_components(
                shape=extended_shape,
                origin=extended_origin,
            )
        except AttributeError as exc:
            raise TypeError(
                "PSF-convolved component models must implement "
                "render_components()."
            ) from exc
        except TypeError as exc:
            raise TypeError(
                "PSF-convolved component models must support "
                "render_components(shape=..., origin=...)."
            ) from exc

        return {
            name: self._crop(
                self.psf_effect.apply(image),
                shape=shape,
                pad_y=pad_y,
                pad_x=pad_x,
            )
            for name, image in components.items()
        }
