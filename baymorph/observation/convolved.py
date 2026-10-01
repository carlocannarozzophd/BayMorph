"""PSF-convolved model wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np

from baymorph.observation.effects import PSFEffect
from baymorph.rendering.effective_psf import (
    SUBPIXEL_FACTOR,
    WINDOW_HALF,
    apply_outer_weight,
    fine_box_origin,
    local_images,
)


class ConvolvedModel:
    """Apply a PSF effect after rendering on an extended domain.

    The intrinsic model is rendered beyond the requested output footprint by
    the half-width of the PSF kernel. This retains light originating outside
    the observed cutout that is scattered into it by the PSF.

    With a PSF that provides the effective PSF at sub-pixel offsets, the
    model follows the effective-PSF rendering of AD-046: about its centre
    the profile is integrated over finer cells and convolved with the
    sub-pixel kernels, elsewhere it is sampled at pixel centres and
    convolved with the native kernel. The model must then declare its
    centre and render on a finer grid.

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

    def _model_render_kwargs(self) -> dict[str, Any]:
        """Return extra ``render()`` kwargs enforcing the PSF pixel-response contract.

        A PSF object must declare ``includes_pixel_response`` (see
        ``GaussianPSF``/``EmpiricalPSF`` in ``baymorph.simulation.psf``). A
        kernel without the pixel response is convolved with the profile
        integrated over pixels. A kernel with it takes the effective-PSF
        rendering of AD-046 and must provide sub-pixel kernels; reaching
        this method with such a kernel means it does not, which is an error.
        """
        if not hasattr(self.psf, "includes_pixel_response"):
            raise TypeError(
                "PSF objects used with ConvolvedModel must declare "
                "`includes_pixel_response` to state whether their kernel "
                "already includes the detector pixel response."
            )

        if self.psf.includes_pixel_response:
            raise ValueError(
                "A PSF that already includes the pixel response must provide "
                "the effective PSF at sub-pixel offsets (AD-046); sampling "
                "the profile at pixel centres is no longer supported. Build "
                "the empirical PSF with its sub-pixel kernels."
            )

        return {}

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

    def _uses_effective_psf(self) -> bool:
        """Return whether the PSF selects the rendering of AD-046."""
        return bool(
            getattr(self.psf, "has_subpixel_kernels", False)
        )

    def _render_effective(
        self,
        extended_shape: tuple[int, int],
        extended_origin: tuple[float, float],
        *,
        components: bool,
    ) -> dict[str, np.ndarray]:
        """Render with the effective-PSF convention of AD-046, uncropped.

        The point-sampled images are weighted by one minus the window and
        convolved with the native kernel; the fine images, integrated over
        cells ``SUBPIXEL_FACTOR`` times finer within the box of the window by
        the ``hybrid_cusp`` rule, are weighted by the window and convolved
        with the sub-pixel kernels by the polyphase convolution of the PSF.
        """
        model = self.model

        if not hasattr(model, "centre") or not hasattr(model, "scaled"):
            raise TypeError(
                "The effective-PSF rendering of AD-046 needs a model that "
                "declares its centre and renders on a finer grid."
            )

        x, y = model.centre()
        factor = SUBPIXEL_FACTOR
        box = (2 * WINDOW_HALF + 1) * factor
        first_row, first_column = fine_box_origin(x, y, factor=factor)
        fine_origin = (float(first_row), float(first_column))
        scaled = model.scaled(factor)

        if components:
            outer = model.render_components(
                shape=extended_shape,
                origin=extended_origin,
                renderer="point",
            )
            fine = scaled.render_components(
                shape=(box, box),
                origin=fine_origin,
                renderer="hybrid_cusp",
            )
        else:
            outer = {
                "model": model.render(
                    shape=extended_shape,
                    origin=extended_origin,
                    renderer="point",
                )
            }
            fine = {
                "model": scaled.render(
                    shape=(box, box),
                    origin=fine_origin,
                    renderer="hybrid_cusp",
                )
            }

        local = local_images(
            {name: np.asarray(image, dtype=float) for name, image in fine.items()},
            self.psf.fine_convolution(),
            x,
            y,
            extended_shape,
            extended_origin,
            factor=factor,
        )
        convolved = {}

        for name, image in outer.items():
            weighted = np.array(image, dtype=float)
            apply_outer_weight(weighted, extended_origin, x, y)
            convolved[name] = self.psf_effect.apply(weighted) + local[name]

        return convolved

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

        if self._uses_effective_psf():
            return self._crop(
                self._render_effective(
                    extended_shape,
                    extended_origin,
                    components=False,
                )["model"],
                shape=shape,
                pad_y=pad_y,
                pad_x=pad_x,
            )

        render_kwargs = self._model_render_kwargs()

        try:
            intrinsic = self.model.render(
                shape=extended_shape,
                origin=extended_origin,
                **render_kwargs,
            )
        except TypeError as exc:
            raise TypeError(
                "PSF-convolved models must support "
                "render(shape=..., origin=...)"
                + (
                    ", and a `renderer=\"point\"` keyword argument when "
                    "used with a PSF whose kernel already includes the "
                    "detector pixel response."
                    if render_kwargs
                    else "."
                )
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

        if self._uses_effective_psf():
            if not hasattr(self.model, "render_components"):
                raise TypeError(
                    "PSF-convolved component models must implement "
                    "render_components()."
                )

            return {
                name: self._crop(
                    image,
                    shape=shape,
                    pad_y=pad_y,
                    pad_x=pad_x,
                )
                for name, image in self._render_effective(
                    extended_shape,
                    extended_origin,
                    components=True,
                ).items()
            }

        render_kwargs = self._model_render_kwargs()

        try:
            components = self.model.render_components(
                shape=extended_shape,
                origin=extended_origin,
                **render_kwargs,
            )
        except AttributeError as exc:
            raise TypeError(
                "PSF-convolved component models must implement "
                "render_components()."
            ) from exc
        except TypeError as exc:
            raise TypeError(
                "PSF-convolved component models must support "
                "render_components(shape=..., origin=...)"
                + (
                    ", and a `renderer=\"point\"` keyword argument when "
                    "used with a PSF whose kernel already includes the "
                    "detector pixel response."
                    if render_kwargs
                    else "."
                )
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
