"""Model wrappers."""

from __future__ import annotations

from baymorph.simulation.psf import convolve_with_psf


class ConvolvedModel:
    """Model wrapper that applies PSF convolution after rendering.

    Parameters
    ----------
    model : object
        Model object exposing ``render(shape=...)`` and optionally
        ``render_components(shape=...)``.
    psf : array-like
        Point-spread function used to convolve the rendered model image.
    """

    def __init__(
        self,
        model,
        psf,
    ):
        self.model = model
        self.psf = psf

    def render(
        self,
        shape,
    ):
        """Render the wrapped model and convolve it with the PSF.

        Parameters
        ----------
        shape : tuple of int
            Output image shape as ``(ny, nx)``.

        Returns
        -------
        ndarray
            PSF-convolved model image.
        """
        image = self.model.render(
            shape=shape,
        )

        return convolve_with_psf(
            image,
            self.psf,
        )

    def render_components(
        self,
        shape,
    ):
        """Render and PSF-convolve every model component.

        Parameters
        ----------
        shape : tuple of int
            Output image shape as ``(ny, nx)``.

        Returns
        -------
        dict of str, ndarray
            Dictionary containing the PSF-convolved model components.

        Raises
        ------
        AttributeError
            If the wrapped model does not expose ``render_components``.
        """
        components = self.model.render_components(
            shape=shape,
        )

        return {
            name: convolve_with_psf(image, self.psf)
            for name, image in components.items()
        }