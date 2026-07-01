"""Model wrappers."""

from baymorph.simulation.psf import convolve_with_psf


class ConvolvedModel:
    """Model wrapper that applies PSF convolution after rendering.

    Parameters
    ----------
    model : object
        Model object exposing a ``render(shape=...)`` method.
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