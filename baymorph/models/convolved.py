"""Model wrappers."""

from baymorph.simulation.psf import convolve_with_psf

class ConvolvedModel:
    """Model wrapper that applies PSF convolution after rendering."""

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
        image = self.model.render(
            shape=shape,
        )

        return convolve_with_psf(
            image,
            self.psf,
        )