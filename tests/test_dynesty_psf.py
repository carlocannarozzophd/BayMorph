"""Tests for Dynesty fits with PSF convolution."""

import numpy as np

from baymorph.core import make_constant_noise_map
from baymorph.fitting import (
    DynestyFitter,
    Parameter,
    ParameterSet,
)
from baymorph.likelihood import GaussianImageLikelihood
from baymorph.models import SersicModel
from baymorph.simulation import GaussianPSF



def test_dynesty_accepts_psf():
    """Dynesty should convolve rendered models with the supplied PSF."""
    data = SersicModel(
        flux=1000.0,
        re=8.0,
        n=2.0,
        q=0.8,
        theta=0.0,
        x0=30.0,
        y0=30.0,
    ).render(
        shape=(60, 60)
    )

    noise = make_constant_noise_map(
        data.shape,
        sigma=1.0,
    )

    likelihood = GaussianImageLikelihood(
        data=data,
        noise=noise,
    )

    parameters = ParameterSet(
        [
            Parameter(
                name="flux",
                lower=100.0,
                upper=2000.0,
            ),
        ]
    )

    psf = GaussianPSF(
        fwhm=3.0,
    )

    def model_factory(
        *,
        flux,
    ):
        return SersicModel(
            flux=flux,
            re=8.0,
            n=2.0,
            q=0.8,
            theta=0.0,
            x0=30.0,
            y0=30.0,
        )

    fitter = DynestyFitter(
        model_factory=model_factory,
        parameters=parameters,
        likelihood=likelihood,
        psf=psf,
    )

    theta = np.array(
        [
            1000.0,
        ]
    )

    model_image = fitter.render_model(
        theta
    )

    unconvolved_image = model_factory(
        flux=1000.0,
    ).render(
        shape=data.shape
    )

    assert model_image.shape == data.shape
    assert np.all(np.isfinite(model_image))
    assert not np.allclose(
        model_image,
        unconvolved_image,
    )