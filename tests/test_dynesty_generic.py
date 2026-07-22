import numpy as np
import pytest

from baymorph import (
    DynestyFitter,
    SersicModel,
    make_constant_noise_map,
)


def test_dynesty_generic_fit_runs():
    true_flux = 1000.0
    true_re = 8.0
    true_n = 2.0

    data = SersicModel(
        flux=true_flux,
        re=true_re,
        n=true_n,
        q=0.8,
        theta=0.0,
        x0=30,
        y0=30,
    ).render(
        shape=(60, 60)
    )

    noise = make_constant_noise_map(
        data.shape,
        sigma=1.0,
    )

    def model_factory(flux, re, n):
        return SersicModel(
            flux=flux,
            re=re,
            n=n,
            q=0.8,
            theta=0.0,
            x0=30,
            y0=30,
        )

    fitter = DynestyFitter(
        model_factory=model_factory,
        data=data,
        noise=noise,
        parameter_names=["flux", "re", "n"],
        prior_bounds={
            "flux": (100.0, 2000.0),
            "re": (2.0, 20.0),
            "n": (0.3, 10.0),
        },
    )

    result = fitter.run(
        nlive=50,
        sample="rslice",
        print_progress=False,
    )

    assert result.samples.shape[1] == 3
    assert result.parameter_names == ["flux", "re", "n"]


def test_dynesty_generic_rejects_missing_prior():
    data = np.ones((10, 10))
    noise = np.ones((10, 10))

    def model_factory(flux):
        return SersicModel(
            flux=flux,
            re=5,
            n=2,
            q=0.8,
            theta=0,
            x0=5,
            y0=5,
        )

    with pytest.raises(ValueError):
        DynestyFitter(
            model_factory=model_factory,
            data=data,
            noise=noise,
            parameter_names=["flux"],
            prior_bounds={},
        )


def test_dynesty_generic_theta_to_parameters():
    data = np.ones((10, 10))
    noise = np.ones((10, 10))

    def model_factory(flux, re):
        return SersicModel(
            flux=flux,
            re=re,
            n=2,
            q=0.8,
            theta=0,
            x0=5,
            y0=5,
        )

    fitter = DynestyFitter(
        model_factory=model_factory,
        data=data,
        noise=noise,
        parameter_names=["flux", "re"],
        prior_bounds={
            "flux": (100, 2000),
            "re": (2, 20),
        },
    )

    params = fitter.theta_to_parameters(
        np.array([1000.0, 8.0])
    )

    assert params == {
        "flux": 1000.0,
        "re": 8.0,
    }


def test_dynesty_generic_accepts_sampling_options():
    data = SersicModel(
        flux=1000,
        re=8,
        n=2,
        q=0.8,
        theta=0,
        x0=30,
        y0=30,
    ).render(
        shape=(60, 60)
    )

    noise = make_constant_noise_map(
        data.shape,
        sigma=1.0,
    )

    def model_factory(flux):
        return SersicModel(
            flux=flux,
            re=8,
            n=2,
            q=0.8,
            theta=0,
            x0=30,
            y0=30,
        )

    fitter = DynestyFitter(
        model_factory=model_factory,
        data=data,
        noise=noise,
        parameter_names=["flux"],
        prior_bounds={
            "flux": (100.0, 2000.0),
        },
    )

    result = fitter.run(
        nlive=50,
        sample="rwalk",
        bound="multi",
        dlogz=1.0,
        print_progress=False,
    )

    assert result.samples.shape[1] == 1