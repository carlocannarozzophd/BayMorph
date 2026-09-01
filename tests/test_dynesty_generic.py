"""Tests for the Dynesty fitting backend."""

from types import SimpleNamespace

import numpy as np
import pytest

from baymorph.core import make_constant_noise_map
from baymorph.fitting import (
    dynesty as dynesty_module,
    DynestyFitter,
    Parameter,
    ParameterSet,
)
from baymorph.fitting.image_problem import ImageInferenceProblem
from baymorph.likelihood import GaussianImageLikelihood
from baymorph.models import SersicModel



def make_test_problem():
    """Construct a small Sérsic fitting problem."""
    true_flux = 1000.0
    true_re = 8.0
    true_n = 2.0

    data = SersicModel(
        flux=true_flux,
        re=true_re,
        n=true_n,
        q=0.8,
        theta=0.0,
        x0=15.0,
        y0=15.0,
    ).render(
        shape=(30, 30)
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
            Parameter(
                name="re",
                lower=2.0,
                upper=20.0,
            ),
            Parameter(
                name="n",
                lower=0.3,
                upper=10.0,
            ),
        ]
    )

    def model_factory(
        *,
        flux,
        re,
        n,
    ):
        return SersicModel(
            flux=flux,
            re=re,
            n=n,
            q=0.8,
            theta=0.0,
            x0=15.0,
            y0=15.0,
        )

    problem = ImageInferenceProblem(
        model_factory=model_factory,
        parameters=parameters,
        likelihood=likelihood,
    )
    fitter = DynestyFitter(
        problem=problem,
    )

    return fitter


def test_dynesty_generic_fit_runs():
    """Dynesty should complete a small generic fit."""
    fitter = make_test_problem()

    result = fitter.run(
        mode="static",
        init_kwargs={
            "nlive": 20,
            "sample": "rwalk",
            "bootstrap": 0,
            "rstate": np.random.default_rng(
                123
            ),
        },
        run_kwargs={
            "dlogz": 5.0,
            "print_progress": False,
        },
    )

    assert result.samples.ndim == 2
    assert result.samples.shape[1] == 3

    assert result.parameter_names == [
        "flux",
        "re",
        "n",
    ]

    assert np.isfinite(
        result.logz
    )
    assert np.isfinite(
        result.logz_error
    )


def test_dynesty_generic_requires_parameter_set():
    """The fitter should reject invalid parameter containers."""
    data = np.ones(
        (5, 5),
        dtype=float,
    )
    noise = np.ones_like(data)

    likelihood = GaussianImageLikelihood(
        data=data,
        noise=noise,
    )

    def model_factory(
        *,
        flux,
    ):
        return SersicModel(
            flux=flux,
            re=1.0,
            n=1.0,
            q=1.0,
            theta=0.0,
            x0=2.0,
            y0=2.0,
        )

    with pytest.raises(
        TypeError,
        match="ParameterSet",
    ):
        ImageInferenceProblem(
            model_factory=model_factory,
            parameters=["flux"],
            likelihood=likelihood,
        )


def test_dynesty_generic_parameter_mapping():
    """Parameter vectors should map correctly to model-factory keywords."""
    fitter = make_test_problem()

    theta = np.array(
        [
            1000.0,
            8.0,
            2.0,
        ]
    )

    model_image = fitter.render_model(
        theta
    )

    expected_image = SersicModel(
        flux=1000.0,
        re=8.0,
        n=2.0,
        q=0.8,
        theta=0.0,
        x0=15.0,
        y0=15.0,
    ).render(
        shape=(30, 30)
    )

    assert np.allclose(
        model_image,
        expected_image,
    )


def test_dynesty_generic_prior_transform():
    """The fitter should delegate prior transformation to ParameterSet."""
    fitter = make_test_problem()

    transformed = fitter.prior_transform(
        np.array(
            [
                0.0,
                0.5,
                1.0,
            ]
        )
    )

    expected = np.array(
        [
            100.0,
            11.0,
            10.0,
        ]
    )

    assert np.allclose(
        transformed,
        expected,
    )


def test_dynesty_generic_rejects_invalid_unit_cube_sample():
    """Prior-transform input must lie inside the unit cube."""
    fitter = make_test_problem()

    with pytest.raises(
        ValueError,
        match="between zero and one",
    ):
        fitter.prior_transform(
            np.array(
                [
                    -0.1,
                    0.5,
                    1.0,
                ]
            )
        )


def test_dynesty_generic_accepts_sampling_options(
    monkeypatch,
):
    """Sampling options should be forwarded to Dynesty."""
    captured = {}

    class FakeNestedSampler:
        """Minimal Dynesty sampler used to inspect supplied options."""

        def __init__(
            self,
            loglikelihood,
            prior_transform,
            ndim,
            **kwargs,
        ):
            captured["loglikelihood"] = loglikelihood
            captured["prior_transform"] = prior_transform
            captured["ndim"] = ndim
            captured["sampler_kwargs"] = kwargs

            self.results = SimpleNamespace(
                samples=np.zeros(
                    (2, ndim),
                    dtype=float,
                ),
                logwt=np.array(
                    [-1.0, 0.0]
                ),
                logz=np.array(
                    [-1.0, 0.0]
                ),
                logl=np.array(
                    [-2.0, -1.0]
                ),
                logzerr=np.array(
                    [0.2, 0.1]
                ),
                ncall=np.array(
                    [1, 1]
                ),
            )

        def run_nested(
            self,
            **kwargs,
        ):
            captured["run_kwargs"] = kwargs

    monkeypatch.setattr(
        dynesty_module.dynesty,
        "NestedSampler",
        FakeNestedSampler,
    )

    fitter = make_test_problem()

    random_state = np.random.default_rng(
        42
    )

    result = fitter.run(
        mode="static",
        init_kwargs={
            "nlive": 50,
            "sample": "rwalk",
            "bound": "single",
            "bootstrap": 3,
            "rstate": random_state,
        },
        run_kwargs={
            "dlogz": 0.25,
            "print_progress": False,
        },
    )

    assert captured["ndim"] == 3

    assert captured["sampler_kwargs"]["nlive"] == 50
    assert captured["sampler_kwargs"]["sample"] == "rwalk"
    assert captured["sampler_kwargs"]["bound"] == "single"
    assert captured["sampler_kwargs"]["bootstrap"] == 3
    assert (
        captured["sampler_kwargs"]["rstate"]
        is random_state
    )

    assert captured["run_kwargs"] == {
        "print_progress": False,
        "dlogz": 0.25,
    }

    assert result.parameter_names == [
        "flux",
        "re",
        "n",
    ]
