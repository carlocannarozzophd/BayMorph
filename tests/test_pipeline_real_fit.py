from types import SimpleNamespace

import numpy as np
import pytest

from baymorph.fitting import (
    Parameter,
    ParameterSet,
)
from baymorph import PosteriorResult
from baymorph.likelihood import GaussianImageLikelihood
from baymorph.pipeline.config import resolve_config

from baymorph.pipeline.real_fit import (
    RealFitProducts,
    _build_parameter_set,
    _make_model_factory,
    _compute_best_model,
    _compute_fit_statistics,
    _compute_bayesian_model_comparison,
    _run_sampler,
    load_real_fit_products,
    run_real_fit_from_config,
    save_real_fit_products,
)



def test_run_real_fit_requires_image_preparation_products(
    monkeypatch,
):
    resolved = {
        "data": {},
    }

    monkeypatch.setattr(
        "baymorph.pipeline.real_fit.resolve_config",
        lambda config: resolved,
    )
    monkeypatch.setattr(
        "baymorph.pipeline.real_fit.make_output_tree",
        lambda config: None,
    )
    monkeypatch.setattr(
        "baymorph.pipeline.real_fit.save_resolved_config",
        lambda config: None,
    )

    def raise_missing_products(config):
        del config
        raise FileNotFoundError("missing prepared image")

    monkeypatch.setattr(
        "baymorph.pipeline.real_fit.load_image_preparation_products",
        raise_missing_products,
    )

    with pytest.raises(
        FileNotFoundError,
        match="Image-preparation products are required before fitting",
    ):
        run_real_fit_from_config({})


def test_run_real_fit_requires_psf_products(
    monkeypatch,
):
    resolved = {
        "data": {},
    }

    monkeypatch.setattr(
        "baymorph.pipeline.real_fit.resolve_config",
        lambda config: resolved,
    )
    monkeypatch.setattr(
        "baymorph.pipeline.real_fit.make_output_tree",
        lambda config: None,
    )
    monkeypatch.setattr(
        "baymorph.pipeline.real_fit.save_resolved_config",
        lambda config: None,
    )
    monkeypatch.setattr(
        "baymorph.pipeline.real_fit.load_image_preparation_products",
        lambda config: object(),
    )

    def raise_missing_psf(config):
        del config
        raise FileNotFoundError("missing PSF")

    monkeypatch.setattr(
        "baymorph.pipeline.real_fit._load_psf_from_config",
        raise_missing_psf,
    )

    with pytest.raises(
        FileNotFoundError,
        match="PSF products are required before fitting",
    ):
        run_real_fit_from_config({})



def test_real_fit_products_preserve_posterior_round_trip(
    tmp_path,
    monkeypatch,
):
    config = resolve_config(
        {
            "object": {
                "name": "NGC1234",
            },
            "data": {
                "image_path": "data/image.fits",
            },
            "outputs": {
                "root": str(tmp_path / "outputs"),
            },
            "fit": {
                "models": [
                    "single_sersic",
                ],
            },
        }
    )

    posterior = PosteriorResult(
        samples=np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
            ]
        ),
        logwt=np.array([-2.0, -1.0]),
        logz=np.array([-3.0, -0.5]),
        parameter_names=[
            "flux",
            "re",
        ],
        logl=np.array([-10.0, -5.0]),
        logzerr=np.array([0.4, 0.2]),
    )

    products = RealFitProducts(
        model_name="single_sersic",
        posterior=posterior,
        best_parameters={
            "flux": 2.0,
            "re": 20.0,
        },
        best_model_image=np.ones((2, 2)),
        residual=np.zeros((2, 2)),
        norm_residual=np.zeros((2, 2)),
        chi2=1.0,
        ndof=2,
        reduced_chi2=0.5,
        noise=np.ones((2, 2)),
    )

    monkeypatch.setattr(
        "baymorph.pipeline.real_fit.load_image_preparation_products",
        lambda config: SimpleNamespace(
            noise=np.ones((2, 2)),
        ),
    )

    save_real_fit_products(
        config,
        "single_sersic",
        products,
    )

    loaded = load_real_fit_products(
        config,
        "single_sersic",
    )

    assert loaded.posterior is not None
    assert np.array_equal(
        loaded.posterior.samples,
        posterior.samples,
    )
    assert np.array_equal(
        loaded.posterior.logwt,
        posterior.logwt,
    )
    assert np.array_equal(
        loaded.posterior.logz,
        posterior.logz,
    )
    assert np.array_equal(
        loaded.posterior.logl,
        posterior.logl,
    )
    assert np.array_equal(
        loaded.posterior.logzerr,
        posterior.logzerr,
    )
    assert (
        loaded.posterior.parameter_names
        == posterior.parameter_names
    )


class DummyResult:
    best_parameters = {
        "a": 2.0,
        "b": 3.0,
    }


class DummyModel:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def render(self, shape):
        return np.full(shape, self.a + self.b)


class DeltaPSF:
    def kernel(self):
        psf = np.zeros((9, 9))
        psf[4, 4] = 1.0
        return psf


class DummyPosterior:
    def __init__(self, logz):
        self.logz = np.array([logz])


def test_compute_best_model_uses_best_parameters():
    def model_factory(a, b):
        return DummyModel(a, b)

    image, best_parameters = _compute_best_model(
        result=DummyResult(),
        parameter_names=["a", "b"],
        model_factory=model_factory,
        image_shape=(4, 5),
    )

    assert image.shape == (4, 5)
    assert best_parameters == {
        "a": 2.0,
        "b": 3.0,
    }
    assert np.all(image == 5.0)


def test_compute_fit_statistics_uses_unmasked_valid_pixels():
    data = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    model = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )

    noise = np.ones_like(data)

    mask = np.array(
        [
            [False, False],
            [True, False],
        ]
    )

    stats = _compute_fit_statistics(
        data=data,
        model_image=model,
        noise=noise,
        mask=mask,
        n_parameters=2,
    )

    assert stats["chi2"] == pytest.approx(10.0)
    assert stats["ndof"] == 1
    assert stats["reduced_chi2"] == pytest.approx(10.0)
    assert np.allclose(
        stats["residual"],
        data - model,
    )
    assert np.isnan(
        stats["norm_residual"][1, 0]
    )

    assert np.allclose(
        stats["norm_residual"][~mask],
        (data - model)[~mask],
)


def test_build_parameter_set_single_sersic():
    parameter_names, parameters, _, _ = (
        _build_parameter_set(
            "single_sersic",
            data_fit=np.ones((10, 10)),
            noise_fit=np.ones((10, 10)),
            mask_fit=np.zeros((10, 10), dtype=bool),
            target_x=5,
            target_y=5,
        )
    )

    assert len(parameter_names) == 8
    assert parameter_names[0] == "flux"
    assert parameters.names[0] == "flux"


def test_build_parameter_set_double_sersic():
    parameter_names, parameters, _, _ = (
        _build_parameter_set(
            "double_sersic",
            data_fit=np.ones((10, 10)),
            noise_fit=np.ones((10, 10)),
            mask_fit=np.zeros((10, 10), dtype=bool),
            target_x=5,
            target_y=5,
        )
    )

    assert len(parameter_names) == 13

    assert parameter_names == [
        "flux_bulge",
        "re_bulge",
        "n_bulge",
        "q_bulge",
        "theta_bulge",
        "flux_disk",
        "re_disk",
        "n_disk",
        "q_disk",
        "theta_disk",
        "x0",
        "y0",
        "sky",
    ]

    assert parameters.names == parameter_names


def test_make_model_factory_single_sersic():
    factory = _make_model_factory(
        "single_sersic",
        DeltaPSF(),
    )

    model = factory(
        100.0,
        5.0,
        2.0,
        0.8,
        0.0,
        15.0,
        15.0,
        1.0,
    )

    image = model.render((31, 31))

    assert image.shape == (31, 31)
    assert np.all(np.isfinite(image))


def test_make_model_factory_double_sersic():
    factory = _make_model_factory(
        "double_sersic",
        DeltaPSF(),
    )

    model = factory(
        100.0,   # flux_bulge
        3.0,     # re_bulge
        2.0,     # n_bulge
        0.9,     # q_bulge
        0.1,     # theta_bulge
        50.0,    # flux_disk
        8.0,     # re_disk
        1.0,     # n_disk
        0.8,     # q_disk
        0.0,     # theta_disk
        15.0,    # x0
        15.0,    # y0
        1.0,     # sky
    )

    image = model.render((31, 31))

    assert image.shape == (31, 31)
    assert np.all(np.isfinite(image))


def test_compute_bayesian_model_comparison():
    single = RealFitProducts(
        posterior=DummyPosterior(10.0),
        best_parameters={},
        best_model_image=np.zeros((2, 2)),
        residual=np.zeros((2, 2)),
        norm_residual=np.zeros((2, 2)),
        chi2=1.0,
        ndof=1,
        reduced_chi2=1.0,
    )

    double = RealFitProducts(
        posterior=DummyPosterior(13.0),
        best_parameters={},
        best_model_image=np.zeros((2, 2)),
        residual=np.zeros((2, 2)),
        norm_residual=np.zeros((2, 2)),
        chi2=1.0,
        ndof=1,
        reduced_chi2=1.0,
    )

    comparison = _compute_bayesian_model_comparison(
        {
            "single_sersic": single,
            "double_sersic": double,
        }
    )

    assert comparison["logz"]["single_sersic"] == pytest.approx(10.0)
    assert comparison["logz"]["double_sersic"] == pytest.approx(13.0)
    assert comparison["delta_logz"]["double_sersic_minus_single_sersic"] == pytest.approx(3.0)
    assert comparison["bayes_factor"]["double_sersic_over_single_sersic"] == pytest.approx(
        np.exp(3.0)
    )

def test_run_sampler_builds_likelihood_and_backend(
    monkeypatch,
):
    data = np.ones((5, 5), dtype=float)
    noise = np.full(data.shape, 2.0)
    mask = np.zeros(data.shape, dtype=bool)

    parameters = ParameterSet(
        [
            Parameter(
                "flux",
                1.0,
                10.0,
            )
        ]
    )

    expected_result = object()
    captured = {}

    class DummySampler:
        def run(self, **kwargs):
            captured["run_kwargs"] = kwargs
            return expected_result

    def fake_make_sampler(
        config,
        *,
        model_factory,
        parameters,
        likelihood,
    ):
        captured["sampler_cfg"] = config
        captured["model_factory"] = model_factory
        captured["parameters"] = parameters
        captured["likelihood"] = likelihood
        return DummySampler()

    def model_factory(*, flux):
        del flux
        return None

    monkeypatch.setattr(
        "baymorph.pipeline.real_fit.make_sampler",
        fake_make_sampler,
    )

    sampler_cfg = {
        "name": "dynesty",
        "nlive": 40,
    }

    result = _run_sampler(
        data_fit=data,
        noise_fit=noise,
        mask_fit=mask,
        parameters=parameters,
        model_factory=model_factory,
        likelihood_cfg={
            "type": "gaussian",
        },
        sampler_cfg=sampler_cfg,
        nlive=40,
        sample="rwalk",
        dlogz=0.5,
        n_processes=1,
        queue_size=None,
        seed=123,
        progress_label="single_sersic",
    )

    assert result is expected_result
    assert captured["sampler_cfg"] is sampler_cfg
    assert captured["model_factory"] is model_factory
    assert captured["parameters"] is parameters
    assert isinstance(
        captured["likelihood"],
        GaussianImageLikelihood,
    )
    assert np.array_equal(
        captured["likelihood"].data,
        data,
    )
    assert captured["run_kwargs"] == {
        "nlive": 40,
        "sample": "rwalk",
        "dlogz": 0.5,
        "print_progress": True,
        "n_processes": 1,
        "queue_size": None,
        "seed": 123,
        "progress_label": "single_sersic",
    }


def test_run_sampler_executes_native_backend_end_to_end():
    class ConstantFluxModel:
        def __init__(
            self,
            flux: float,
        ) -> None:
            self.flux = float(flux)

        def render(
            self,
            shape,
        ):
            return np.full(
                shape,
                self.flux,
                dtype=float,
            )

    def model_factory(
        *,
        flux,
    ):
        return ConstantFluxModel(
            flux
        )

    data = np.full(
        (3, 3),
        1.5,
        dtype=float,
    )
    noise = np.ones_like(
        data
    )
    mask = np.zeros(
        data.shape,
        dtype=bool,
    )

    parameters = ParameterSet(
        [
            Parameter(
                "flux",
                1.0,
                2.0,
            )
        ]
    )

    result = _run_sampler(
        data_fit=data,
        noise_fit=noise,
        mask_fit=mask,
        parameters=parameters,
        model_factory=model_factory,
        likelihood_cfg={
            "type": "gaussian",
            "include_normalization": False,
        },
        sampler_cfg={
            "name": "native",
            "nlive": 8,
            "sample": "prior",
            "max_iterations": 2,
            "seed": 123,
        },
        nlive=8,
        sample="prior",
        dlogz=1.0e-12,
        n_processes=1,
        queue_size=None,
        seed=123,
        progress_label="native_test",
    )

    assert result.parameter_names == [
        "flux"
    ]
    assert result.samples.shape == (
        10,
        1,
    )
    assert result.logl.shape == (
        10,
    )
    assert result.logwt.shape == (
        10,
    )
    assert result.logz.shape == (
        10,
    )
    assert np.all(
        np.isfinite(
            result.samples
        )
    )
    assert np.all(
        np.isfinite(
            result.logl
        )
    )
    assert np.isfinite(
        result.final_logz
    )
    assert np.sum(
        result.weights
    ) == pytest.approx(
        1.0
    )
