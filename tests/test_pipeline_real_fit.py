import json

from types import SimpleNamespace

import numpy as np
import pytest

from baymorph.fitting.image_problem import ImageInferenceProblem
from baymorph.fitting import (
    Parameter,
    ParameterSet,
)
from baymorph.results import (
    EvidenceEstimate,
    InferenceResult,
    PosteriorSamples,
)
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

    posterior = InferenceResult(
        posterior=PosteriorSamples(
            samples=np.array(
                [
                    [1.0, 10.0],
                    [2.0, 20.0],
                ]
            ),
            weights=np.array(
                [
                    0.25,
                    0.75,
                ]
            ),
            parameter_names=[
                "flux",
                "re",
            ],
            log_likelihood=np.array(
                [
                    -10.0,
                    -5.0,
                ]
            ),
        ),
        evidence=EvidenceEstimate(
            logz=-0.5,
            uncertainty=0.2,
            method="test",
            uncertainty_kind="test",
        ),
        provenance={
            "backend": "test",
        },
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
    assert loaded.posterior.weights == pytest.approx(
        posterior.weights
    )
    assert loaded.posterior.logz == pytest.approx(
        posterior.logz
    )
    assert loaded.posterior.log_likelihood == pytest.approx(
        posterior.log_likelihood
    )
    assert loaded.posterior.logz_error == pytest.approx(
        posterior.logz_error
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
        self.logz = float(logz)


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
        flux=100.0,
        re=5.0,
        n=2.0,
        q=0.8,
        theta=0.0,
        x0=15.0,
        y0=15.0,
        sky=1.0,
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
        flux_bulge=100.0,
        re_bulge=3.0,
        n_bulge=2.0,
        q_bulge=0.9,
        theta_bulge=0.1,
        flux_disk=50.0,
        re_disk=8.0,
        n_disk=1.0,
        q_disk=0.8,
        theta_disk=0.0,
        x0=15.0,
        y0=15.0,
        sky=1.0,
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

def test_run_sampler_builds_problem_and_backend(
    monkeypatch,
):
    data = np.ones(
        (5, 5),
        dtype=float,
    )
    noise = np.full(
        data.shape,
        2.0,
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
                10.0,
            )
        ]
    )

    expected_result = object()
    captured = {}

    class DummyBackend:
        def fit(
            self,
            *,
            progress_label=None,
        ):
            captured["progress_label"] = progress_label
            return expected_result

    def fake_make_sampler(
        config,
        *,
        problem,
    ):
        captured["sampler_cfg"] = config
        captured["problem"] = problem
        return DummyBackend()

    def model_factory(
        *,
        flux,
    ):
        del flux
        return None

    monkeypatch.setattr(
        "baymorph.pipeline.real_fit.make_sampler",
        fake_make_sampler,
    )

    sampler_cfg = {
        "name": "dynesty",
        "mode": "static",
        "init": {
            "nlive": 40,
            "sample": "rwalk",
            "rstate": 123,
        },
        "run": {
            "dlogz": 0.5,
            "print_progress": False,
        },
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
        progress_label="single_sersic",
    )

    assert result is expected_result
    assert captured["sampler_cfg"] is sampler_cfg
    assert isinstance(
        captured["problem"],
        ImageInferenceProblem,
    )
    assert captured["problem"].parameters is parameters
    assert isinstance(
        captured["problem"].likelihood,
        GaussianImageLikelihood,
    )
    assert np.array_equal(
        captured["problem"].likelihood.data,
        data,
    )
    assert captured["progress_label"] == "single_sersic"


def _run_two_model_fit_for_comparison_test(
    monkeypatch,
    *,
    comparison_enabled,
    comparison_metric="bayes_factor",
):
    resolved = {
        "data": {
            "image_path": "missing-image.fits",
        },
        "fit": {
            "models": [
                "single_sersic",
                "double_sersic",
            ],
            "priors": {},
            "likelihood": {
                "type": "gaussian",
            },
            "sampler": {},
            "comparison": {
                "enabled": comparison_enabled,
                "metric": comparison_metric,
            },
        },
    }

    image_products = SimpleNamespace()
    saved_comparisons = []

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
        lambda config: image_products,
    )
    monkeypatch.setattr(
        "baymorph.pipeline.real_fit._load_psf_from_config",
        lambda config: object(),
    )
    monkeypatch.setattr(
        "baymorph.pipeline.real_fit._execute_single_model_fit",
        lambda **kwargs: RealFitProducts(
            model_name=kwargs["model_name"],
            posterior=SimpleNamespace(
                logz=1.0,
                logz_error=0.1,
            ),
            best_parameters={},
            best_model_image=np.zeros((2, 2)),
            residual=np.zeros((2, 2)),
            norm_residual=np.zeros((2, 2)),
            chi2=1.0,
            ndof=1,
            reduced_chi2=1.0,
            noise=np.ones((2, 2)),
        ),
    )
    monkeypatch.setattr(
        "baymorph.pipeline.real_fit.save_real_fit_products",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "baymorph.pipeline.real_fit.save_bayesian_model_comparison",
        lambda config, comparison: saved_comparisons.append(comparison),
    )

    products = run_real_fit_from_config({})

    return products, saved_comparisons


def test_multiple_models_skip_comparison_when_disabled(
    monkeypatch,
):
    """Multiple fits must not imply model comparison."""
    products, saved_comparisons = (
        _run_two_model_fit_for_comparison_test(
            monkeypatch,
            comparison_enabled=False,
        )
    )

    assert set(products) == {
        "single_sersic",
        "double_sersic",
    }
    assert saved_comparisons == []


def test_multiple_models_compute_comparison_when_enabled(
    monkeypatch,
):
    """Enabled comparison must compute and persist Bayes factors."""
    products, saved_comparisons = (
        _run_two_model_fit_for_comparison_test(
            monkeypatch,
            comparison_enabled=True,
        )
    )

    assert set(products) == {
        "single_sersic",
        "double_sersic",
    }
    assert len(saved_comparisons) == 1
    assert "bayes_factor" in saved_comparisons[0]


def test_model_comparison_rejects_unknown_metric(
    monkeypatch,
):
    """Unsupported comparison metrics must fail explicitly."""
    with pytest.raises(
        ValueError,
        match="Unsupported model-comparison metric",
    ):
        _run_two_model_fit_for_comparison_test(
            monkeypatch,
            comparison_enabled=True,
            comparison_metric="aic",
        )


def test_build_parameter_set_sersic_point_source():
    """Build the Sérsic plus point-source parameter set."""
    parameter_names, parameters, _, _ = (
        _build_parameter_set(
            "sersic_point_source",
            data_fit=np.ones((10, 10)),
            noise_fit=np.ones((10, 10)),
            mask_fit=np.zeros((10, 10), dtype=bool),
            target_x=5,
            target_y=5,
        )
    )

    assert parameter_names == [
        "flux_sersic",
        "re",
        "n",
        "q",
        "theta",
        "flux_point_source",
        "x0",
        "y0",
        "sky",
    ]

    assert parameters.names == parameter_names


def test_make_model_factory_sersic_point_source():
    """The pipeline must resolve the Sérsic plus point-source factory."""
    factory = _make_model_factory(
        "sersic_point_source",
        DeltaPSF(),
    )

    model = factory(
        flux_sersic=100.0,
        re=3.0,
        n=2.0,
        q=0.8,
        theta=0.0,
        flux_point_source=25.0,
        x0=15.0,
        y0=15.0,
        sky=1.0,
    )

    image = model.render((31, 31))

    assert image.shape == (31, 31)
    assert np.all(np.isfinite(image))


def test_build_parameter_set_bulge_disk():
    """Build the Sérsic-bulge plus exponential-disc parameter set."""
    parameter_names, parameters, _, _ = (
        _build_parameter_set(
            "bulge_disk",
            data_fit=np.ones((10, 10)),
            noise_fit=np.ones((10, 10)),
            mask_fit=np.zeros((10, 10), dtype=bool),
            target_x=5,
            target_y=5,
        )
    )

    assert parameter_names == [
        "flux_bulge",
        "re_bulge",
        "n_bulge",
        "q_bulge",
        "theta_bulge",
        "flux_disk",
        "re_disk",
        "q_disk",
        "theta_disk",
        "x0",
        "y0",
        "sky",
    ]

    assert "n_disk" not in parameter_names
    assert parameters.names == parameter_names


def test_make_model_factory_bulge_disk():
    """The pipeline must resolve the production bulge-disc factory."""
    factory = _make_model_factory(
        "bulge_disk",
        DeltaPSF(),
    )

    model = factory(
        flux_bulge=100.0,
        re_bulge=3.0,
        n_bulge=3.0,
        q_bulge=0.9,
        theta_bulge=0.1,
        flux_disk=200.0,
        re_disk=10.0,
        q_disk=0.7,
        theta_disk=0.2,
        x0=15.0,
        y0=15.0,
        sky=1.0,
    )

    image = model.render((31, 31))

    assert image.shape == (31, 31)
    assert np.all(np.isfinite(image))


def test_fit_summary_preserves_complete_sampler_configuration(
    tmp_path,
):
    """Fit summaries should preserve the complete sampler configuration."""
    from baymorph.pipeline.real_fit import save_real_fit_products
    from baymorph.results import (
        EvidenceEstimate,
        InferenceResult,
        PosteriorSamples,
        RealFitProducts,
    )

    sampler_config = {
        "name": "dynesty",
        "mode": "static",
        "init": {
            "nlive": 500,
            "sample": "rwalk",
            "future_init_option": 17,
            "rstate": np.random.default_rng(
                42
            ),
        },
        "run": {
            "dlogz": 0.05,
            "print_progress": False,
            "future_run_option": "value",
        },
    }

    config = resolve_config(
        {
            "object": {
                "name": "test",
            },
            "data": {
                "image_path": "image.fits",
            },
            "outputs": {
                "root": str(
                    tmp_path
                ),
            },
            "fit": {
                "models": [
                    "single_sersic",
                ],
                "sampler": sampler_config,
            },
        }
    )

    posterior = InferenceResult(
        posterior=PosteriorSamples(
            samples=np.asarray(
                [
                    [1.0],
                    [2.0],
                ]
            ),
            weights=np.asarray(
                [
                    0.5,
                    0.5,
                ]
            ),
            parameter_names=[
                "flux",
            ],
            log_likelihood=np.asarray(
                [
                    -1.0,
                    -0.5,
                ]
            ),
        ),
        evidence=EvidenceEstimate(
            logz=-10.0,
            uncertainty=0.1,
            method="dynesty",
        ),
    )

    products = RealFitProducts(
        posterior=posterior,
        best_parameters={
            "flux": 2.0,
        },
        best_model_image=np.zeros(
            (2, 2)
        ),
        residual=np.zeros(
            (2, 2)
        ),
        norm_residual=np.zeros(
            (2, 2)
        ),
        chi2=1.0,
        ndof=3,
        reduced_chi2=1.0 / 3.0,
        model_name="single_sersic",
    )

    save_real_fit_products(
        config,
        "single_sersic",
        products,
    )

    summary_path = (
        tmp_path
        / "test"
        / "fits"
        / "single_sersic"
        / "fit_summary.json"
    )

    summary = json.loads(
        summary_path.read_text()
    )

    assert summary["sampler"]["name"] == "dynesty"
    assert summary["sampler"]["mode"] == "static"

    assert summary["sampler"]["init"]["nlive"] == 500
    assert summary["sampler"]["init"]["sample"] == "rwalk"
    assert summary["sampler"]["init"]["future_init_option"] == 17

    assert summary["sampler"]["init"]["rstate"] == {
        "__runtime_object__": True,
        "type": "numpy.random.Generator",
        "bit_generator": "PCG64",
    }

    assert summary["sampler"]["run"] == {
        "dlogz": 0.05,
        "print_progress": False,
        "future_run_option": "value",
    }
