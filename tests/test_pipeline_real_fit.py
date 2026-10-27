import json
from copy import deepcopy
from pathlib import Path

from types import SimpleNamespace

import numpy as np
import pytest

from baymorph.results import (
    EvidenceEstimate,
    InferenceResult,
    PosteriorSamples,
)
from baymorph.pipeline.config import (
    DEFAULT_CONFIG,
    resolve_config,
)

from baymorph.pipeline.real_fit import (
    RealFitProducts,
    _build_parameter_set,
    _make_model_factory,
    _compute_best_model,
    _compute_fit_statistics,
    _compute_bayesian_model_comparison,
    _resolved_boundary_priors,
    _single_model_fit_config,
    load_real_fit_products,
    run_real_fit_from_config,
    save_real_fit_products,
)


def _prior_config_with_explicit_fluxes(
    model_name: str,
) -> dict:
    """Return package defaults plus explicit test flux priors."""
    priors = deepcopy(
        DEFAULT_CONFIG["fit"]["priors"]
    )

    flux_parameters = {
        "single_sersic": (
            "flux",
        ),
        "double_sersic": (
            "flux_bulge",
            "flux_disk",
        ),
        "sersic_point_source": (
            "flux_sersic",
            "flux_point_source",
        ),
        "bulge_disk": (
            "flux_bulge",
            "flux_disk",
        ),
    }

    for parameter_name in flux_parameters[
        model_name
    ]:
        priors[
            model_name
        ][
            parameter_name
        ] = {
            "type": "uniform",
            "lower": 1.0,
            "upper": 1000.0,
        }

    return priors


def _test_prior_references() -> dict[str, float]:
    """Return explicit detector-seed references for builder tests."""
    return {
        "target_x": 5.0,
        "target_y": 5.0,
    }


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
        "baymorph.pipeline.real_fit.validate_fit_priors",
        lambda config: None,
    )
    output_writes = []
    monkeypatch.setattr(
        "baymorph.pipeline.real_fit.make_output_tree",
        lambda config: output_writes.append(config),
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

    assert output_writes == []


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
        "baymorph.pipeline.real_fit.validate_fit_priors",
        lambda config: None,
    )
    output_writes = []
    monkeypatch.setattr(
        "baymorph.pipeline.real_fit.make_output_tree",
        lambda config: output_writes.append(config),
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

    assert output_writes == []


def test_real_fit_products_preserve_posterior_round_trip(
    tmp_path,
    monkeypatch,
):
    config = resolve_config(
        {
            "object": {
                "name": "example_galaxy",
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
    includes_pixel_response = False

    def kernel(self):
        psf = np.zeros((9, 9))
        psf[4, 4] = 1.0
        return psf


class DummyPosterior:
    def __init__(self, logz, logz_error):
        self.logz = float(logz)
        self.logz_error = float(logz_error)
        self.evidence = EvidenceEstimate(
            logz=logz,
            uncertainty=logz_error,
            method="test-evidence",
            uncertainty_kind="test-standard-error",
        )
        self.evidence_method = self.evidence.method


def test_compute_best_model_uses_best_parameters():
    def model_factory(a, b):
        return DummyModel(a, b)

    image, best_parameters, source = _compute_best_model(
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
    # A model without a residual background is its own source part.
    assert source is image


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
    parameter_names, parameters = _build_parameter_set(
        "single_sersic",
        prior_cfg=_prior_config_with_explicit_fluxes(
            "single_sersic"
        ),
        background_cfg=DEFAULT_CONFIG["fit"]["background"],
        references=_test_prior_references(),
    )

    assert len(parameter_names) == 8
    assert parameter_names[0] == "flux"
    assert parameters.names[0] == "flux"


def test_build_parameter_set_double_sersic():
    parameter_names, parameters = _build_parameter_set(
        "double_sersic",
        prior_cfg=_prior_config_with_explicit_fluxes(
            "double_sersic"
        ),
        background_cfg=DEFAULT_CONFIG["fit"]["background"],
        references=_test_prior_references(),
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
        "background_level",
    ]

    assert parameters.names == parameter_names


def test_single_model_fit_config_resolves_nautilus_checkpoint_per_model(
    tmp_path,
):
    """Each model must own an explicit Nautilus checkpoint path."""
    priors = _prior_config_with_explicit_fluxes(
        "single_sersic"
    )
    priors["double_sersic"] = (
        _prior_config_with_explicit_fluxes(
            "double_sersic"
        )["double_sersic"]
    )
    resolved = resolve_config(
        {
            "object": {
                "name": "example_galaxy",
            },
            "data": {
                "image_path": "data/image.fits",
            },
            "outputs": {
                "root": str(
                    tmp_path
                    / "outputs"
                ),
            },
            "fit": {
                "models": [
                    "single_sersic",
                    "double_sersic",
                ],
                "priors": priors,
                "sampler": {
                    "name": "nautilus",
                    "init": {
                        "filepath": "auto",
                        "resume": True,
                    },
                },
                "comparison": {
                    "enabled": True,
                    "metric": "bayes_factor",
                },
            },
        }
    )

    single = _single_model_fit_config(
        resolved,
        "single_sersic",
    )
    double = _single_model_fit_config(
        resolved,
        "double_sersic",
    )

    assert resolved["fit"]["sampler"]["init"]["filepath"] == "auto"
    assert single["fit"]["models"] == [
        "single_sersic"
    ]
    assert double["fit"]["models"] == [
        "double_sersic"
    ]
    assert single["fit"]["comparison"]["enabled"] is False
    assert double["fit"]["comparison"]["enabled"] is False
    assert Path(
        single["fit"]["sampler"]["init"]["filepath"]
    ).name == "nautilus_checkpoint.hdf5"
    assert Path(
        single["fit"]["sampler"]["init"]["filepath"]
    ).parent.name == "single_sersic"
    assert Path(
        double["fit"]["sampler"]["init"]["filepath"]
    ).parent.name == "double_sersic"


def test_boundary_priors_are_resolved_for_every_sampled_parameter():
    """Boundary diagnostics must use absolute and component-specific priors."""
    priors = _prior_config_with_explicit_fluxes(
        "double_sersic"
    )
    resolved = _resolved_boundary_priors(
        model_name="double_sersic",
        prior_cfg=priors,
        background_cfg=DEFAULT_CONFIG["fit"]["background"],
        references={
            "target_x": 10.0,
            "target_y": 20.0,
        },
    )

    assert resolved["x0"] == {
        "type": "uniform",
        "lower": 2.0,
        "upper": 18.0,
    }
    assert resolved["y0"] == {
        "type": "uniform",
        "lower": 12.0,
        "upper": 28.0,
    }
    assert resolved["q_bulge"] == resolved["q_disk"]
    assert resolved["theta_bulge"] == resolved["theta_disk"]
    assert "background_level" in resolved


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
        background_level=1.0,
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
        background_level=1.0,
    )

    image = model.render((31, 31))

    assert image.shape == (31, 31)
    assert np.all(np.isfinite(image))


def test_compute_bayesian_model_comparison():
    single = RealFitProducts(
        posterior=DummyPosterior(10.0, 0.2),
        best_parameters={},
        best_model_image=np.zeros((2, 2)),
        residual=np.zeros((2, 2)),
        norm_residual=np.zeros((2, 2)),
        chi2=1.0,
        ndof=1,
        reduced_chi2=1.0,
    )

    double = RealFitProducts(
        posterior=DummyPosterior(13.0, 0.3),
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
    assert comparison["delta_logz_error"][
        "double_sersic_minus_single_sersic"
    ] == pytest.approx(
        np.hypot(
            0.2,
            0.3,
        )
    )
    assert comparison["evidence_method"] == {
        "single_sersic": "test-evidence",
        "double_sersic": "test-evidence",
    }
    assert comparison["bayes_factor"]["double_sersic_over_single_sersic"] == pytest.approx(
        np.exp(3.0)
    )


def _run_two_model_fit_for_comparison_test(
    monkeypatch,
    *,
    comparison_enabled,
    comparison_metric="bayes_factor",
    output_root=None,
    complete_model=None,
):
    output_root = (
        Path("unused")
        if output_root is None
        else Path(
            output_root
        )
    )
    resolved = {
        "data": {
            "image_path": "missing-image.fits",
        },
        "outputs": {
            "root": str(
                output_root
            ),
            "overwrite": False,
        },
        "fit": {
            "models": [
                "single_sersic",
                "double_sersic",
            ],
            "priors": {},
            # A complete likelihood without a calibration keeps the test on
            # the comparison; the noise model of AD-047 has its own tests.
            "likelihood": {
                "type": "poisson_gaussian",
            },
            "sampler": {},
            "comparison": {
                "enabled": comparison_enabled,
                "metric": comparison_metric,
            },
        },
    }

    image_products = SimpleNamespace(
        prepared_data=np.zeros(
            (2, 2),
            dtype=float,
        ),
        noise=np.ones(
            (2, 2),
            dtype=float,
        ),
        fitting_mask=np.zeros(
            (2, 2),
            dtype=bool,
        ),
        cutout_bounds=None,
    )
    saved_comparisons = []

    if complete_model is not None:
        completion_path = (
            output_root
            / complete_model
            / "COMPLETE.json"
        )
        completion_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        completion_path.write_text(
            "{}\n"
        )

    monkeypatch.setattr(
        "baymorph.pipeline.real_fit.resolve_config",
        lambda config: resolved,
    )
    monkeypatch.setattr(
        "baymorph.pipeline.real_fit.validate_fit_priors",
        lambda config: None,
    )
    monkeypatch.setattr(
        "baymorph.pipeline.real_fit._target_reference_coordinates",
        lambda **kwargs: {
            "target_x": 0.0,
            "target_y": 0.0,
        },
    )
    monkeypatch.setattr(
        "baymorph.pipeline.real_fit.make_output_tree",
        lambda config: None,
    )
    monkeypatch.setattr(
        "baymorph.pipeline.real_fit.save_fit_resolved_config",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "baymorph.pipeline.real_fit.pipeline_paths",
        lambda config: {
            "fits": {
                model: {
                    "dir": output_root
                    / model,
                    "complete": output_root
                    / model
                    / "COMPLETE.json",
                }
                for model in (
                    "single_sersic",
                    "double_sersic",
                )
            }
        },
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
        "baymorph.pipeline.real_fit._resolved_boundary_priors",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(
        "baymorph.pipeline.real_fit._require_prepared_centre_within_priors",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "baymorph.pipeline.real_fit._write_fit_completion_marker",
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


def test_completed_model_is_protected_when_overwrite_is_false(
    tmp_path,
    monkeypatch,
):
    """A completion marker must prevent accidental fit replacement."""
    with pytest.raises(
        FileExistsError,
        match="complete fit already exists",
    ):
        _run_two_model_fit_for_comparison_test(
            monkeypatch,
            comparison_enabled=False,
            output_root=tmp_path,
            complete_model="single_sersic",
        )


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
    parameter_names, parameters = _build_parameter_set(
        "sersic_point_source",
        prior_cfg=_prior_config_with_explicit_fluxes(
            "sersic_point_source"
        ),
        background_cfg=DEFAULT_CONFIG["fit"]["background"],
        references=_test_prior_references(),
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
        "background_level",
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
        background_level=1.0,
    )

    image = model.render((31, 31))

    assert image.shape == (31, 31)
    assert np.all(np.isfinite(image))


def test_build_parameter_set_bulge_disk():
    """Build the Sérsic-bulge plus exponential-disc parameter set."""
    parameter_names, parameters = _build_parameter_set(
        "bulge_disk",
        prior_cfg=_prior_config_with_explicit_fluxes(
            "bulge_disk"
        ),
        background_cfg=DEFAULT_CONFIG["fit"]["background"],
        references=_test_prior_references(),
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
        "background_level",
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
        background_level=1.0,
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
