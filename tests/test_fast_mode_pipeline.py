"""The fast mode (AD-057) through the single-observation fit path."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from baymorph.pipeline.config import resolve_config
from baymorph.pipeline.likelihood_state import RealFitLikelihoodState
from baymorph.pipeline.real_fit import (
    _execute_single_model_fit,
    _inference_summary,
    _make_model_factory,
    save_fit_summary_text,
)
from baymorph.runner import BayMorphRunner
from baymorph.simulation.psf import GaussianPSF
from baymorph.stats import weighted_quantile

TRUTH = {
    "flux": 4000.0,
    "re": 4.0,
    "n": 1.5,
    "q": 0.7,
    "theta": 0.5,
    "x0": 16.0,
    "y0": 16.0,
    "background_level": 0.0,
}

PRIORS = {
    "single_sersic": {
        "flux": {"type": "log_uniform", "lower": 500.0, "upper": 50000.0},
        "re": {"type": "log_uniform", "lower": 1.0, "upper": 15.0},
        "n": {"type": "uniform", "lower": 0.5, "upper": 6.0},
        "q": {"type": "uniform", "lower": 0.3, "upper": 1.0},
        "theta": {"type": "uniform", "lower": "-pi/2", "upper": "pi/2"},
        "x0": {
            "type": "uniform",
            "lower": -3.0,
            "upper": 3.0,
            "relative_to": "target_x",
        },
        "y0": {
            "type": "uniform",
            "lower": -3.0,
            "upper": 3.0,
            "relative_to": "target_y",
        },
    }
}

BACKGROUND = {
    "model": "constant",
    "priors": {"background_level": {"type": "uniform", "lower": -5.0, "upper": 5.0}},
}


@pytest.fixture(scope="module")
def fast_products():
    shape = (33, 33)
    sigma = 1.0
    psf = GaussianPSF(fwhm=2.0)
    factory = _make_model_factory("single_sersic", psf, background_cfg=BACKGROUND)
    expected = factory(**TRUTH).render(shape)
    rng = np.random.default_rng(5)
    data = expected + rng.normal(0.0, sigma, shape)
    image_products = SimpleNamespace(
        prepared_data=data,
        fitting_mask=np.zeros(shape, dtype=bool),
    )
    state = RealFitLikelihoodState(
        likelihood_type="gaussian",
        noise=np.full(shape, sigma),
        metadata={"type": "gaussian", "normalization_complete": True},
    )

    return _execute_single_model_fit(
        model_name="single_sersic",
        image_products=image_products,
        psf=psf,
        prior_cfg=PRIORS,
        background_cfg=BACKGROUND,
        references={"target_x": 16.0, "target_y": 16.0},
        likelihood_cfg={"type": "gaussian", "include_normalization": True},
        likelihood_state=state,
        sampler_cfg={"name": "nautilus"},
        mode="fast",
        fast_cfg={
            "seed": 1,
            "draws": 1000,
            "scan": 128,
            "execution": {"processes": 1, "progress": False},
        },
    )


def test_fast_fit_recovers_the_truth(fast_products):
    posterior = fast_products.posterior
    diagnostics = posterior.diagnostics

    assert posterior.evidence is None
    assert diagnostics["backend"] == "fast"
    assert diagnostics["reliable"] is True

    for index, name in enumerate(posterior.parameter_names):
        lower, median, upper = (
            weighted_quantile(posterior.samples[:, index], level, posterior.weights)
            for level in (0.16, 0.5, 0.84)
        )
        half_width = 0.5 * (upper - lower)

        assert abs(median - TRUTH[name]) < 4.0 * half_width, name


def test_fast_fit_reports_its_mode(fast_products, tmp_path):
    config = {"fit": {"mode": "fast"}}
    summary = _inference_summary(config, fast_products)

    assert summary["mode"] == "fast"
    assert summary["evidence"] is None
    assert summary["reliable"] is True
    assert summary["pareto_k"] == fast_products.posterior.diagnostics["pareto_k"]
    json.dumps(summary)

    path = tmp_path / "fit_summary.txt"
    save_fit_summary_text(path, model_title="Single Sersic", products=fast_products)
    text = path.read_text()

    assert "Inference mode: fast (approximate posterior, no Bayesian evidence)" in text
    assert "Posterior approximation reliable: yes" in text
    assert "logZ = None (the fast mode computes no evidence)" in text


def _config(**fit):
    return {
        "object": {"name": "target"},
        "data": {"image_path": "image.fits"},
        "outputs": {"root": "runs"},
        "fit": fit,
    }


def test_configuration_defaults_keep_the_evidence_mode():
    resolved = resolve_config(_config())

    assert resolved["fit"]["mode"] == "evidence"
    assert resolved["fit"]["fast"]["draws"] == 2000


def test_runner_switches_the_mode_and_refuses_comparison():
    runner = BayMorphRunner(_config(), verbose=False)
    runner.set_inference_mode("FAST")

    assert runner.config["fit"]["mode"] == "fast"

    with pytest.raises(ValueError, match="Unknown fit.mode"):
        runner.set_inference_mode("quick")

    compared = BayMorphRunner(
        _config(
            models=["single_sersic", "double_sersic"],
            comparison={"enabled": True},
        ),
        verbose=False,
    )

    with pytest.raises(ValueError, match="Bayesian evidence"):
        compared.set_inference_mode("fast")


def test_configuration_refuses_fast_comparison():
    with pytest.raises(ValueError, match="Bayesian evidence"):
        resolve_config(
            _config(
                mode="fast",
                models=["single_sersic", "double_sersic"],
                comparison={"enabled": True},
            )
        )


def test_joint_path_refuses_the_fast_mode():
    from baymorph.pipeline.multi_observation_fit import _refuse_fast_mode

    _refuse_fast_mode({"fit": {"models": ["single_sersic"]}})

    with pytest.raises(ValueError, match="joint fits"):
        _refuse_fast_mode({"fit": {"mode": "fast"}})

    with pytest.raises(ValueError, match="joint fits"):
        _refuse_fast_mode({"fit": {"fast": {"draws": 500}}})


def test_fast_fits_have_their_own_directory():
    from baymorph.pipeline.outputs import pipeline_paths

    evidence = pipeline_paths(_config())["fits"]["single_sersic"]["dir"]
    fast = pipeline_paths(_config(mode="fast"))["fits"]["single_sersic"]["dir"]

    assert fast == evidence / "fast"


def test_unreliable_fast_fits_are_marked_and_refused(fast_products, tmp_path):
    import copy

    from baymorph.cli import _unreliable_fast_fits
    from baymorph.pipeline.real_fit import (
        FIT_STATUS_COMPLETE,
        FIT_STATUS_UNRELIABLE,
        fit_status,
        load_real_fit_products,
    )

    assert fit_status(fast_products) == FIT_STATUS_COMPLETE
    assert _unreliable_fast_fits(fast_products) == []

    unreliable = copy.deepcopy(fast_products)
    unreliable.posterior.diagnostics["reliable"] = False

    assert fit_status(unreliable) == FIT_STATUS_UNRELIABLE
    assert _unreliable_fast_fits({"single_sersic": unreliable}) == ["single_sersic"]

    config = _config(mode="fast")
    config["outputs"] = {"root": str(tmp_path)}
    from baymorph.pipeline.outputs import pipeline_paths

    complete = pipeline_paths(config)["fits"]["single_sersic"]["complete"]
    complete.parent.mkdir(parents=True)
    complete.write_text(json.dumps({"status": FIT_STATUS_UNRELIABLE}))

    with pytest.raises(ValueError, match="unreliable"):
        load_real_fit_products(config, "single_sersic")
