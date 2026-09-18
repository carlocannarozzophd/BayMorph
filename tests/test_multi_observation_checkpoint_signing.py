"""Tests for joint multi-observation checkpoint-safety support."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from baymorph.pipeline.multi_observation_fit import (
    _prepare_multi_observation_emcee_sampler_config,
    _prepare_multi_observation_nautilus_sampler_config,
    _prepare_multi_observation_pocomc_sampler_config,
    run_multi_observation_fit_from_config,
)


def _patch_signature(monkeypatch, value="signature:multi"):
    """Replace the real multi-observation signature with a fixed stub."""
    monkeypatch.setattr(
        "baymorph.pipeline.multi_observation_fit."
        "_multi_observation_problem_signature",
        lambda **kwargs: value,
    )


def test_multi_observation_emcee_injects_checkpoint_signature(
    monkeypatch,
) -> None:
    """Emcee checkpointing must be signed exactly like a single-obs fit."""
    _patch_signature(monkeypatch)

    resolved = _prepare_multi_observation_emcee_sampler_config(
        {
            "name": "emcee",
            "init": {
                "checkpoint_path": "/tmp/does-not-matter.h5",
            },
        },
        model_name="single_sersic",
        context=object(),
        checkpoint_directory=None,
    )

    assert (
        resolved["init"]["checkpoint_signature"]
        == "signature:multi"
    )
    assert (
        resolved["init"]["checkpoint_mode"]
        == "fresh"
    )


def test_multi_observation_nautilus_auto_checkpoint_is_model_specific(
    tmp_path,
    monkeypatch,
) -> None:
    """Automatic joint-fit checkpoints and signatures must live per model."""
    _patch_signature(
        monkeypatch,
        "signature:single_sersic",
    )
    single = _prepare_multi_observation_nautilus_sampler_config(
        {
            "name": "nautilus",
            "init": {
                "filepath": "auto",
                "resume": True,
            },
        },
        model_name="single_sersic",
        context=object(),
        checkpoint_directory=(
            tmp_path
            / "single_sersic"
        ),
    )

    _patch_signature(
        monkeypatch,
        "signature:double_sersic",
    )
    double = _prepare_multi_observation_nautilus_sampler_config(
        {
            "name": "nautilus",
            "init": {
                "filepath": "auto",
                "resume": True,
            },
        },
        model_name="double_sersic",
        context=object(),
        checkpoint_directory=(
            tmp_path
            / "double_sersic"
        ),
    )

    assert (
        single["init"]["filepath"]
        != double["init"]["filepath"]
    )
    assert (
        tmp_path
        / "single_sersic"
        / "nautilus_checkpoint.hdf5.target.json"
    ).is_file()


def test_multi_observation_nautilus_resume_rejects_target_mismatch(
    tmp_path,
    monkeypatch,
) -> None:
    """A joint-fit checkpoint must not resume against another target."""
    _patch_signature(
        monkeypatch,
        "signature:original",
    )
    resolved = _prepare_multi_observation_nautilus_sampler_config(
        {
            "name": "nautilus",
            "init": {
                "filepath": "auto",
                "resume": True,
            },
        },
        model_name="single_sersic",
        context=object(),
        checkpoint_directory=(
            tmp_path
            / "single_sersic"
        ),
    )

    checkpoint = (
        tmp_path
        / "single_sersic"
        / "nautilus_checkpoint.hdf5"
    )
    checkpoint.touch()

    _patch_signature(
        monkeypatch,
        "signature:changed",
    )

    with pytest.raises(
        RuntimeError,
        match="does not match",
    ):
        _prepare_multi_observation_nautilus_sampler_config(
            resolved,
            model_name="single_sersic",
            context=object(),
            checkpoint_directory=checkpoint.parent,
        )


def test_multi_observation_pocomc_auto_resume_state_path_is_model_specific(
    tmp_path,
    monkeypatch,
) -> None:
    """Automatic joint-fit pocoMC resume states must live per model."""
    _patch_signature(monkeypatch)

    resolved = _prepare_multi_observation_pocomc_sampler_config(
        {
            "name": "pocomc",
            "run": {
                "resume_state_path": "auto",
            },
        },
        model_name="single_sersic",
        context=object(),
        checkpoint_directory=(
            tmp_path
            / "single_sersic"
        ),
    )

    assert resolved["run"]["resume_state_path"].endswith(
        "single_sersic/pocomc_state.h5"
    )
    assert (
        tmp_path
        / "single_sersic"
        / "pocomc_state.h5.target.json"
    ).is_file()


def test_multi_observation_fit_requires_outputs_overwrite_to_replace_completed_fit(
    tmp_path,
    monkeypatch,
) -> None:
    """A completed joint fit must not be silently replaced."""
    from baymorph.pipeline import multi_observation_fit

    raw_config = {
        "fit": {
            "models": [
                "single_sersic",
            ],
        },
        "observations": {
            "obs_a": {},
        },
        "outputs": {
            "root": str(
                tmp_path
            ),
            "analysis_name": "run",
        },
    }

    monkeypatch.setattr(
        multi_observation_fit,
        "_load_multi_observation_config",
        lambda config: (
            raw_config,
            tmp_path,
        ),
    )
    monkeypatch.setattr(
        multi_observation_fit,
        "_build_multi_observation_context",
        lambda raw, *, base_dir: SimpleNamespace(
            observation_configs={},
        ),
    )
    monkeypatch.setattr(
        multi_observation_fit,
        "_require_evidence_safe_likelihoods",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        multi_observation_fit,
        "_estimate_pre_fit_noise_diagnostics",
        lambda **kwargs: {},
    )

    completion_path = (
        tmp_path
        / "run"
        / "fits"
        / "single_sersic"
        / "COMPLETE.json"
    )
    completion_path.parent.mkdir(
        parents=True,
    )
    completion_path.write_text(
        "{}"
    )

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        run_multi_observation_fit_from_config(
            raw_config
        )


def test_write_multi_observation_completion_marker_records_evidence(
    tmp_path,
) -> None:
    """The joint completion marker must record the joint evidence."""
    import numpy as np

    from baymorph.pipeline.multi_observation_fit import (
        _write_multi_observation_completion_marker,
    )
    from baymorph.results import (
        EvidenceEstimate,
        InferenceResult,
        MultiObservationFitProducts,
        ObservationFitProducts,
        PosteriorSamples,
    )

    posterior = InferenceResult(
        posterior=PosteriorSamples(
            samples=np.zeros(
                (1, 1),
            ),
            weights=np.ones(
                1,
            ),
            parameter_names=[
                "flux",
            ],
            log_likelihood=np.zeros(
                1,
            ),
        ),
        evidence=EvidenceEstimate(
            logz=-12.5,
            uncertainty=0.1,
            method="nautilus",
        ),
    )

    products = MultiObservationFitProducts(
        model_name="single_sersic",
        posterior=posterior,
        best_parameters={
            "flux": 1.0,
        },
        observations={
            "obs_a": ObservationFitProducts(
                parameters={
                    "flux": 1.0,
                },
                best_model_image=np.ones(
                    (1, 1),
                ),
                residual=np.zeros(
                    (1, 1),
                ),
                norm_residual=np.zeros(
                    (1, 1),
                ),
                chi2=1.0,
                n_valid_pixels=1,
                noise=np.ones(
                    (1, 1),
                ),
            ),
        },
        chi2=1.0,
        ndof=1,
        reduced_chi2=1.0,
    )

    completion_path = (
        tmp_path
        / "COMPLETE.json"
    )

    _write_multi_observation_completion_marker(
        completion_path,
        model_name="single_sersic",
        config={
            "fit": {
                "models": [
                    "single_sersic",
                ],
            },
        },
        products=products,
    )

    assert completion_path.is_file()

    payload = json.loads(
        completion_path.read_text()
    )

    assert payload["model"] == "single_sersic"
    assert payload["logz"] == pytest.approx(-12.5)
    assert payload["logz_error"] == pytest.approx(0.1)
    assert payload["evidence_method"] == "nautilus"
