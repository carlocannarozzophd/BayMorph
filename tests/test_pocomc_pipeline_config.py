"""Tests for pocoMC production checkpoint-safety support."""

from __future__ import annotations

import pytest

from baymorph.pipeline.real_fit import (
    _prepare_pocomc_sampler_config,
)


def _prepare_checkpoint(
    tmp_path,
    monkeypatch,
    *,
    model_name="single_sersic",
):
    """Resolve one signed test resume target without building a real problem."""
    monkeypatch.setattr(
        "baymorph.pipeline.real_fit._real_fit_problem_signature",
        lambda **kwargs: f"signature:{kwargs['model_name']}",
    )

    return _prepare_pocomc_sampler_config(
        {
            "name": "pocomc",
            "execution": {
                "processes": 1,
                "progress": False,
            },
            "init": {
                "n_effective": 512,
            },
            "run": {
                "resume_state_path": "auto",
            },
        },
        model_name=model_name,
        problem=object(),
        likelihood_cfg={
            "type": "gaussian",
            "include_normalization": True,
        },
        data_fit=None,
        noise_fit=None,
        mask_fit=None,
        correlation_kernel_fit=None,
        checkpoint_directory=(
            tmp_path
            / model_name
        ),
    )


def test_pocomc_sampler_config_is_unchanged_without_resume_state_path() -> None:
    """A pocoMC config with no resume request must pass through untouched."""
    sampler_cfg = {
        "name": "pocomc",
        "init": {
            "n_effective": 512,
        },
        "run": {},
    }

    resolved = _prepare_pocomc_sampler_config(
        sampler_cfg,
        model_name="single_sersic",
        problem=object(),
        likelihood_cfg=None,
        data_fit=None,
        noise_fit=None,
        mask_fit=None,
        correlation_kernel_fit=None,
        checkpoint_directory=None,
    )

    assert resolved == sampler_cfg


def test_pocomc_auto_resume_state_path_is_model_specific(
    tmp_path,
    monkeypatch,
) -> None:
    """Automatic resume-state paths and signatures must live per fitted model."""
    single = _prepare_checkpoint(
        tmp_path,
        monkeypatch,
        model_name="single_sersic",
    )
    double = _prepare_checkpoint(
        tmp_path,
        monkeypatch,
        model_name="double_sersic",
    )

    single_path = single["run"]["resume_state_path"]
    double_path = double["run"]["resume_state_path"]

    assert single_path != double_path
    assert single_path.endswith(
        "single_sersic/pocomc_state.h5"
    )
    assert double_path.endswith(
        "double_sersic/pocomc_state.h5"
    )
    assert (
        tmp_path
        / "single_sersic"
        / "pocomc_state.h5.target.json"
    ).is_file()


def test_pocomc_auto_resume_state_path_requires_checkpoint_directory() -> None:
    """'auto' cannot resolve to a path without a pipeline-managed directory."""
    with pytest.raises(
        ValueError,
        match="requires a pipeline-managed",
    ):
        _prepare_pocomc_sampler_config(
            {
                "name": "pocomc",
                "init": {},
                "run": {
                    "resume_state_path": "auto",
                },
            },
            model_name="single_sersic",
            problem=object(),
            likelihood_cfg=None,
            data_fit=None,
            noise_fit=None,
            mask_fit=None,
            correlation_kernel_fit=None,
            checkpoint_directory=None,
        )


def test_pocomc_resume_rejects_target_mismatch(
    tmp_path,
    monkeypatch,
) -> None:
    """An existing resume state must not resume against another target."""
    resolved = _prepare_checkpoint(
        tmp_path,
        monkeypatch,
    )
    state_file = (
        tmp_path
        / "single_sersic"
        / "pocomc_state.h5"
    )
    state_file.touch()
    monkeypatch.setattr(
        "baymorph.pipeline.real_fit._real_fit_problem_signature",
        lambda **kwargs: "changed-target",
    )

    with pytest.raises(
        RuntimeError,
        match="does not match",
    ):
        _prepare_pocomc_sampler_config(
            resolved,
            model_name="single_sersic",
            problem=object(),
            likelihood_cfg={
                "type": "gaussian",
                "include_normalization": True,
            },
            data_fit=None,
            noise_fit=None,
            mask_fit=None,
            correlation_kernel_fit=None,
            checkpoint_directory=state_file.parent,
        )


def test_pocomc_resume_rejects_state_with_no_recorded_signature(
    tmp_path,
    monkeypatch,
) -> None:
    """A resume state with no BayMorph signature must never be trusted."""
    monkeypatch.setattr(
        "baymorph.pipeline.real_fit._real_fit_problem_signature",
        lambda **kwargs: "signature",
    )

    state_file = tmp_path / "pocomc_state.h5"
    state_file.touch()

    with pytest.raises(
        RuntimeError,
        match="no BayMorph target signature",
    ):
        _prepare_pocomc_sampler_config(
            {
                "name": "pocomc",
                "init": {},
                "run": {
                    "resume_state_path": str(
                        state_file
                    ),
                },
            },
            model_name="single_sersic",
            problem=object(),
            likelihood_cfg=None,
            data_fit=None,
            noise_fit=None,
            mask_fit=None,
            correlation_kernel_fit=None,
            checkpoint_directory=None,
        )
