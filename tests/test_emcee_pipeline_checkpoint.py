"""Tests for pipeline-managed scientific emcee checkpoints."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from baymorph.fitting import Parameter, ParameterSet
from baymorph.fitting.emcee import EmceeFitter
from baymorph.fitting.emcee_checkpoint import (
    pop_emcee_checkpoint_request,
)
from baymorph.fitting.execution import resolve_sampler_execution
from baymorph.fitting.problem import InferenceProblem
from baymorph.pipeline.real_fit import (
    _build_image_inference_problem,
    _prepare_emcee_sampler_config,
)


class _CheckpointModelFactory:
    """Minimal model-factory state used only for fingerprint tests."""

    def __init__(
        self,
        psf,
    ) -> None:
        self.model_name = "single_sersic"
        self.background_model = "constant"
        self.morphology_parameter_names = (
            "flux",
        )
        self.psf = np.asarray(
            psf,
            dtype=float,
        )

    def __call__(
        self,
        **parameters,
    ):
        del parameters
        raise AssertionError(
            "Fingerprint construction must not render a model."
        )


class _OffsetProblem(InferenceProblem):
    """Scalar problem with controllable log-density normalization."""

    def __init__(
        self,
        offset: float,
    ) -> None:
        self.offset = float(
            offset
        )

    @property
    def ndim(self) -> int:
        return 1

    @property
    def parameter_names(self) -> list[str]:
        return [
            "x",
        ]

    def prior_transform(
        self,
        cube: np.ndarray,
    ) -> np.ndarray:
        return (
            -1.0
            + 2.0
            * np.asarray(
                cube,
                dtype=float,
            )
        )

    def log_likelihood(
        self,
        theta: np.ndarray,
    ) -> float:
        value = float(
            np.asarray(
                theta,
                dtype=float,
            )[
                0
            ]
        )

        return (
            -0.5
            * value**2
            + self.offset
        )


def _make_image_problem(
    *,
    data: np.ndarray,
    psf: np.ndarray,
    upper_flux: float = 10.0,
):
    """Build one deterministic image problem for signature tests."""
    noise = np.ones_like(
        data,
        dtype=float,
    )
    mask = np.zeros_like(
        data,
        dtype=bool,
    )
    likelihood_cfg = {
        "type": "gaussian",
        "include_normalization": True,
    }
    parameters = ParameterSet(
        [
            Parameter(
                "flux",
                1.0,
                upper_flux,
            )
        ]
    )

    problem = _build_image_inference_problem(
        model_factory=_CheckpointModelFactory(
            psf
        ),
        parameters=parameters,
        likelihood_cfg=likelihood_cfg,
        data_fit=data,
        noise_fit=noise,
        mask_fit=mask,
    )

    return (
        problem,
        likelihood_cfg,
        noise,
        mask,
    )


def _prepare_config(
    *,
    sampler_cfg,
    problem,
    likelihood_cfg,
    data,
    noise,
    mask,
    checkpoint_directory: Path,
):
    """Resolve a test emcee configuration."""
    return _prepare_emcee_sampler_config(
        sampler_cfg,
        model_name="single_sersic",
        problem=problem,
        likelihood_cfg=likelihood_cfg,
        data_fit=data,
        noise_fit=noise,
        mask_fit=mask,
        correlation_kernel_fit=None,
        checkpoint_directory=checkpoint_directory,
    )


def test_pipeline_generates_strict_emcee_problem_signature(
    tmp_path,
):
    data = np.arange(
        9,
        dtype=float,
    ).reshape(
        3,
        3,
    )
    psf = np.eye(
        3,
        dtype=float,
    )

    problem, likelihood_cfg, noise, mask = (
        _make_image_problem(
            data=data,
            psf=psf,
        )
    )

    sampler_cfg = {
        "name": "emcee",
        "execution": {
            "processes": 1,
            "progress": False,
        },
        "init": {
            "nwalkers": 12,
            "seed": 42,
            "checkpoint_path": "auto",
        },
        "run": {
            "max_steps": 10,
        },
    }
    original = deepcopy(
        sampler_cfg
    )

    first = _prepare_config(
        sampler_cfg=sampler_cfg,
        problem=problem,
        likelihood_cfg=likelihood_cfg,
        data=data,
        noise=noise,
        mask=mask,
        checkpoint_directory=tmp_path,
    )
    second = _prepare_config(
        sampler_cfg=sampler_cfg,
        problem=problem,
        likelihood_cfg=likelihood_cfg,
        data=data,
        noise=noise,
        mask=mask,
        checkpoint_directory=tmp_path,
    )

    assert sampler_cfg == original
    assert first["init"]["checkpoint_mode"] == "fresh"
    assert Path(
        first["init"]["checkpoint_path"]
    ) == (
        tmp_path
        / "emcee_checkpoint.h5"
    ).resolve()

    signature = first["init"][
        "checkpoint_signature"
    ]

    assert signature.startswith(
        "sha256:"
    )
    assert len(
        signature
    ) == len(
        "sha256:"
    ) + 64
    assert (
        second["init"]["checkpoint_signature"]
        == signature
    )

    extended_cfg = deepcopy(
        sampler_cfg
    )
    extended_cfg[
        "run"
    ][
        "max_steps"
    ] = 1000

    extended = _prepare_config(
        sampler_cfg=extended_cfg,
        problem=problem,
        likelihood_cfg=likelihood_cfg,
        data=data,
        noise=noise,
        mask=mask,
        checkpoint_directory=tmp_path,
    )

    assert (
        extended["init"]["checkpoint_signature"]
        == signature
    )

    changed_data = data.copy()
    changed_data[
        0,
        0,
    ] += 1.0

    changed_problem, changed_cfg, changed_noise, changed_mask = (
        _make_image_problem(
            data=changed_data,
            psf=psf,
        )
    )

    changed_data_result = _prepare_config(
        sampler_cfg=sampler_cfg,
        problem=changed_problem,
        likelihood_cfg=changed_cfg,
        data=changed_data,
        noise=changed_noise,
        mask=changed_mask,
        checkpoint_directory=tmp_path,
    )

    assert (
        changed_data_result["init"]["checkpoint_signature"]
        != signature
    )

    changed_psf_problem, changed_psf_cfg, psf_noise, psf_mask = (
        _make_image_problem(
            data=data,
            psf=2.0 * psf,
        )
    )

    changed_psf_result = _prepare_config(
        sampler_cfg=sampler_cfg,
        problem=changed_psf_problem,
        likelihood_cfg=changed_psf_cfg,
        data=data,
        noise=psf_noise,
        mask=psf_mask,
        checkpoint_directory=tmp_path,
    )

    assert (
        changed_psf_result["init"]["checkpoint_signature"]
        != signature
    )

    changed_prior_problem, prior_cfg, prior_noise, prior_mask = (
        _make_image_problem(
            data=data,
            psf=psf,
            upper_flux=11.0,
        )
    )

    changed_prior_result = _prepare_config(
        sampler_cfg=sampler_cfg,
        problem=changed_prior_problem,
        likelihood_cfg=prior_cfg,
        data=data,
        noise=prior_noise,
        mask=prior_mask,
        checkpoint_directory=tmp_path,
    )

    assert (
        changed_prior_result["init"]["checkpoint_signature"]
        != signature
    )

    checkpoint_file = (
        tmp_path
        / "emcee_checkpoint.h5"
    )
    checkpoint_file.write_bytes(
        b"unchanged sentinel"
    )
    before = checkpoint_file.read_bytes()

    mismatched = deepcopy(
        sampler_cfg
    )
    mismatched[
        "init"
    ][
        "checkpoint_signature"
    ] = "sha256:incorrect"

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        _prepare_config(
            sampler_cfg=mismatched,
            problem=problem,
            likelihood_cfg=likelihood_cfg,
            data=data,
            noise=noise,
            mask=mask,
            checkpoint_directory=tmp_path,
        )

    assert checkpoint_file.read_bytes() == before


def test_unresolved_auto_checkpoint_path_is_rejected():
    config = {
        "checkpoint_path": "auto",
        "checkpoint_mode": "fresh",
        "checkpoint_signature": "test",
    }

    with pytest.raises(
        ValueError,
        match="real-fit pipeline",
    ):
        pop_emcee_checkpoint_request(
            config
        )


def test_resume_revalidates_stored_log_probability(
    tmp_path,
):
    checkpoint_file = (
        tmp_path
        / "resume_validation.h5"
    )
    execution = resolve_sampler_execution(
        {
            "processes": 1,
            "progress": False,
        }
    )

    EmceeFitter(
        problem=_OffsetProblem(
            0.0
        )
    ).run(
        init_kwargs={
            "nwalkers": 12,
            "seed": 42,
            "checkpoint_path": checkpoint_file,
            "checkpoint_mode": "fresh",
            "checkpoint_signature": "same-declared-problem",
        },
        run_kwargs={
            "burn_in": 2,
            "max_steps": 4,
            "check_interval": 100,
            "min_autocorr_times": 50.0,
            "tau_tolerance": 0.05,
        },
        execution=execution,
    )

    before = checkpoint_file.read_bytes()

    with pytest.raises(
        ValueError,
        match="Stored emcee log probabilities",
    ):
        EmceeFitter(
            problem=_OffsetProblem(
                1.0
            )
        ).run(
            init_kwargs={
                "nwalkers": 12,
                "seed": 42,
                "checkpoint_path": checkpoint_file,
                "checkpoint_mode": "resume",
                "checkpoint_signature": "same-declared-problem",
            },
            run_kwargs={
                "burn_in": 2,
                "max_steps": 6,
                "check_interval": 100,
                "min_autocorr_times": 50.0,
                "tau_tolerance": 0.05,
            },
            execution=execution,
        )

    assert checkpoint_file.read_bytes() == before
