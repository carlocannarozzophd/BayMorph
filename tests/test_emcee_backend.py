"""Tests for the posterior-only emcee inference backend."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting.emcee import EmceeFitter
from baymorph.fitting.execution import resolve_sampler_execution
from baymorph.fitting.problem import InferenceProblem


class ScalarProblem(InferenceProblem):
    """One-dimensional problem with a bounded uniform prior."""

    @property
    def ndim(self) -> int:
        return 1

    @property
    def parameter_names(self) -> list[str]:
        return ["x"]

    def prior_transform(
        self,
        cube: np.ndarray,
    ) -> np.ndarray:
        return 2.0 * np.asarray(
            cube,
            dtype=float,
        ) - 1.0

    def log_likelihood(
        self,
        theta: np.ndarray,
    ) -> float:
        value = float(
            np.asarray(
                theta,
                dtype=float,
            )[0]
        )

        return -0.5 * (
            (value - 0.2)
            / 0.25
        ) ** 2


def test_emcee_log_probability_uses_unit_cube_prior():
    fitter = EmceeFitter(
        problem=ScalarProblem()
    )

    assert fitter.log_probability(
        np.asarray([0.6])
    ) == pytest.approx(0.0)

    assert fitter.log_probability(
        np.asarray([-0.01])
    ) == -np.inf

    assert fitter.log_probability(
        np.asarray([1.01])
    ) == -np.inf

    with pytest.raises(
        ValueError,
        match="shape",
    ):
        fitter.log_probability(
            np.asarray([0.5, 0.5])
        )


def test_emcee_returns_posterior_without_evidence():
    fitter = EmceeFitter(
        problem=ScalarProblem()
    )

    result = fitter.run(
        init_kwargs={
            "nwalkers": 12,
            "seed": 42,
        },
        run_kwargs={
            "burn_in": 4,
            "max_steps": 8,
            "check_interval": 100,
            "min_autocorr_times": 50.0,
            "tau_tolerance": 0.05,
        },
        execution=resolve_sampler_execution(
            {
                "processes": 1,
                "progress": False,
            }
        ),
    )

    assert result.evidence is None
    assert result.logz is None
    assert result.logz_error is None

    assert result.samples.shape == (
        96,
        1,
    )
    assert result.log_likelihood.shape == (96,)
    assert result.posterior.log_prior is None

    np.testing.assert_allclose(
        result.weights,
        np.full(
            96,
            1.0 / 96.0,
        ),
    )

    assert np.all(result.samples >= -1.0)
    assert np.all(result.samples <= 1.0)

    assert result.diagnostics["backend"] == "emcee"
    assert result.diagnostics["walkers"] == 12
    assert result.diagnostics["burn_in"] == 4
    assert result.diagnostics["steps"] == 8
    assert result.diagnostics["converged"] is False
    assert result.diagnostics["mcmc_effective_sample_size"] is None
    assert result.posterior.effective_sample_size == pytest.approx(
        96.0
    )
    assert result.effective_sample_size is None

    result.diagnostics["mcmc_effective_sample_size"] = 12.5
    assert result.effective_sample_size == pytest.approx(
        12.5
    )

    assert result.provenance["backend"] == "emcee"
    assert (
        result.provenance["inference_kind"]
        == "posterior_only"
    )

def _run_small_emcee(
    *,
    seed: int,
    processes: int,
):
    """Run one deterministic short posterior-only chain."""
    return EmceeFitter(
        problem=ScalarProblem()
    ).run(
        init_kwargs={
            "nwalkers": 12,
            "seed": seed,
        },
        run_kwargs={
            "burn_in": 2,
            "max_steps": 6,
            "check_interval": 100,
            "min_autocorr_times": 50.0,
            "tau_tolerance": 0.05,
        },
        execution=resolve_sampler_execution(
            {
                "processes": processes,
                "progress": False,
            }
        ),
    )


def test_emcee_is_reproducible_for_fixed_seed():
    first = _run_small_emcee(
        seed=73,
        processes=1,
    )
    second = _run_small_emcee(
        seed=73,
        processes=1,
    )

    np.testing.assert_allclose(
        first.samples,
        second.samples,
    )
    np.testing.assert_allclose(
        first.log_likelihood,
        second.log_likelihood,
    )


def test_emcee_parallel_execution_returns_canonical_result():
    result = _run_small_emcee(
        seed=19,
        processes=2,
    )

    assert result.samples.shape == (
        72,
        1,
    )
    assert result.evidence is None
    assert result.diagnostics["backend"] == "emcee"
    assert (
        result.provenance["sampler"]["execution"]["processes"]
        == 2
    )


def _run_checkpointed_emcee(
    path,
    *,
    mode: str,
    max_steps: int,
    signature: str = "scalar-problem-v1",
):
    """Run one short deterministic checkpointed chain."""
    return EmceeFitter(
        problem=ScalarProblem()
    ).run(
        init_kwargs={
            "nwalkers": 12,
            "seed": 42,
            "checkpoint_path": str(
                path
            ),
            "checkpoint_mode": mode,
            "checkpoint_signature": signature,
        },
        run_kwargs={
            "burn_in": 2,
            "max_steps": max_steps,
            "check_interval": 100,
            "min_autocorr_times": 50.0,
            "tau_tolerance": 0.05,
        },
        execution=resolve_sampler_execution(
            {
                "processes": 1,
                "progress": False,
            }
        ),
    )


def test_emcee_checkpoint_stores_production_chain(
    tmp_path,
):
    """HDF checkpoints must store production samples only."""
    import emcee as emcee_library

    path = tmp_path / "production.h5"

    result = _run_checkpointed_emcee(
        path,
        mode="fresh",
        max_steps=6,
    )

    backend = emcee_library.backends.HDFBackend(
        str(
            path
        ),
        read_only=True,
    )

    assert backend.iteration == 6
    assert result.samples.shape == (
        72,
        1,
    )
    assert result.evidence is None
    assert result.logz is None

    cube_samples = backend.get_chain(
        flat=True
    )
    expected_samples = (
        -1.0
        + 2.0 * cube_samples
    )

    np.testing.assert_array_equal(
        result.samples,
        expected_samples,
    )
    np.testing.assert_array_equal(
        result.log_likelihood,
        backend.get_log_prob(
            flat=True
        ),
    )

    assert result.diagnostics[
        "checkpoint_resumed"
    ] is False
    assert result.diagnostics[
        "preexisting_steps"
    ] == 0
    assert result.diagnostics[
        "added_steps"
    ] == 6


def test_emcee_checkpoint_resume_matches_continuous_run(
    tmp_path,
):
    """A split chain must exactly match an uninterrupted chain."""
    uninterrupted = _run_checkpointed_emcee(
        tmp_path / "continuous.h5",
        mode="fresh",
        max_steps=10,
    )

    _run_checkpointed_emcee(
        tmp_path / "split.h5",
        mode="fresh",
        max_steps=4,
    )
    resumed = _run_checkpointed_emcee(
        tmp_path / "split.h5",
        mode="resume",
        max_steps=10,
    )

    np.testing.assert_array_equal(
        resumed.samples,
        uninterrupted.samples,
    )
    np.testing.assert_array_equal(
        resumed.log_likelihood,
        uninterrupted.log_likelihood,
    )
    np.testing.assert_array_equal(
        resumed.weights,
        uninterrupted.weights,
    )

    assert resumed.samples.shape == (
        120,
        1,
    )
    assert resumed.diagnostics[
        "checkpoint_resumed"
    ] is True
    assert resumed.diagnostics[
        "preexisting_steps"
    ] == 4
    assert resumed.diagnostics[
        "added_steps"
    ] == 6
    assert resumed.diagnostics[
        "steps"
    ] == 10


def test_emcee_checkpoint_fresh_refuses_existing_file(
    tmp_path,
):
    """Fresh mode must never overwrite an existing path."""
    path = tmp_path / "occupied.h5"
    path.write_bytes(
        b"do not overwrite"
    )

    with pytest.raises(
        FileExistsError,
        match="Refusing to overwrite",
    ):
        _run_checkpointed_emcee(
            path,
            mode="fresh",
            max_steps=4,
        )

    assert path.read_bytes() == b"do not overwrite"


def test_emcee_checkpoint_resume_requires_existing_file(
    tmp_path,
):
    """Resume mode must not silently create a checkpoint."""
    path = tmp_path / "missing.h5"

    with pytest.raises(
        FileNotFoundError,
        match="does not exist",
    ):
        _run_checkpointed_emcee(
            path,
            mode="resume",
            max_steps=4,
        )

    assert not path.exists()


def test_emcee_checkpoint_rejects_signature_mismatch(
    tmp_path,
):
    """A checkpoint from another problem must remain untouched."""
    path = tmp_path / "signature.h5"

    _run_checkpointed_emcee(
        path,
        mode="fresh",
        max_steps=4,
        signature="problem-a",
    )
    original = path.read_bytes()

    with pytest.raises(
        ValueError,
        match="checkpoint_signature",
    ):
        _run_checkpointed_emcee(
            path,
            mode="resume",
            max_steps=8,
            signature="problem-b",
        )

    assert path.read_bytes() == original
