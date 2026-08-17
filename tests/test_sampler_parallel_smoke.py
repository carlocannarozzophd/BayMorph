"""Real multiprocessing smoke tests for supported inference backends."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting import (
    Parameter,
    ParameterSet,
)
from baymorph.fitting.backends import (
    DynestyBackend,
    NautilusBackend,
    PocoMCBackend,
)
from baymorph.fitting.problem import InferenceProblem
from baymorph.results import InferenceResult


class ParallelSmokeProblem(InferenceProblem):
    """Minimal two-dimensional problem usable by every public backend."""

    def __init__(self) -> None:
        self.parameters = ParameterSet(
            [
                Parameter(
                    "x",
                    -1.0,
                    1.0,
                ),
                Parameter(
                    "y",
                    -1.0,
                    1.0,
                ),
            ]
        )

    @property
    def ndim(self) -> int:
        return 2

    @property
    def parameter_names(self) -> list[str]:
        return [
            "x",
            "y",
        ]

    def prior_transform(
        self,
        cube: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(
            cube,
            dtype=float,
        )

        return (
            -1.0
            + 2.0 * values
        )

    def log_likelihood(
        self,
        theta: np.ndarray,
    ) -> float:
        values = np.asarray(
            theta,
            dtype=float,
        )

        return float(
            -0.5
            * np.sum(
                values**2
            )
        )


def _assert_parallel_result(
    result: InferenceResult,
    *,
    backend_name: str,
) -> None:
    """Check the backend-neutral contract after a parallel smoke run."""
    assert isinstance(
        result,
        InferenceResult,
    )

    assert result.evidence is not None
    assert np.isfinite(
        result.logz
    )

    assert result.samples.ndim == 2
    assert result.samples.shape[1] == 2
    assert result.parameter_names == [
        "x",
        "y",
    ]

    assert result.provenance[
        "backend"
    ] == backend_name

    execution = result.provenance[
        "sampler"
    ][
        "execution"
    ]

    assert execution[
        "requested_processes"
    ] == 2
    assert execution[
        "processes"
    ] == 2
    assert execution[
        "progress"
    ] is False


def test_dynesty_runs_with_two_processes() -> None:
    """Dynesty must execute through the BayMorph multiprocessing contract."""
    backend = DynestyBackend(
        problem=ParallelSmokeProblem(),
        config={
            "name": "dynesty",
            "mode": "static",
            "execution": {
                "processes": 2,
                "progress": False,
            },
            "init": {
                "nlive": 30,
                "rstate": np.random.default_rng(
                    123
                ),
            },
            "run": {
                "dlogz": 5.0,
            },
        },
    )

    result = backend.fit()

    _assert_parallel_result(
        result,
        backend_name="dynesty",
    )


def test_pocomc_runs_with_two_processes(
    tmp_path,
) -> None:
    """pocoMC must execute through the BayMorph multiprocessing contract."""
    pytest.importorskip(
        "pocomc"
    )

    backend = PocoMCBackend(
        problem=ParallelSmokeProblem(),
        config={
            "name": "pocomc",
            "execution": {
                "processes": 2,
                "progress": False,
            },
            "init": {
                "n_effective": 64,
                "n_active": 32,
                "precondition": False,
                "random_state": 123,
                "output_dir": str(
                    tmp_path
                    / "pocomc"
                ),
            },
            "run": {
                "n_total": 64,
                "n_evidence": 64,
            },
        },
    )

    result = backend.fit()

    _assert_parallel_result(
        result,
        backend_name="pocomc",
    )


def test_nautilus_runs_with_two_processes() -> None:
    """Nautilus must execute through the BayMorph multiprocessing contract."""
    pytest.importorskip(
        "nautilus"
    )

    backend = NautilusBackend(
        problem=ParallelSmokeProblem(),
        config={
            "name": "nautilus",
            "execution": {
                "processes": 2,
                "progress": False,
            },
            "init": {
                "n_live": 100,
                "n_batch": 20,
                "seed": 123,
            },
            "run": {
                "f_live": 0.1,
                "n_eff": 100,
            },
        },
    )

    result = backend.fit()

    _assert_parallel_result(
        result,
        backend_name="nautilus",
    )
