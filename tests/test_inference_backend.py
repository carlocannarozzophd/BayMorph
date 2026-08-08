"""Tests for the backend-neutral inference-engine contract."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting.backend import (
    BackendRegistry,
    InferenceBackend,
)
from baymorph.fitting.problem import (
    InferenceProblem,
)
from baymorph.results import PosteriorResult


class ScalarProblem(InferenceProblem):
    """Minimal one-dimensional inference problem."""

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
        return np.asarray(
            cube,
            dtype=float,
        )

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

        return -0.5 * value**2


class DummyBackend(InferenceBackend):
    """Minimal backend implementation used for contract tests."""

    @property
    def name(self) -> str:
        return "dummy"

    def fit(
        self,
        *,
        progress_label: str | None = None,
    ) -> PosteriorResult:
        del progress_label

        return PosteriorResult(
            samples=np.asarray(
                [[0.0]],
                dtype=float,
            ),
            logwt=np.asarray(
                [0.0],
                dtype=float,
            ),
            logz=np.asarray(
                [0.0],
                dtype=float,
            ),
            parameter_names=["x"],
            logl=np.asarray(
                [0.0],
                dtype=float,
            ),
            logzerr=np.asarray(
                [0.0],
                dtype=float,
            ),
        )


def make_dummy_backend(
    problem,
    config,
):
    """Construct the dummy backend."""
    return DummyBackend(
        problem=problem,
        config=config,
    )


def test_backend_requires_inference_problem():
    with pytest.raises(
        TypeError,
        match="InferenceProblem",
    ):
        DummyBackend(
            problem=object(),
        )


def test_registry_constructs_registered_backend():
    registry = BackendRegistry()
    problem = ScalarProblem()

    registry.register(
        "dummy",
        make_dummy_backend,
    )

    backend = registry.create(
        "dummy",
        problem=problem,
        config={
            "value": 3,
        },
    )

    assert isinstance(
        backend,
        DummyBackend,
    )
    assert backend.problem is problem
    assert backend.config == {
        "value": 3,
    }


def test_registry_normalizes_backend_names():
    registry = BackendRegistry()

    registry.register(
        "Dummy",
        make_dummy_backend,
    )

    backend = registry.create(
        " DUMMY ",
        problem=ScalarProblem(),
    )

    assert backend.name == "dummy"


def test_registry_rejects_duplicate_registration():
    registry = BackendRegistry()

    registry.register(
        "dummy",
        make_dummy_backend,
    )

    with pytest.raises(
        ValueError,
        match="already registered",
    ):
        registry.register(
            "dummy",
            make_dummy_backend,
        )


def test_registry_rejects_unknown_backend():
    registry = BackendRegistry()

    with pytest.raises(
        ValueError,
        match="Unknown inference backend",
    ):
        registry.create(
            "pincopallo",
            problem=ScalarProblem(),
        )


def test_registry_reports_registered_names():
    registry = BackendRegistry()

    registry.register(
        "zeta",
        make_dummy_backend,
    )
    registry.register(
        "alpha",
        make_dummy_backend,
    )

    assert registry.names == (
        "alpha",
        "zeta",
    )


def test_backend_fit_contract_returns_result():
    backend = DummyBackend(
        problem=ScalarProblem(),
    )

    result = backend.fit()

    assert isinstance(
        result,
        PosteriorResult,
    )
    assert result.parameter_names == [
        "x",
    ]
