"""Tests for pocoMC canonical inference results."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from baymorph.fitting import (
    Parameter,
    ParameterSet,
)
from baymorph.fitting.image_problem import ImageInferenceProblem
from baymorph.fitting.pocomc import PocoMCFitter
from baymorph.likelihood import GaussianImageLikelihood
from baymorph.results import (
    InferenceResult,
    PosteriorSamples,
)


class ConstantImageModel:
    """Minimal constant-valued image model."""

    def __init__(
        self,
        x: float,
    ) -> None:
        self.x = float(x)

    def render(
        self,
        shape,
    ) -> np.ndarray:
        """Render a constant image."""
        return np.full(
            shape,
            self.x,
            dtype=float,
        )


def _make_problem() -> ImageInferenceProblem:
    """Construct a minimal image inference problem."""
    parameters = ParameterSet(
        [
            Parameter(
                "x",
                -1.0,
                1.0,
            )
        ]
    )

    likelihood = GaussianImageLikelihood(
        data=np.zeros(
            (2, 2),
            dtype=float,
        ),
        noise=np.ones(
            (2, 2),
            dtype=float,
        ),
    )

    return ImageInferenceProblem(
        model_factory=lambda x: ConstantImageModel(
            x
        ),
        parameters=parameters,
        likelihood=likelihood,
    )


def test_pocomc_returns_canonical_native_result(
    monkeypatch,
):
    class FakePrior:
        def __init__(
            self,
            distributions,
        ):
            self.distributions = distributions

    class FakeSampler:
        def __init__(
            self,
            **kwargs,
        ):
            self.kwargs = kwargs
            self.results = {
                "calls": np.asarray(
                    [100, 250]
                ),
                "efficiency": np.asarray(
                    [0.4, 0.8]
                ),
                "ess": np.asarray(
                    [100.0, 200.0]
                ),
                "accept": np.asarray(
                    [0.5, 0.7]
                ),
                "beta": np.asarray(
                    [0.5, 1.0]
                ),
            }

        def run(
            self,
            **kwargs,
        ):
            self.run_kwargs = kwargs

        def posterior(self):
            return (
                np.asarray(
                    [
                        [-0.2],
                        [0.1],
                        [0.3],
                    ]
                ),
                np.asarray(
                    [
                        0.2,
                        0.3,
                        0.5,
                    ]
                ),
                np.asarray(
                    [
                        -1.0,
                        -0.5,
                        -0.8,
                    ]
                ),
                np.asarray(
                    [
                        -0.7,
                        -0.6,
                        -0.5,
                    ]
                ),
            )

        def evidence(self):
            return (
                -2.5,
                0.15,
            )

    monkeypatch.setitem(
        sys.modules,
        "pocomc",
        SimpleNamespace(
            Prior=FakePrior,
            Sampler=FakeSampler,
        ),
    )

    fitter = PocoMCFitter(
        problem=_make_problem(),
    )

    result = fitter.run(
        init_kwargs={
            "n_effective": 64,
            "n_active": 32,
            "random_state": 42,
        },
        run_kwargs={
            "n_total": 128,
            "n_evidence": 64,
            "progress": False,
        },
    )

    assert isinstance(
        result,
        InferenceResult,
    )

    assert isinstance(
        result.posterior,
        PosteriorSamples,
    )

    assert result.weights == pytest.approx(
        np.asarray(
            [
                0.2,
                0.3,
                0.5,
            ]
        )
    )

    assert result.log_likelihood == pytest.approx(
        np.asarray(
            [
                -1.0,
                -0.5,
                -0.8,
            ]
        )
    )

    assert result.log_prior == pytest.approx(
        np.asarray(
            [
                -0.7,
                -0.6,
                -0.5,
            ]
        )
    )

    assert result.logz == pytest.approx(
        -2.5
    )

    assert result.logz_error == pytest.approx(
        0.15
    )

    assert result.evidence is not None
    assert result.evidence.method == "pocomc"

    assert (
        result.evidence.uncertainty_kind
        == "importance_sampling_internal"
    )

    assert result.diagnostics[
        "calls"
    ] == pytest.approx(
        250.0
    )

    assert result.provenance[
        "backend"
    ] == "pocomc"

    assert result.provenance[
        "sampler"
    ]["init"]["random_state"] == 42


def test_pocomc_result_has_no_nested_sampling_public_state(
    monkeypatch,
):
    class FakePrior:
        def __init__(
            self,
            distributions,
        ):
            self.distributions = distributions

    class FakeSampler:
        def __init__(
            self,
            **kwargs,
        ):
            self.results = {}

        def run(
            self,
            **kwargs,
        ):
            pass

        def posterior(self):
            return (
                np.asarray(
                    [[0.0]]
                ),
                np.asarray(
                    [1.0]
                ),
                np.asarray(
                    [-1.0]
                ),
                np.asarray(
                    [-0.5]
                ),
            )

        def evidence(self):
            return (
                -2.0,
                0.1,
            )

    monkeypatch.setitem(
        sys.modules,
        "pocomc",
        SimpleNamespace(
            Prior=FakePrior,
            Sampler=FakeSampler,
        ),
    )

    result = PocoMCFitter(
        problem=_make_problem(),
    ).run(
        init_kwargs={
            "n_effective": 64,
            "n_active": 32,
        },
        run_kwargs={
            "progress": False,
        },
    )

    assert not hasattr(
        result,
        "logwt",
    )

    assert not hasattr(
        result,
        "logzerr",
    )


def test_pocomc_accepts_missing_evidence_uncertainty(
    monkeypatch,
):
    """pocoMC may return a finite evidence estimate without an uncertainty."""

    class FakePrior:
        def __init__(
            self,
            distributions,
        ):
            self.distributions = distributions

    class FakeSampler:
        def __init__(
            self,
            **kwargs,
        ):
            self.kwargs = kwargs
            self.results = {}

        def run(
            self,
            **kwargs,
        ):
            self.run_kwargs = kwargs

        def posterior(self):
            return (
                np.asarray(
                    [
                        [-0.2],
                        [0.2],
                    ]
                ),
                np.asarray(
                    [
                        0.5,
                        0.5,
                    ]
                ),
                np.asarray(
                    [
                        -0.5,
                        -0.5,
                    ]
                ),
                np.asarray(
                    [
                        -0.7,
                        -0.7,
                    ]
                ),
            )

        def evidence(self):
            return (
                -2.0,
                None,
            )

    monkeypatch.setitem(
        sys.modules,
        "pocomc",
        SimpleNamespace(
            Prior=FakePrior,
            Sampler=FakeSampler,
        ),
    )

    result = PocoMCFitter(
        problem=_make_problem(),
    ).run()

    assert result.logz == pytest.approx(
        -2.0
    )
    assert result.logz_error is None
    assert result.evidence is not None
    assert result.evidence.uncertainty is None
    assert result.evidence.uncertainty_kind is None
