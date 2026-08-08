"""Tests for the optional pocoMC fitting backend."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from baymorph.fitting import (
    Parameter,
    ParameterSet,
    PocoMCFitter,
)
from baymorph.fitting.backends import PocoMCBackend
from baymorph.fitting.image_problem import ImageInferenceProblem
from baymorph.fitting.sampler_factory import make_sampler
from baymorph.likelihood import GaussianImageLikelihood


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
    ):
        """Render a constant image."""
        return np.full(
            shape,
            self.x,
            dtype=float,
        )


def _make_problem() -> ImageInferenceProblem:
    """Construct a minimal one-parameter image inference problem."""
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
        ),
        noise=np.ones(
            (2, 2),
        ),
    )

    return ImageInferenceProblem(
        model_factory=lambda x: ConstantImageModel(
            x
        ),
        parameters=parameters,
        likelihood=likelihood,
    )


def test_factory_constructs_pocomc_backend():
    problem = _make_problem()

    sampler = make_sampler(
        {
            "name": "pocomc",
        },
        problem=problem,
    )

    assert isinstance(
        sampler,
        PocoMCBackend,
    )
    assert sampler.problem is problem


def test_pocomc_result_preserves_posterior_weights(
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
                "calls": np.array(
                    [100, 250]
                ),
                "efficiency": np.array(
                    [0.4, 0.8]
                ),
                "ess": np.array(
                    [100.0, 200.0]
                ),
                "accept": np.array(
                    [0.5, 0.7]
                ),
                "beta": np.array(
                    [0.5, 1.0]
                ),
            }

        def run(
            self,
            **kwargs,
        ):
            self.run_kwargs = kwargs

        def posterior(
            self,
        ):
            return (
                np.array(
                    [
                        [-0.2],
                        [0.1],
                        [0.3],
                    ]
                ),
                np.array(
                    [
                        0.2,
                        0.3,
                        0.5,
                    ]
                ),
                np.array(
                    [
                        -1.0,
                        -0.5,
                        -0.8,
                    ]
                ),
                np.array(
                    [
                        -0.7,
                        -0.7,
                        -0.7,
                    ]
                ),
            )

        def evidence(
            self,
        ):
            return (
                -2.5,
                0.15,
            )

    fake_module = SimpleNamespace(
        Prior=FakePrior,
        Sampler=FakeSampler,
    )

    monkeypatch.setitem(
        sys.modules,
        "pocomc",
        fake_module,
    )

    fitter = PocoMCFitter(
        problem=_make_problem(),
        config={
            "name": "pocomc",
            "n_effective": 64,
            "n_active": 32,
            "n_total": 128,
            "n_evidence": 64,
        },
    )

    result = fitter.run(
        print_progress=False,
        seed=42,
    )
    assert result.logz == pytest.approx(
        -2.5
    )

    assert result.logz_error == pytest.approx(
        0.15
    )

    assert result.weights == pytest.approx(
        np.array(
            [
                0.2,
                0.3,
                0.5,
            ]
        )
    )

    assert result.parameter_names == [
        "x",
    ]

    assert result.diagnostics[
        "calls"
    ] == pytest.approx(
        250.0
    )

    assert result.diagnostics[
        "ess"
    ] == pytest.approx(
        200.0
    )


def test_pocomc_rejects_vectorized_likelihood_for_now():
    fitter = PocoMCFitter(
        problem=_make_problem(),
        config={
            "name": "pocomc",
            "vectorize": True,
        },
    )

    with pytest.raises(
        NotImplementedError,
        match="Vectorized",
    ):
        fitter.run(
            print_progress=False,
        )
