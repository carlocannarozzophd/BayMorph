"""Tests for Nautilus canonical inference results."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from baymorph.fitting.nautilus import NautilusFitter
from baymorph.fitting.problem import InferenceProblem
from baymorph.results import (
    InferenceResult,
    PosteriorSamples,
)


class TwoDimensionalProblem(InferenceProblem):
    """Minimal two-dimensional inference problem."""

    @property
    def ndim(self) -> int:
        return 2

    @property
    def parameter_names(self) -> list[str]:
        return ["x", "y"]

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
        values = np.asarray(
            theta,
            dtype=float,
        )

        return float(
            -0.5 * np.sum(
                values**2
            )
        )


def test_nautilus_returns_canonical_result(
    monkeypatch,
):
    class FakeSampler:
        def __init__(
            self,
            prior,
            likelihood,
            n_dim,
            **kwargs,
        ):
            self.prior = prior
            self.likelihood = likelihood
            self.n_dim = n_dim
            self.kwargs = kwargs
            self.n_like = 420
            self.n_eff = 180.0
            self.eta = 0.43
            self.f_live = 0.004
            self.discard_exploration = True
            self.log_z = -3.25

        def run(
            self,
            **kwargs,
        ) -> bool:
            self.run_kwargs = kwargs
            self.discard_exploration = bool(
                kwargs["discard_exploration"]
            )
            return True

        def posterior(
            self,
            **kwargs,
        ):
            return (
                np.asarray(
                    [
                        [-0.2, 0.1],
                        [0.0, -0.1],
                        [0.3, 0.2],
                    ]
                ),
                np.log(
                    np.asarray(
                        [
                            0.2,
                            0.3,
                            0.5,
                        ]
                    )
                ),
                np.asarray(
                    [
                        -1.0,
                        -0.5,
                        -0.8,
                    ]
                ),
            )

    monkeypatch.setitem(
        sys.modules,
        "nautilus",
        SimpleNamespace(
            Sampler=FakeSampler,
        ),
    )

    result = NautilusFitter(
        problem=TwoDimensionalProblem(),
    ).run(
        init_kwargs={
            "n_live": 500,
            "seed": 42,
        },
        run_kwargs={
            "n_eff": 1000,
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
    assert result.parameter_names == [
        "x",
        "y",
    ]
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
    assert result.log_prior is None

    assert result.logz == pytest.approx(
        -3.25
    )
    assert result.logz_error == pytest.approx(
        1.0 / np.sqrt(180.0)
    )

    assert result.evidence is not None
    assert result.evidence.method == "nautilus"
    assert result.evidence.uncertainty_kind == (
        "statistical_ess_approximation"
    )

    assert result.diagnostics[
        "calls"
    ] == 420

    assert result.diagnostics[
        "ess"
    ] == pytest.approx(
        180.0
    )

    assert result.diagnostics[
        "efficiency"
    ] == pytest.approx(
        0.43
    )

    assert result.diagnostics[
        "success"
    ] is True

    assert result.diagnostics[
        "discard_exploration"
    ] is True


def test_nautilus_rejects_sampler_owned_constructor_arguments(
    monkeypatch,
):
    monkeypatch.setitem(
        sys.modules,
        "nautilus",
        SimpleNamespace(
            Sampler=object,
        ),
    )

    fitter = NautilusFitter(
        problem=TwoDimensionalProblem(),
    )

    with pytest.raises(
        ValueError,
        match="owned by BayMorph",
    ):
        fitter.run(
            init_kwargs={
                "n_dim": 2,
            },
        )
