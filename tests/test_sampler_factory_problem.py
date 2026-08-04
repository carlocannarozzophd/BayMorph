"""Tests for backend-neutral sampler construction."""

import numpy as np
import pytest

from baymorph.fitting import DynestyFitter
from baymorph.fitting.problem import InferenceProblem
from baymorph.fitting.sampler_factory import make_sampler


class ScalarProblem(InferenceProblem):
    """Minimal inference problem for sampler-factory tests."""

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


def test_make_sampler_constructs_dynesty_from_problem():
    problem = ScalarProblem()

    sampler = make_sampler(
        {
            "name": "dynesty",
        },
        problem=problem,
    )

    assert isinstance(
        sampler,
        DynestyFitter,
    )
    assert sampler.problem is problem
    assert sampler.ndim == 1
    assert sampler.parameter_names == ["x"]


def test_make_sampler_defaults_to_dynesty_for_problem():
    sampler = make_sampler(
        None,
        problem=ScalarProblem(),
    )

    assert isinstance(
        sampler,
        DynestyFitter,
    )


def test_make_sampler_rejects_mixed_construction():
    with pytest.raises(
        ValueError,
        match="cannot be combined",
    ):
        make_sampler(
            "dynesty",
            problem=ScalarProblem(),
            model_factory=lambda: None,
        )


def test_make_sampler_rejects_problem_for_native_backend():
    with pytest.raises(
        ValueError,
        match="archived native backend",
    ):
        make_sampler(
            "native",
            problem=ScalarProblem(),
        )


def test_make_sampler_rejects_invalid_problem():
    with pytest.raises(
        TypeError,
        match="InferenceProblem",
    ):
        make_sampler(
            "dynesty",
            problem=object(),
        )
