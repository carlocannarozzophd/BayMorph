"""Tests for backend-neutral sampler-factory construction."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting.backend import InferenceBackend
from baymorph.fitting.backends import (
    DynestyBackend,
    PocoMCBackend,
)
from baymorph.fitting.dynesty import DynestyFitter
from baymorph.fitting.parameters import (
    Parameter,
    ParameterSet,
)
from baymorph.fitting.pocomc import PocoMCFitter
from baymorph.fitting.problem import InferenceProblem
from baymorph.fitting.sampler_factory import make_sampler
from baymorph.likelihood import GaussianImageLikelihood


class ScalarProblem(InferenceProblem):
    """Minimal scalar inference problem."""

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


def test_problem_path_returns_dynesty_backend():
    backend = make_sampler(
        {
            "name": "dynesty",
            "nlive": 250,
        },
        problem=ScalarProblem(),
    )

    assert isinstance(
        backend,
        DynestyBackend,
    )
    assert isinstance(
        backend,
        InferenceBackend,
    )
    assert backend.config["nlive"] == 250


def test_problem_path_returns_pocomc_backend():
    backend = make_sampler(
        {
            "name": "pocomc",
            "n_effective": 256,
            "n_active": 128,
        },
        problem=ScalarProblem(),
    )

    assert isinstance(
        backend,
        PocoMCBackend,
    )
    assert isinstance(
        backend,
        InferenceBackend,
    )


def test_problem_path_defaults_to_pocomc():
    backend = make_sampler(
        None,
        problem=ScalarProblem(),
    )

    assert isinstance(
        backend,
        PocoMCBackend,
    )


def test_problem_path_uses_registry_for_unknown_backend():
    with pytest.raises(
        ValueError,
        match="Unknown inference backend",
    ):
        make_sampler(
            {
                "name": "pincopallo",
            },
            problem=ScalarProblem(),
        )


def test_problem_path_rejects_native_backend():
    with pytest.raises(
        ValueError,
        match="archived native backend",
    ):
        make_sampler(
            {
                "name": "native",
            },
            problem=ScalarProblem(),
        )


def test_problem_cannot_be_combined_with_legacy_arguments():
    parameters = ParameterSet(
        [
            Parameter(
                "x",
                0.0,
                1.0,
            )
        ]
    )

    likelihood = GaussianImageLikelihood(
        data=np.zeros((2, 2)),
        noise=1.0,
    )

    with pytest.raises(
        ValueError,
        match="problem cannot be combined",
    ):
        make_sampler(
            "dynesty",
            problem=ScalarProblem(),
            model_factory=object(),
            parameters=parameters,
            likelihood=likelihood,
        )


def test_legacy_dynesty_construction_remains_supported():
    parameters = ParameterSet(
        [
            Parameter(
                "flux",
                0.0,
                1.0,
            )
        ]
    )

    likelihood = GaussianImageLikelihood(
        data=np.zeros((2, 2)),
        noise=1.0,
    )

    backend = make_sampler(
        "dynesty",
        model_factory=lambda **kwargs: kwargs,
        parameters=parameters,
        likelihood=likelihood,
    )

    assert isinstance(
        backend,
        DynestyFitter,
    )


def test_legacy_pocomc_construction_remains_supported():
    parameters = ParameterSet(
        [
            Parameter(
                "flux",
                0.0,
                1.0,
            )
        ]
    )

    likelihood = GaussianImageLikelihood(
        data=np.zeros((2, 2)),
        noise=1.0,
    )

    backend = make_sampler(
        "pocomc",
        model_factory=lambda **kwargs: kwargs,
        parameters=parameters,
        likelihood=likelihood,
    )

    assert isinstance(
        backend,
        PocoMCFitter,
    )


def test_backend_specific_options_are_rejected_on_problem_path():
    with pytest.raises(
        ValueError,
        match="Unknown pocoMC configuration",
    ):
        make_sampler(
            {
                "name": "pocomc",
                "nlive": 500,
            },
            problem=ScalarProblem(),
        )
