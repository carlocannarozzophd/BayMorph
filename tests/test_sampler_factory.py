"""Tests for backend-neutral inference-backend construction."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting.backend import InferenceBackend
from baymorph.fitting.backends import (
    DynestyBackend,
    PocoMCBackend,
)
from baymorph.fitting.problem import InferenceProblem
from baymorph.fitting.sampler_factory import make_sampler


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


def test_make_sampler_returns_dynesty_backend():
    backend = make_sampler(
        {
            "name": "dynesty",
            "mode": "static",
            "init": {
                "nlive": 250,
            },
            "run": {},
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
    assert backend.config["init"]["nlive"] == 250


def test_make_sampler_returns_pocomc_backend():
    backend = make_sampler(
        {
            "name": "pocomc",
            "init": {
                "n_effective": 256,
                "n_active": 128,
            },
            "run": {},
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


def test_make_sampler_defaults_to_nautilus():
    backend = make_sampler(
        None,
        problem=ScalarProblem(),
    )

    assert isinstance(
        backend,
        InferenceBackend,
    )
    assert backend.name == "nautilus"


def test_make_sampler_mapping_without_name_defaults_to_nautilus():
    backend = make_sampler(
        {},
        problem=ScalarProblem(),
    )

    assert backend.name == "nautilus"


def test_make_sampler_accepts_backend_name_string():
    backend = make_sampler(
        "nautilus",
        problem=ScalarProblem(),
    )

    assert backend.name == "nautilus"


def test_make_sampler_uses_registry_for_unknown_backend():
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


def test_archived_native_backend_is_not_registered():
    with pytest.raises(
        ValueError,
        match="Unknown inference backend",
    ):
        make_sampler(
            {
                "name": "native",
            },
            problem=ScalarProblem(),
        )


def test_make_sampler_requires_inference_problem():
    with pytest.raises(
        TypeError,
        match="problem must be an InferenceProblem",
    ):
        make_sampler(
            "nautilus",
            problem=object(),
        )


def test_backend_specific_options_are_validated():
    with pytest.raises(
        ValueError,
        match="placed under 'init' or 'run'",
    ):
        make_sampler(
            {
                "name": "pocomc",
                "nlive": 500,
            },
            problem=ScalarProblem(),
        )
