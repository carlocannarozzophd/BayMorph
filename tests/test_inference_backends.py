"""Tests for BayMorph built-in inference backend adapters."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting.backend import (
    BackendRegistry,
    InferenceBackend,
)
from baymorph.fitting.backends import (
    DynestyBackend,
    PocoMCBackend,
    register_builtin_backends,
)
from baymorph.fitting.problem import (
    InferenceProblem,
)


class ScalarProblem(InferenceProblem):
    """Minimal backend-neutral scalar inference problem."""

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


def test_dynesty_backend_owns_dynesty_config():
    backend = DynestyBackend(
        problem=ScalarProblem(),
        config={
            "name": "dynesty",
            "nlive": 300,
            "sample": "auto",
            "n_processes": 4,
        },
    )

    assert backend.name == "dynesty"
    assert backend.config["nlive"] == 300
    assert backend.config["n_processes"] == 4


def test_pocomc_backend_owns_pocomc_config():
    backend = PocoMCBackend(
        problem=ScalarProblem(),
        config={
            "name": "pocomc",
            "n_effective": 512,
            "n_active": 256,
            "n_processes": 4,
        },
    )

    assert backend.name == "pocomc"
    assert backend.config["n_effective"] == 512
    assert backend.config["n_active"] == 256


def test_dynesty_backend_rejects_pocomc_option():
    with pytest.raises(
        ValueError,
        match="Unknown Dynesty configuration",
    ):
        DynestyBackend(
            problem=ScalarProblem(),
            config={
                "n_effective": 512,
            },
        )


def test_pocomc_backend_rejects_dynesty_option():
    with pytest.raises(
        ValueError,
        match="Unknown pocoMC configuration",
    ):
        PocoMCBackend(
            problem=ScalarProblem(),
            config={
                "nlive": 500,
            },
        )


def test_builtin_backends_register_cleanly():
    registry = BackendRegistry()

    register_builtin_backends(
        registry
    )

    assert registry.names == (
        "dynesty",
        "pocomc",
    )


def test_registry_constructs_dynesty_backend():
    registry = BackendRegistry()

    register_builtin_backends(
        registry
    )

    backend = registry.create(
        "dynesty",
        problem=ScalarProblem(),
        config={
            "nlive": 250,
        },
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


def test_registry_constructs_pocomc_backend():
    registry = BackendRegistry()

    register_builtin_backends(
        registry
    )

    backend = registry.create(
        "pocomc",
        problem=ScalarProblem(),
        config={
            "n_effective": 256,
            "n_active": 128,
        },
    )

    assert isinstance(
        backend,
        PocoMCBackend,
    )
    assert isinstance(
        backend,
        InferenceBackend,
    )


def test_builtin_registration_rejects_duplicate_registration():
    registry = BackendRegistry()

    register_builtin_backends(
        registry
    )

    with pytest.raises(
        ValueError,
        match="already registered",
    ):
        register_builtin_backends(
            registry
        )


def test_builtin_registration_can_replace_existing_entries():
    registry = BackendRegistry()

    register_builtin_backends(
        registry
    )

    register_builtin_backends(
        registry,
        replace=True,
    )

    assert registry.names == (
        "dynesty",
        "pocomc",
    )
