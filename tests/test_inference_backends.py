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
    NautilusBackend,
    PocoMCBackend,
    register_builtin_backends,
)
from baymorph.fitting.problem import InferenceProblem


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


def test_dynesty_backend_preserves_static_native_config():
    backend = DynestyBackend(
        problem=ScalarProblem(),
        config={
            "name": "dynesty",
            "mode": "static",
            "init": {
                "nlive": 300,
                "sample": "rslice",
                "future_init_option": 17,
            },
            "run": {
                "dlogz": 0.05,
                "future_run_option": "value",
            },
        },
    )

    assert backend.name == "dynesty"
    assert backend.mode == "static"

    assert backend.config == {
        "name": "dynesty",
        "mode": "static",
        "init": {
            "nlive": 300,
            "sample": "rslice",
            "future_init_option": 17,
        },
        "run": {
            "dlogz": 0.05,
            "future_run_option": "value",
        },
    }


def test_dynesty_backend_preserves_dynamic_native_config():
    backend = DynestyBackend(
        problem=ScalarProblem(),
        config={
            "name": "dynesty",
            "mode": "dynamic",
            "init": {
                "sample": "rwalk",
            },
            "run": {
                "nlive_init": 500,
                "nlive_batch": 250,
                "dlogz_init": 0.01,
            },
        },
    )

    assert backend.mode == "dynamic"

    assert backend.config["run"] == {
        "nlive_init": 500,
        "nlive_batch": 250,
        "dlogz_init": 0.01,
    }


def test_dynesty_backend_defaults_to_static():
    backend = DynestyBackend(
        problem=ScalarProblem(),
    )

    assert backend.config == {
        "name": "dynesty",
        "mode": "static",
        "init": {},
        "run": {},
    }


def test_dynesty_backend_rejects_flat_native_option():
    with pytest.raises(
        ValueError,
        match="placed under 'init' or 'run'",
    ):
        DynestyBackend(
            problem=ScalarProblem(),
            config={
                "nlive": 500,
            },
        )


def test_dynesty_backend_rejects_invalid_mode():
    with pytest.raises(
        ValueError,
        match="static.*dynamic",
    ):
        DynestyBackend(
            problem=ScalarProblem(),
            config={
                "mode": "hyperdynamic",
            },
        )


def test_pocomc_backend_preserves_native_config():
    backend = PocoMCBackend(
        problem=ScalarProblem(),
        config={
            "name": "pocomc",
            "init": {
                "n_effective": 512,
                "sample": "tpcn",
                "random_state": 42,
                "future_init_option": 17,
            },
            "run": {
                "n_total": 4096,
                "n_evidence": 4096,
                "future_run_option": "value",
            },
        },
    )

    assert backend.name == "pocomc"

    assert backend.config == {
        "name": "pocomc",
        "init": {
            "n_effective": 512,
            "sample": "tpcn",
            "random_state": 42,
            "future_init_option": 17,
        },
        "run": {
            "n_total": 4096,
            "n_evidence": 4096,
            "future_run_option": "value",
        },
    }


def test_pocomc_backend_rejects_flat_native_option():
    with pytest.raises(
        ValueError,
        match="placed under 'init' or 'run'",
    ):
        PocoMCBackend(
            problem=ScalarProblem(),
            config={
                "n_effective": 512,
            },
        )


def test_pocomc_backend_rejects_mode():
    with pytest.raises(
        ValueError,
        match="Unexpected top-level option",
    ):
        PocoMCBackend(
            problem=ScalarProblem(),
            config={
                "mode": "dynamic",
            },
        )


def test_nautilus_backend_preserves_native_config():
    backend = NautilusBackend(
        problem=ScalarProblem(),
        config={
            "name": "nautilus",
            "init": {
                "n_live": 500,
                "seed": 42,
            },
            "run": {
                "n_eff": 1000,
            },
        },
    )

    assert backend.name == "nautilus"

    assert backend.config == {
        "name": "nautilus",
        "init": {
            "n_live": 500,
            "seed": 42,
        },
        "run": {
            "n_eff": 1000,
        },
    }


def test_nautilus_backend_rejects_flat_native_option():
    with pytest.raises(
        ValueError,
        match="placed under 'init' or 'run'",
    ):
        NautilusBackend(
            problem=ScalarProblem(),
            config={
                "n_live": 500,
            },
        )


def test_builtin_backends_register_cleanly():
    registry = BackendRegistry()

    register_builtin_backends(
        registry
    )

    assert registry.names == (
        "dynesty",
        "nautilus",
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
            "mode": "dynamic",
            "init": {
                "sample": "rwalk",
            },
            "run": {
                "nlive_init": 250,
            },
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
    assert backend.mode == "dynamic"


def test_registry_constructs_nautilus_backend():
    registry = BackendRegistry()

    register_builtin_backends(
        registry
    )

    backend = registry.create(
        "nautilus",
        problem=ScalarProblem(),
        config={
            "init": {
                "n_live": 500,
                "seed": 42,
            },
            "run": {
                "n_eff": 1000,
            },
        },
    )

    assert isinstance(
        backend,
        NautilusBackend,
    )
    assert isinstance(
        backend,
        InferenceBackend,
    )


def test_registry_constructs_pocomc_backend():
    registry = BackendRegistry()

    register_builtin_backends(
        registry
    )

    backend = registry.create(
        "pocomc",
        problem=ScalarProblem(),
        config={
            "init": {
                "n_effective": 256,
                "n_active": 128,
            },
            "run": {
                "n_total": 1024,
            },
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
