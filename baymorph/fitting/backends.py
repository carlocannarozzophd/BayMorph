"""Built-in backend-neutral inference adapters for BayMorph."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from baymorph.fitting.backend import (
    BackendRegistry,
    InferenceBackend,
    backend_registry,
)
from baymorph.fitting.dynesty import DynestyFitter
from baymorph.fitting.pocomc import PocoMCFitter
from baymorph.fitting.problem import InferenceProblem
from baymorph.results import PosteriorResult


_DYNESTY_ALLOWED_CONFIG = frozenset(
    {
        "name",
        "nlive",
        "sample",
        "bound",
        "dlogz",
        "bootstrap",
        "print_progress",
        "n_processes",
        "queue_size",
        "seed",
    }
)

_POCOMC_ALLOWED_CONFIG = frozenset(
    {
        "name",
        "n_effective",
        "n_active",
        "n_total",
        "n_evidence",
        "pytorch_threads",
        "precondition",
        "dynamic",
        "metric",
        "mcmc_sampler",
        "flow",
        "train_frequency",
        "n_prior",
        "n_steps",
        "n_max_steps",
        "train_config",
        "vectorize",
        "print_progress",
        "n_processes",
        "seed",
    }
)


def _validate_config_keys(
    config: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    backend_name: str,
) -> dict[str, Any]:
    """Validate and copy one backend-specific configuration."""
    resolved = dict(config)

    unknown = sorted(
        set(resolved) - allowed
    )

    if unknown:
        names = ", ".join(unknown)

        raise ValueError(
            f"Unknown {backend_name} configuration "
            f"option(s): {names}."
        )

    return resolved


class DynestyBackend(InferenceBackend):
    """Backend-neutral adapter for Dynesty inference."""

    def __init__(
        self,
        *,
        problem: InferenceProblem,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        resolved = _validate_config_keys(
            {} if config is None else config,
            allowed=_DYNESTY_ALLOWED_CONFIG,
            backend_name="Dynesty",
        )

        super().__init__(
            problem=problem,
            config=resolved,
        )

    @property
    def name(self) -> str:
        """Return the stable backend identifier."""
        return "dynesty"

    def fit(
        self,
        *,
        progress_label: str | None = None,
    ) -> PosteriorResult:
        """Run Dynesty using only Dynesty-specific configuration."""
        fitter = DynestyFitter(
            problem=self.problem
        )

        return fitter.run(
            nlive=int(
                self.config.get(
                    "nlive",
                    500,
                )
            ),
            sample=str(
                self.config.get(
                    "sample",
                    "auto",
                )
            ),
            bound=str(
                self.config.get(
                    "bound",
                    "multi",
                )
            ),
            dlogz=float(
                self.config.get(
                    "dlogz",
                    0.05,
                )
            ),
            bootstrap=self.config.get(
                "bootstrap"
            ),
            print_progress=bool(
                self.config.get(
                    "print_progress",
                    True,
                )
            ),
            n_processes=int(
                self.config.get(
                    "n_processes",
                    1,
                )
            ),
            queue_size=self.config.get(
                "queue_size"
            ),
            seed=self.config.get(
                "seed"
            ),
            progress_label=progress_label,
        )


class PocoMCBackend(InferenceBackend):
    """Backend-neutral adapter for pocoMC inference."""

    def __init__(
        self,
        *,
        problem: InferenceProblem,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        resolved = _validate_config_keys(
            {} if config is None else config,
            allowed=_POCOMC_ALLOWED_CONFIG,
            backend_name="pocoMC",
        )

        super().__init__(
            problem=problem,
            config=resolved,
        )

    @property
    def name(self) -> str:
        """Return the stable backend identifier."""
        return "pocomc"

    def fit(
        self,
        *,
        progress_label: str | None = None,
    ) -> PosteriorResult:
        """Run pocoMC using only pocoMC-specific configuration."""
        fitter_config = dict(
            self.config
        )

        fitter_config.pop(
            "name",
            None,
        )

        print_progress = bool(
            fitter_config.pop(
                "print_progress",
                True,
            )
        )

        n_processes = int(
            fitter_config.pop(
                "n_processes",
                1,
            )
        )

        seed = fitter_config.pop(
            "seed",
            None,
        )

        fitter = PocoMCFitter(
            problem=self.problem,
            config=fitter_config,
        )

        return fitter.run(
            print_progress=print_progress,
            n_processes=n_processes,
            seed=seed,
            progress_label=progress_label,
        )


def _make_dynesty_backend(
    problem: InferenceProblem,
    config: Mapping[str, Any],
) -> InferenceBackend:
    """Construct the built-in Dynesty backend."""
    return DynestyBackend(
        problem=problem,
        config=config,
    )


def _make_pocomc_backend(
    problem: InferenceProblem,
    config: Mapping[str, Any],
) -> InferenceBackend:
    """Construct the built-in pocoMC backend."""
    return PocoMCBackend(
        problem=problem,
        config=config,
    )


def register_builtin_backends(
    registry: BackendRegistry = backend_registry,
    *,
    replace: bool = False,
) -> None:
    """Register BayMorph's built-in inference backends."""
    registry.register(
        "dynesty",
        _make_dynesty_backend,
        replace=replace,
    )

    registry.register(
        "pocomc",
        _make_pocomc_backend,
        replace=replace,
    )
