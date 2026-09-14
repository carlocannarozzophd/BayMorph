"""Backend-neutral inference-engine contracts and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

from baymorph.fitting.problem import InferenceProblem
from baymorph.results import InferenceResult


BackendFactory: TypeAlias = Callable[
    [InferenceProblem, Mapping[str, Any]],
    "InferenceBackend",
]


class InferenceBackend(ABC):
    """Backend-neutral interface for BayMorph inference engines.

    An inference backend owns all sampler-specific configuration and translates
    BayMorph's :class:`InferenceProblem` contract into the API required by one
    external inference library.

    Pipeline code must interact only with this interface and must never pass
    backend-specific runtime arguments directly.
    """

    def __init__(
        self,
        *,
        problem: InferenceProblem,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(problem, InferenceProblem):
            raise TypeError(
                "problem must be an InferenceProblem instance."
            )

        self.problem = problem
        self.config = (
            {}
            if config is None
            else dict(config)
        )

    @property
    def produces_evidence(self) -> bool:
        """Return whether this backend produces Bayesian evidence."""
        return True

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stable backend identifier."""

    @abstractmethod
    def fit(
        self,
        *,
        progress_label: str | None = None,
    ) -> InferenceResult:
        """Execute inference and return a canonical BayMorph result."""


class BackendRegistry:
    """Registry mapping stable backend names to backend factories."""

    def __init__(self) -> None:
        self._factories: dict[
            str,
            BackendFactory,
        ] = {}

    @staticmethod
    def _normalize_name(
        name: str,
    ) -> str:
        normalized = str(
            name
        ).strip().lower()

        if not normalized:
            raise ValueError(
                "Backend name cannot be empty."
            )

        return normalized

    def register(
        self,
        name: str,
        factory: BackendFactory,
        *,
        replace: bool = False,
    ) -> None:
        """Register one inference backend factory."""
        normalized = self._normalize_name(
            name
        )

        if not callable(factory):
            raise TypeError(
                "Backend factory must be callable."
            )

        if (
            normalized in self._factories
            and not replace
        ):
            raise ValueError(
                f"Backend {normalized!r} is already registered."
            )

        self._factories[
            normalized
        ] = factory

    def create(
        self,
        name: str,
        *,
        problem: InferenceProblem,
        config: Mapping[str, Any] | None = None,
    ) -> InferenceBackend:
        """Construct one registered inference backend."""
        normalized = self._normalize_name(
            name
        )

        try:
            factory = self._factories[
                normalized
            ]
        except KeyError as exc:
            available = ", ".join(
                sorted(
                    self._factories
                )
            )

            raise ValueError(
                f"Unknown inference backend {normalized!r}. "
                f"Available backends: {available or 'none'}."
            ) from exc

        backend = factory(
            problem,
            {}
            if config is None
            else dict(config),
        )

        if not isinstance(
            backend,
            InferenceBackend,
        ):
            raise TypeError(
                f"Backend factory {normalized!r} did not return "
                "an InferenceBackend instance."
            )

        return backend

    @property
    def names(
        self,
    ) -> tuple[str, ...]:
        """Return registered backend names in deterministic order."""
        return tuple(
            sorted(
                self._factories
            )
        )


backend_registry = BackendRegistry()
