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
from baymorph.fitting.emcee import EmceeFitter
from baymorph.fitting.execution import resolve_sampler_execution
from baymorph.fitting.nautilus import NautilusFitter
from baymorph.fitting.pocomc import PocoMCFitter
from baymorph.fitting.problem import InferenceProblem
from baymorph.results import InferenceResult


def _mapping_section(
    config: Mapping[str, Any],
    name: str,
) -> dict[str, Any]:
    """Return one native sampler-configuration section."""
    value = config.get(
        name,
        {},
    )

    if not isinstance(
        value,
        Mapping,
    ):
        raise TypeError(
            f"Sampler configuration section {name!r} must be a mapping."
        )

    return dict(
        value
    )


class DynestyBackend(InferenceBackend):
    """Backend-neutral adapter for Dynesty inference.

    BayMorph owns only the backend name and Dynesty execution mode.
    Entries under ``init`` and ``run`` retain their native Dynesty names
    and semantics.
    """

    def __init__(
        self,
        *,
        problem: InferenceProblem,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        raw = (
            {}
            if config is None
            else dict(config)
        )

        allowed_envelope = {
            "name",
            "mode",
            "execution",
            "init",
            "run",
        }

        unknown = sorted(
            set(raw)
            - allowed_envelope
        )

        if unknown:
            names = ", ".join(
                unknown
            )
            raise ValueError(
                "Dynesty configuration options must be placed under "
                f"'init' or 'run'. Unexpected top-level option(s): {names}."
            )

        mode = str(
            raw.get(
                "mode",
                "static",
            )
        ).strip().lower()

        if mode not in {
            "static",
            "dynamic",
        }:
            raise ValueError(
                "Dynesty mode must be 'static' or 'dynamic'."
            )

        resolved = {
            "name": "dynesty",
            "mode": mode,
            "execution": _mapping_section(
                raw,
                "execution",
            ),
            "init": _mapping_section(
                raw,
                "init",
            ),
            "run": _mapping_section(
                raw,
                "run",
            ),
        }

        super().__init__(
            problem=problem,
            config=resolved,
        )

    @property
    def name(self) -> str:
        """Return the stable backend identifier."""
        return "dynesty"

    @property
    def mode(self) -> str:
        """Return the selected Dynesty nested-sampling mode."""
        return str(
            self.config["mode"]
        )

    def fit(
        self,
        *,
        progress_label: str | None = None,
    ) -> InferenceResult:
        """Run Dynesty with native constructor and run keyword arguments."""
        fitter = DynestyFitter(
            problem=self.problem
        )

        return fitter.run(
            mode=self.mode,
            init_kwargs=self.config[
                "init"
            ],
            run_kwargs=self.config[
                "run"
            ],
            execution=resolve_sampler_execution(
                self.config[
                    "execution"
                ]
            ),
            progress_label=progress_label,
        )


class EmceeBackend(InferenceBackend):
    """Backend-neutral adapter for posterior-only emcee inference."""

    def __init__(
        self,
        *,
        problem: InferenceProblem,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        raw = (
            {}
            if config is None
            else dict(config)
        )

        allowed_envelope = {
            "name",
            "execution",
            "init",
            "run",
        }

        unknown = sorted(
            set(raw)
            - allowed_envelope
        )

        if unknown:
            names = ", ".join(unknown)
            raise ValueError(
                "emcee configuration options must be placed under "
                f"'init' or 'run'. Unexpected top-level option(s): {names}."
            )

        resolved = {
            "name": "emcee",
            "execution": _mapping_section(
                raw,
                "execution",
            ),
            "init": _mapping_section(
                raw,
                "init",
            ),
            "run": _mapping_section(
                raw,
                "run",
            ),
        }

        super().__init__(
            problem=problem,
            config=resolved,
        )

    @property
    def name(self) -> str:
        """Return the stable backend identifier."""
        return "emcee"

    @property
    def produces_evidence(self) -> bool:
        """Return that emcee is a posterior-only backend."""
        return False

    def fit(
        self,
        *,
        progress_label: str | None = None,
    ) -> InferenceResult:
        """Run posterior-only emcee inference."""
        fitter = EmceeFitter(
            problem=self.problem
        )

        return fitter.run(
            init_kwargs=self.config[
                "init"
            ],
            run_kwargs=self.config[
                "run"
            ],
            execution=resolve_sampler_execution(
                self.config[
                    "execution"
                ]
            ),
            progress_label=progress_label,
        )


class PocoMCBackend(InferenceBackend):
    """Backend-neutral adapter for pocoMC inference.

    Entries under ``init`` and ``run`` retain their native pocoMC names
    and semantics.
    """

    def __init__(
        self,
        *,
        problem: InferenceProblem,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        raw = (
            {}
            if config is None
            else dict(config)
        )

        allowed_envelope = {
            "name",
            "execution",
            "init",
            "run",
        }

        unknown = sorted(
            set(raw)
            - allowed_envelope
        )

        if unknown:
            names = ", ".join(
                unknown
            )
            raise ValueError(
                "pocoMC configuration options must be placed under "
                f"'init' or 'run'. Unexpected top-level option(s): {names}."
            )

        resolved = {
            "name": "pocomc",
            "execution": _mapping_section(
                raw,
                "execution",
            ),
            "init": _mapping_section(
                raw,
                "init",
            ),
            "run": _mapping_section(
                raw,
                "run",
            ),
        }

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
    ) -> InferenceResult:
        """Run pocoMC with native constructor and run keyword arguments."""
        fitter = PocoMCFitter(
            problem=self.problem
        )

        return fitter.run(
            init_kwargs=self.config[
                "init"
            ],
            run_kwargs=self.config[
                "run"
            ],
            execution=resolve_sampler_execution(
                self.config[
                    "execution"
                ]
            ),
            progress_label=progress_label,
        )


class NautilusBackend(InferenceBackend):
    """Backend-neutral adapter for Nautilus inference.

    Entries under ``init`` and ``run`` retain their native Nautilus names
    and semantics.
    """

    def __init__(
        self,
        *,
        problem: InferenceProblem,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        raw = (
            {}
            if config is None
            else dict(config)
        )

        allowed_envelope = {
            "name",
            "execution",
            "init",
            "run",
        }

        unknown = sorted(
            set(raw)
            - allowed_envelope
        )

        if unknown:
            names = ", ".join(
                unknown
            )
            raise ValueError(
                "Nautilus configuration options must be placed under "
                f"'init' or 'run'. Unexpected top-level option(s): {names}."
            )

        resolved = {
            "name": "nautilus",
            "execution": _mapping_section(
                raw,
                "execution",
            ),
            "init": _mapping_section(
                raw,
                "init",
            ),
            "run": _mapping_section(
                raw,
                "run",
            ),
        }

        super().__init__(
            problem=problem,
            config=resolved,
        )

    @property
    def name(self) -> str:
        """Return the stable backend identifier."""
        return "nautilus"

    def fit(
        self,
        *,
        progress_label: str | None = None,
    ) -> InferenceResult:
        """Run Nautilus with native constructor and run keyword arguments."""
        fitter = NautilusFitter(
            problem=self.problem
        )

        return fitter.run(
            init_kwargs=self.config[
                "init"
            ],
            run_kwargs=self.config[
                "run"
            ],
            execution=resolve_sampler_execution(
                self.config[
                    "execution"
                ]
            ),
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


def _make_emcee_backend(
    problem: InferenceProblem,
    config: Mapping[str, Any],
) -> InferenceBackend:
    """Construct the built-in emcee backend."""
    return EmceeBackend(
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


def _make_nautilus_backend(
    problem: InferenceProblem,
    config: Mapping[str, Any],
) -> InferenceBackend:
    """Construct the built-in Nautilus backend."""
    return NautilusBackend(
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
        "emcee",
        _make_emcee_backend,
        replace=replace,
    )

    registry.register(
        "nautilus",
        _make_nautilus_backend,
        replace=replace,
    )

    registry.register(
        "pocomc",
        _make_pocomc_backend,
        replace=replace,
    )
