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


class _NativeSamplerBackend(InferenceBackend):
    """Thin BayMorph envelope around one built-in sampler.

    BayMorph owns the backend name, the ``execution`` section and any
    backend-specific envelope option. Entries under ``init`` and ``run``
    retain their native upstream names and semantics.
    """

    backend_name: str
    display_name: str
    fitter_class: type
    envelope_options: tuple[str, ...] = ()

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
            *self.envelope_options,
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
                f"{self.display_name} configuration options must be placed "
                "under 'init' or 'run'. Unexpected top-level option(s): "
                f"{names}."
            )

        resolved = {
            "name": self.backend_name,
            **self._resolve_envelope_options(
                raw
            ),
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

    def _resolve_envelope_options(
        self,
        raw: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return the validated backend-specific envelope options."""
        return {}

    def _run_options(
        self,
    ) -> dict[str, Any]:
        """Return the backend-specific options of the fitter run."""
        return {}

    @property
    def name(self) -> str:
        """Return the stable backend identifier."""
        return self.backend_name

    def fit(
        self,
        *,
        progress_label: str | None = None,
    ) -> InferenceResult:
        """Run the sampler with native constructor and run keyword arguments."""
        fitter = self.fitter_class(
            problem=self.problem
        )

        return fitter.run(
            **self._run_options(),
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


class DynestyBackend(_NativeSamplerBackend):
    """Backend-neutral adapter for Dynesty inference.

    BayMorph owns only the backend name and Dynesty execution mode.
    Entries under ``init`` and ``run`` retain their native Dynesty names
    and semantics.
    """

    backend_name = "dynesty"
    display_name = "Dynesty"
    fitter_class = DynestyFitter
    envelope_options = ("mode",)

    def _resolve_envelope_options(
        self,
        raw: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return the validated Dynesty nested-sampling mode."""
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

        return {
            "mode": mode,
        }

    def _run_options(
        self,
    ) -> dict[str, Any]:
        """Pass the selected nested-sampling mode to the fitter."""
        return {
            "mode": self.mode,
        }

    @property
    def mode(self) -> str:
        """Return the selected Dynesty nested-sampling mode."""
        return str(
            self.config["mode"]
        )


class PocoMCBackend(_NativeSamplerBackend):
    """Backend-neutral adapter for pocoMC inference.

    Entries under ``init`` and ``run`` retain their native pocoMC names
    and semantics.
    """

    backend_name = "pocomc"
    display_name = "pocoMC"
    fitter_class = PocoMCFitter


class NautilusBackend(_NativeSamplerBackend):
    """Backend-neutral adapter for Nautilus inference.

    Entries under ``init`` and ``run`` retain their native Nautilus names
    and semantics.
    """

    backend_name = "nautilus"
    display_name = "Nautilus"
    fitter_class = NautilusFitter


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
        "nautilus",
        _make_nautilus_backend,
        replace=replace,
    )

    registry.register(
        "pocomc",
        _make_pocomc_backend,
        replace=replace,
    )
