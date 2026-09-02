"""Execution records for scientific validation runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from time import perf_counter
from types import MappingProxyType
from typing import Any

import numpy as np

from baymorph.fitting.backend import InferenceBackend
from baymorph.results import InferenceResult
from baymorph.validation.reference import ReferenceValidationCase


@dataclass(frozen=True)
class ValidationRun:
    """Record one production-backend execution of a validation case.

    The canonical inference result remains the sole owner of posterior,
    evidence, diagnostics, and inference provenance. This object stores only
    validation-specific execution context.

    Parameters
    ----------
    case_name : str
        Stable validation-case identifier.
    backend_name : str
        Stable production-backend identifier.
    result : InferenceResult
        Canonical BayMorph inference result.
    runtime_seconds : float
        Wall-clock runtime measured by the validation harness.
    seed : int, optional
        Random seed assigned by the validation campaign.
    config : mapping, optional
        Production-backend configuration used for the run.
    metadata : mapping, optional
        Additional validation-only provenance.
    """

    case_name: str
    backend_name: str
    result: InferenceResult
    runtime_seconds: float
    seed: int | None = None
    config: Mapping[str, Any] = field(
        default_factory=dict
    )
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        case_name = str(
            self.case_name
        ).strip()
        backend_name = str(
            self.backend_name
        ).strip()

        if not case_name:
            raise ValueError(
                "case_name must not be empty."
            )

        if not backend_name:
            raise ValueError(
                "backend_name must not be empty."
            )

        if not isinstance(
            self.result,
            InferenceResult,
        ):
            raise TypeError(
                "result must be an InferenceResult instance."
            )

        runtime_seconds = float(
            self.runtime_seconds
        )

        if (
            not isfinite(
                runtime_seconds
            )
            or runtime_seconds < 0.0
        ):
            raise ValueError(
                "runtime_seconds must be finite and non-negative."
            )

        seed = (
            None
            if self.seed is None
            else int(
                self.seed
            )
        )

        object.__setattr__(
            self,
            "case_name",
            case_name,
        )
        object.__setattr__(
            self,
            "backend_name",
            backend_name,
        )
        object.__setattr__(
            self,
            "runtime_seconds",
            runtime_seconds,
        )
        object.__setattr__(
            self,
            "seed",
            seed,
        )
        object.__setattr__(
            self,
            "config",
            MappingProxyType(
                dict(
                    self.config
                )
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                dict(
                    self.metadata
                )
            ),
        )

    @property
    def likelihood_calls(self) -> int | None:
        """Return reported likelihood evaluations when available."""
        calls = self.result.diagnostics.get(
            "calls"
        )

        if calls is None:
            return None

        resolved = int(
            calls
        )

        if resolved < 0:
            raise ValueError(
                "Inference-result likelihood calls must be non-negative."
            )

        return resolved


def execute_validation_case(
    case: ReferenceValidationCase,
    backend: InferenceBackend,
    *,
    seed: int | None = None,
    progress_label: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ValidationRun:
    """Execute one validation case through a production inference backend.

    Notes
    -----
    ``seed`` is validation provenance only. The caller must configure the
    production backend with the same seed using that backend's native
    configuration option. This function intentionally performs no
    backend-specific configuration translation.
    """
    if not isinstance(
        case,
        ReferenceValidationCase,
    ):
        raise TypeError(
            "case must be a ReferenceValidationCase instance."
        )

    if not isinstance(
        backend,
        InferenceBackend,
    ):
        raise TypeError(
            "backend must be an InferenceBackend instance."
        )

    if backend.problem is not case.problem:
        raise ValueError(
            "backend must be configured with the validation case problem."
        )

    start = perf_counter()

    result = backend.fit(
        progress_label=progress_label
    )

    runtime_seconds = (
        perf_counter()
        - start
    )

    return ValidationRun(
        case_name=case.name,
        backend_name=backend.name,
        result=result,
        runtime_seconds=runtime_seconds,
        seed=seed,
        config=backend.config,
        metadata=(
            {}
            if metadata is None
            else metadata
        ),
    )


def with_validation_seed(
    backend_name: str,
    config: Mapping[str, Any],
    seed: int | None,
) -> dict[str, Any]:
    """Return production-backend configuration for one validation seed.

    Random-seed configuration remains backend-native. This helper exists only
    in the research validation harness and does not alter BayMorph's production
    inference API.
    """
    name = str(
        backend_name
    ).strip().lower()

    if not name:
        raise ValueError(
            "backend_name must not be empty."
        )

    resolved = dict(
        config
    )

    configured_name = resolved.get(
        "name"
    )

    if (
        configured_name is not None
        and str(
            configured_name
        ).strip().lower()
        != name
    ):
        raise ValueError(
            "Validation backend name conflicts with sampler configuration."
        )

    resolved[
        "name"
    ] = name

    init = dict(
        resolved.get(
            "init",
            {},
        )
    )

    seed_key = {
        "dynesty": "rstate",
        "nautilus": "seed",
        "pocomc": "random_state",
    }.get(
        name
    )

    if seed_key is not None:
        if seed_key in init:
            raise ValueError(
                f"{name} seed configuration is owned by the "
                "validation campaign."
            )

        if seed is not None:
            if name == "dynesty":
                init[
                    seed_key
                ] = np.random.default_rng(
                    int(
                        seed
                    )
                )
            else:
                init[
                    seed_key
                ] = int(
                    seed
                )

    resolved[
        "init"
    ] = init

    return resolved
