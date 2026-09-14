"""Safe HDF checkpoint support for posterior-only emcee runs."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_CHECKPOINT_SCHEMA = "baymorph-emcee-checkpoint-v1"
EMCEE_CHECKPOINT_CHAIN_GROUP = "mcmc"

_SCHEMA_ATTR = "baymorph_checkpoint_schema"
_SIGNATURE_ATTR = "baymorph_checkpoint_signature"
_METADATA_ATTR = "baymorph_checkpoint_metadata"
_METADATA_HASH_ATTR = "baymorph_checkpoint_metadata_sha256"
_INITIALIZATION_CALLS_ATTR = "baymorph_initialization_calls"
_READY_ATTR = "baymorph_checkpoint_ready"
_CONVERGED_ATTR = "baymorph_converged"
_PREVIOUS_TAU_ATTR = "baymorph_previous_tau"


@dataclass(frozen=True)
class EmceeCheckpointRequest:
    """One explicit HDF checkpoint request."""

    path: Path
    mode: str
    signature: str


@dataclass(frozen=True)
class EmceeCheckpointState:
    """Validated state associated with one HDF checkpoint."""

    request: EmceeCheckpointRequest
    backend: Any
    initialization_calls: int
    converged: bool
    previous_tau: np.ndarray | None

    @property
    def resumed(self) -> bool:
        """Return whether an existing checkpoint was loaded."""
        return self.request.mode == "resume"


def _require_h5py():
    """Import h5py with a checkpoint-specific error."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "emcee HDF checkpointing requires h5py."
        ) from exc

    return h5py


def pop_emcee_checkpoint_request(
    init_config: dict[str, Any],
) -> EmceeCheckpointRequest | None:
    """Remove and validate BayMorph-owned checkpoint options."""
    raw_path = init_config.pop(
        "checkpoint_path",
        None,
    )
    raw_mode = init_config.pop(
        "checkpoint_mode",
        None,
    )
    raw_signature = init_config.pop(
        "checkpoint_signature",
        None,
    )

    if raw_path is None:
        if (
            raw_mode is not None
            or raw_signature is not None
        ):
            raise ValueError(
                "emcee checkpoint_mode and checkpoint_signature "
                "require checkpoint_path."
            )

        return None

    if not isinstance(
        raw_path,
        (str, os.PathLike),
    ):
        raise TypeError(
            "emcee checkpoint_path must be a filesystem path."
        )

    if (
        isinstance(raw_path, str)
        and not raw_path.strip()
    ):
        raise ValueError(
            "emcee checkpoint_path cannot be empty."
        )

    if (
        isinstance(raw_path, str)
        and raw_path.strip().lower() == "auto"
    ):
        raise ValueError(
            "emcee checkpoint_path='auto' must be resolved by "
            "the single-observation real-fit pipeline."
        )

    mode = str(
        raw_mode
    ).strip().lower()

    if mode not in {
        "fresh",
        "resume",
    }:
        raise ValueError(
            "emcee checkpoint_mode must be 'fresh' or 'resume'."
        )

    if (
        not isinstance(
            raw_signature,
            str,
        )
        or not raw_signature.strip()
    ):
        raise ValueError(
            "emcee checkpoint_signature must be a non-empty string."
        )

    return EmceeCheckpointRequest(
        path=Path(
            raw_path
        ).expanduser().resolve(),
        mode=mode,
        signature=raw_signature.strip(),
    )


def emcee_checkpoint_metadata(
    *,
    request: EmceeCheckpointRequest,
    emcee_version: str,
    problem_class: str,
    ndim: int,
    parameter_names: list[str],
    nwalkers: int,
    burn_in: int,
    seed: int,
    max_initialization_attempts: int,
    init_config: dict[str, Any],
    run_config: dict[str, Any],
    check_interval: int,
    min_autocorr_times: float,
    tau_tolerance: float,
) -> str:
    """Return canonical checkpoint-compatibility metadata."""
    payload = {
        "schema": _CHECKPOINT_SCHEMA,
        "checkpoint_signature": request.signature,
        "backend": "emcee",
        "coordinate_space": "unit_cube",
        "emcee_version": str(
            emcee_version
        ),
        "problem_class": str(
            problem_class
        ),
        "ndim": int(
            ndim
        ),
        "parameter_names": list(
            parameter_names
        ),
        "nwalkers": int(
            nwalkers
        ),
        "burn_in": int(
            burn_in
        ),
        "seed": int(
            seed
        ),
        "max_initialization_attempts": int(
            max_initialization_attempts
        ),
        "init": init_config,
        "sample": run_config,
        "check_interval": int(
            check_interval
        ),
        "min_autocorr_times": float(
            min_autocorr_times
        ),
        "tau_tolerance": float(
            tau_tolerance
        ),
        "max_steps_semantics": (
            "total_production_target"
        ),
    }

    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise TypeError(
            "Checkpointed emcee configuration must contain "
            "only stable JSON-compatible values."
        ) from exc


def ensure_fresh_checkpoint_available(
    request: EmceeCheckpointRequest,
) -> None:
    """Fail before sampling if a fresh path already exists."""
    if request.mode != "fresh":
        return

    if request.path.exists():
        raise FileExistsError(
            "Refusing to overwrite existing emcee checkpoint: "
            f"{request.path}"
        )

    request.path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def create_emcee_checkpoint(
    *,
    request: EmceeCheckpointRequest,
    emcee_library,
    nwalkers: int,
    ndim: int,
    metadata: str,
    initialization_calls: int,
) -> EmceeCheckpointState:
    """Create a new checkpoint for production samples."""
    if request.mode != "fresh":
        raise ValueError(
            "Creating a checkpoint requires "
            "checkpoint_mode='fresh'."
        )

    h5py = _require_h5py()

    try:
        with h5py.File(
            request.path,
            "x",
        ):
            pass
    except FileExistsError as exc:
        raise FileExistsError(
            "Refusing to overwrite existing emcee checkpoint: "
            f"{request.path}"
        ) from exc

    backend = emcee_library.backends.HDFBackend(
        str(
            request.path
        ),
        name=EMCEE_CHECKPOINT_CHAIN_GROUP,
    )
    backend.reset(
        nwalkers,
        ndim,
    )

    metadata_hash = hashlib.sha256(
        metadata.encode(
            "utf-8"
        )
    ).hexdigest()

    with backend.open(
        "a"
    ) as handle:
        group = handle[
            EMCEE_CHECKPOINT_CHAIN_GROUP
        ]
        group.attrs[
            _SCHEMA_ATTR
        ] = _CHECKPOINT_SCHEMA
        group.attrs[
            _SIGNATURE_ATTR
        ] = request.signature
        group.attrs[
            _METADATA_ATTR
        ] = metadata
        group.attrs[
            _METADATA_HASH_ATTR
        ] = metadata_hash
        group.attrs[
            _INITIALIZATION_CALLS_ATTR
        ] = int(
            initialization_calls
        )
        group.attrs[
            _CONVERGED_ATTR
        ] = False

        # Written last. Missing or False means incomplete setup.
        group.attrs[
            _READY_ATTR
        ] = True

        handle.flush()

    return EmceeCheckpointState(
        request=request,
        backend=backend,
        initialization_calls=int(
            initialization_calls
        ),
        converged=False,
        previous_tau=None,
    )


def load_emcee_checkpoint(
    *,
    request: EmceeCheckpointRequest,
    emcee_library,
    metadata: str,
    nwalkers: int,
    ndim: int,
) -> EmceeCheckpointState:
    """Validate an existing checkpoint without changing it."""
    if request.mode != "resume":
        raise ValueError(
            "Loading a checkpoint requires "
            "checkpoint_mode='resume'."
        )

    if not request.path.is_file():
        raise FileNotFoundError(
            "emcee checkpoint does not exist: "
            f"{request.path}"
        )

    _require_h5py()

    try:
        backend = emcee_library.backends.HDFBackend(
            str(
                request.path
            ),
            name=EMCEE_CHECKPOINT_CHAIN_GROUP,
        )

        if not backend.initialized:
            raise RuntimeError(
                "emcee checkpoint is not initialized."
            )

        required_attributes = {
            _SCHEMA_ATTR,
            _SIGNATURE_ATTR,
            _METADATA_ATTR,
            _METADATA_HASH_ATTR,
            _INITIALIZATION_CALLS_ATTR,
            _READY_ATTR,
            _CONVERGED_ATTR,
        }

        with backend.open(
            "r"
        ) as handle:
            group = handle[
                EMCEE_CHECKPOINT_CHAIN_GROUP
            ]
            missing = sorted(
                required_attributes
                - set(
                    group.attrs
                )
            )

            if missing:
                raise RuntimeError(
                    "emcee checkpoint metadata is incomplete: "
                    + ", ".join(
                        missing
                    )
                )

            if not bool(
                group.attrs[
                    _READY_ATTR
                ]
            ):
                raise RuntimeError(
                    "emcee checkpoint initialization "
                    "did not complete."
                )

            schema = str(
                group.attrs[
                    _SCHEMA_ATTR
                ]
            )
            stored_signature = str(
                group.attrs[
                    _SIGNATURE_ATTR
                ]
            )
            stored_metadata = str(
                group.attrs[
                    _METADATA_ATTR
                ]
            )
            stored_hash = str(
                group.attrs[
                    _METADATA_HASH_ATTR
                ]
            )
            initialization_calls = int(
                group.attrs[
                    _INITIALIZATION_CALLS_ATTR
                ]
            )
            converged = bool(
                group.attrs[
                    _CONVERGED_ATTR
                ]
            )

            previous_tau = None

            if _PREVIOUS_TAU_ATTR in group.attrs:
                previous_tau = np.asarray(
                    group.attrs[
                        _PREVIOUS_TAU_ATTR
                    ],
                    dtype=float,
                )
    except (
        KeyError,
        OSError,
    ) as exc:
        raise RuntimeError(
            "emcee checkpoint cannot be read safely."
        ) from exc

    if schema != _CHECKPOINT_SCHEMA:
        raise ValueError(
            "Unsupported emcee checkpoint schema."
        )

    if stored_signature != request.signature:
        raise ValueError(
            "emcee checkpoint_signature does not match."
        )

    actual_hash = hashlib.sha256(
        stored_metadata.encode(
            "utf-8"
        )
    ).hexdigest()

    if actual_hash != stored_hash:
        raise ValueError(
            "emcee checkpoint metadata checksum is invalid."
        )

    if stored_metadata != metadata:
        raise ValueError(
            "emcee checkpoint configuration does not match "
            "the requested inference problem."
        )

    if backend.shape != (
        nwalkers,
        ndim,
    ):
        raise ValueError(
            "emcee checkpoint walker or parameter "
            "dimensions do not match."
        )

    if previous_tau is not None:
        if (
            previous_tau.shape != (ndim,)
            or not np.all(
                np.isfinite(
                    previous_tau
                )
            )
            or np.any(
                previous_tau <= 0.0
            )
        ):
            raise ValueError(
                "emcee checkpoint contains invalid "
                "autocorrelation metadata."
            )

    if backend.iteration < 1:
        raise RuntimeError(
            "emcee checkpoint contains no production step "
            "and cannot be resumed safely. "
            "Use a new checkpoint path."
        )

    return EmceeCheckpointState(
        request=request,
        backend=backend,
        initialization_calls=initialization_calls,
        converged=converged,
        previous_tau=previous_tau,
    )


def record_emcee_checkpoint_progress(
    checkpoint: EmceeCheckpointState | None,
    *,
    tau: np.ndarray | None = None,
    converged: bool | None = None,
) -> None:
    """Persist convergence metadata after stored steps."""
    if checkpoint is None:
        return

    with checkpoint.backend.open(
        "a"
    ) as handle:
        group = handle[
            EMCEE_CHECKPOINT_CHAIN_GROUP
        ]

        if tau is not None:
            group.attrs[
                _PREVIOUS_TAU_ATTR
            ] = np.asarray(
                tau,
                dtype=float,
            )

        if converged is not None:
            group.attrs[
                _CONVERGED_ATTR
            ] = bool(
                converged
            )

        handle.flush()


__all__ = [
    "EmceeCheckpointRequest",
    "EmceeCheckpointState",
    "create_emcee_checkpoint",
    "emcee_checkpoint_metadata",
    "ensure_fresh_checkpoint_available",
    "load_emcee_checkpoint",
    "pop_emcee_checkpoint_request",
    "record_emcee_checkpoint_progress",
]
