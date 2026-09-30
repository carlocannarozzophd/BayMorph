"""Shared checkpoint-signature machinery for BayMorph fitting backends.

A checkpoint (or its resume-target file) must never be trusted unless it is
signed with the exact scientific problem it was produced for: parameters and
priors, prepared inputs, mask, PSF, likelihood, and sampler settings. This
module owns the canonical, backend-neutral pieces of that contract:

- deterministic signature-value canonicalization for arbitrary Python and
  NumPy state (:func:`checkpoint_signature_value` and friends);
- the SHA-256 digest of strict canonical JSON
  (:func:`canonical_json_sha256`), shared by every signature and completion
  marker of both fit workflows;
- a generic sibling ``<checkpoint>.target.json`` signing contract for native
  checkpoint formats that BayMorph does not itself annotate internally
  (:func:`resolve_json_signed_checkpoint`), used by both Nautilus and pocoMC,
  for both single-observation and multi-observation fits;
- the resolution of each backend's native checkpoint path in the sampler
  configuration (:func:`prepare_signed_sampler_checkpoint`).

Backend- and target-specific signature payloads (what exactly gets hashed
for one real fit vs. one multi-observation joint fit) and signature schemas
remain the responsibility of their own pipeline modules.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import sys
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def write_json_atomic(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    """Write one JSON document atomically in its destination directory."""
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )

    try:
        with temporary.open(
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(
                payload,
                stream,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            stream.write(
                "\n"
            )
            stream.flush()
            os.fsync(
                stream.fileno()
            )

        os.replace(
            temporary,
            path,
        )
    finally:
        temporary.unlink(
            missing_ok=True
        )


def qualified_class_name(
    value,
) -> str:
    """Return a stable module-qualified class name."""
    value_type = type(
        value
    )

    return (
        f"{value_type.__module__}."
        f"{value_type.__qualname__}"
    )


def checkpoint_array_signature(
    value,
) -> dict[str, Any]:
    """Return a canonical signature for one numerical array."""
    array = np.ascontiguousarray(
        np.asarray(
            value
        )
    )

    if array.dtype.hasobject:
        raise TypeError(
            "Checkpoint signatures do not support object arrays."
        )

    return {
        "dtype": array.dtype.str,
        "shape": list(
            array.shape
        ),
        "sha256": hashlib.sha256(
            array.tobytes(
                order="C"
            )
        ).hexdigest(),
    }


def checkpoint_signature_value(
    value,
):
    """Convert one value into strict canonical signature metadata."""
    if value is None:
        return None

    if isinstance(
        value,
        np.ndarray,
    ):
        return {
            "__array__": checkpoint_array_signature(
                value
            )
        }

    if isinstance(
        value,
        np.generic,
    ):
        return checkpoint_signature_value(
            value.item()
        )

    if isinstance(
        value,
        Path,
    ):
        return {
            "__path__": str(
                value
            )
        }

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        int,
    ):
        return value

    if isinstance(
        value,
        float,
    ):
        if not np.isfinite(
            value
        ):
            raise ValueError(
                "Checkpoint signature scalars must be finite."
            )

        return value

    if isinstance(
        value,
        str,
    ):
        return value

    if isinstance(
        value,
        bytes,
    ):
        return {
            "__bytes_sha256__": hashlib.sha256(
                value
            ).hexdigest()
        }

    if isinstance(
        value,
        dict,
    ):
        if not all(
            isinstance(
                key,
                str,
            )
            for key in value
        ):
            raise TypeError(
                "Checkpoint signature mappings require string keys."
            )

        return {
            key: checkpoint_signature_value(
                value[
                    key
                ]
            )
            for key in sorted(
                value
            )
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            checkpoint_signature_value(
                item
            )
            for item in value
        ]

    raise TypeError(
        "Unsupported checkpoint-signature value "
        f"{qualified_class_name(value)!r}."
    )


def prior_checkpoint_signature(
    prior,
) -> dict[str, Any]:
    """Return the complete canonical state of one active prior."""
    if dataclasses.is_dataclass(
        prior
    ):
        state = {
            field.name: getattr(
                prior,
                field.name,
            )
            for field in dataclasses.fields(
                prior
            )
        }
    else:
        try:
            state = dict(
                vars(
                    prior
                )
            )
        except TypeError as exc:
            raise TypeError(
                "Checkpointing requires priors with canonical "
                "dataclass fields or instance state."
            ) from exc

    return {
        "class": qualified_class_name(
            prior
        ),
        "state": checkpoint_signature_value(
            state
        ),
    }


def runtime_psf_checkpoint_signature(
    psf,
) -> dict[str, Any] | None:
    """Return the effective numerical state of the runtime PSF."""
    if psf is None:
        return None

    if isinstance(
        psf,
        np.ndarray,
    ):
        return {
            "class": qualified_class_name(
                psf
            ),
            "array": checkpoint_array_signature(
                psf
            ),
        }

    try:
        candidate_array = np.asarray(
            psf
        )
    except (
        TypeError,
        ValueError,
    ):
        candidate_array = None

    if (
        candidate_array is not None
        and candidate_array.ndim > 0
        and not candidate_array.dtype.hasobject
    ):
        return {
            "class": qualified_class_name(
                psf
            ),
            "array": checkpoint_array_signature(
                candidate_array
            ),
        }

    if dataclasses.is_dataclass(
        psf
    ):
        state = {
            field.name: getattr(
                psf,
                field.name,
            )
            for field in dataclasses.fields(
                psf
            )
        }
    else:
        try:
            state = dict(
                vars(
                    psf
                )
            )
        except TypeError:
            state = None

        if state is None:
            slots = getattr(
                type(
                    psf
                ),
                "__slots__",
                (),
            )

            if isinstance(
                slots,
                str,
            ):
                slots = (
                    slots,
                )

            state = {
                name: getattr(
                    psf,
                    name,
                )
                for name in slots
                if not name.startswith(
                    "__"
                )
                and hasattr(
                    psf,
                    name,
                )
            }

    return {
        "class": qualified_class_name(
            psf
        ),
        "state": checkpoint_signature_value(
            state
        ),
    }


def baymorph_source_signature() -> dict[str, Any]:
    """Hash the installed BayMorph Python source tree."""
    package_root = Path(
        __file__
    ).resolve().parents[
        1
    ]

    source_paths = sorted(
        package_root.rglob(
            "*.py"
        ),
        key=lambda path: path.relative_to(
            package_root
        ).as_posix(),
    )

    if not source_paths:
        raise RuntimeError(
            "No BayMorph Python sources were found for "
            "checkpoint fingerprinting."
        )

    digest = hashlib.sha256()

    for source_path in source_paths:
        relative = source_path.relative_to(
            package_root
        ).as_posix().encode(
            "utf-8"
        )
        contents = source_path.read_bytes()

        digest.update(
            len(
                relative
            ).to_bytes(
                8,
                "big",
            )
        )
        digest.update(
            relative
        )
        digest.update(
            len(
                contents
            ).to_bytes(
                8,
                "big",
            )
        )
        digest.update(
            contents
        )

    return {
        "algorithm": "sha256",
        "file_count": len(
            source_paths
        ),
        "digest": digest.hexdigest(),
    }


def checkpoint_runtime_versions() -> dict[str, Any]:
    """Return conservative numerical-runtime version metadata."""
    versions: dict[str, Any] = {
        "python": (
            f"{sys.version_info.major}."
            f"{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        ),
    }

    for distribution in (
        "baymorph",
        "numpy",
        "scipy",
        "astropy",
        "photutils",
    ):
        try:
            versions[
                distribution
            ] = importlib.metadata.version(
                distribution
            )
        except importlib.metadata.PackageNotFoundError:
            versions[
                distribution
            ] = None

    return versions


def canonical_json_sha256(
    value: Any,
) -> str:
    """Return the SHA-256 digest of strict canonical JSON.

    Keys are sorted, separators carry no whitespace, the output is ASCII and
    non-finite numbers are rejected, so equal values always serialize to the
    same bytes. Every checkpoint signature and fit completion marker hashes
    through this function, so changing its serialization would invalidate
    all of them.

    Parameters
    ----------
    value : Any
        JSON-compatible value, typically canonicalized first with
        :func:`checkpoint_signature_value`.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest of the UTF-8 serialization.
    """
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=True,
        allow_nan=False,
    )

    return hashlib.sha256(
        canonical.encode(
            "utf-8"
        )
    ).hexdigest()


def resolve_json_signed_checkpoint(
    *,
    checkpoint_path: Path,
    schema: str,
    problem_signature: str,
    runtime_sampler_config: Mapping[str, Any],
    label: str,
    require_resume_flag: bool,
    resume_requested: bool,
) -> None:
    """Validate or create a sibling ``<checkpoint>.target.json`` signature.

    Used for native checkpoint formats that BayMorph does not itself
    annotate internally (Nautilus's HDF5 file, pocoMC's native state file),
    for both single-observation and multi-observation fits.

    Parameters
    ----------
    checkpoint_path : Path
        Resolved path to the native checkpoint (or resume-source) file.
    schema : str
        Schema tag recorded in the signature payload, to keep signatures
        from different checkpoint formats or target kinds from ever being
        compared as if they were the same contract.
    problem_signature : str
        Strict signature of the scientific posterior target, as computed by
        the caller (for example ``_real_fit_problem_signature`` or a
        multi-observation equivalent).
    runtime_sampler_config : mapping
        The fully resolved sampler configuration passed to the native
        backend, folded into the signature so sampler-setting changes also
        invalidate a stale checkpoint.
    label : str
        Human-readable checkpoint identity used in raised error messages
        (for example ``"Nautilus checkpoint"``).
    require_resume_flag : bool
        If True, an existing checkpoint file requires ``resume_requested``
        to be True, or a :class:`FileExistsError` is raised. Pass False when
        the native backend has no separate "start fresh at this path"
        convention (for example pocoMC's ``resume_state_path``, whose mere
        presence already means "resume").
    resume_requested : bool
        Whether the caller declared intent to resume from an existing
        checkpoint at this path. Ignored when ``require_resume_flag`` is
        False.

    Raises
    ------
    FileExistsError
        If `checkpoint_path` already exists, `require_resume_flag` is True,
        and `resume_requested` is False.
    RuntimeError
        If a checkpoint or its signature file exists but does not match the
        current scientific problem, or if a checkpoint exists with no
        recorded signature at all.
    """
    signature_path = checkpoint_path.with_suffix(
        checkpoint_path.suffix
        + ".target.json"
    )
    signature_payload = {
        "schema": schema,
        "problem_signature": problem_signature,
        "sampler": checkpoint_signature_value(
            dict(
                runtime_sampler_config
            )
        ),
    }
    expected = {
        "schema_version": 1,
        "signature": (
            "sha256:"
            + canonical_json_sha256(
                signature_payload
            )
        ),
        "target": signature_payload,
    }

    if checkpoint_path.exists():
        if (
            require_resume_flag
            and not resume_requested
        ):
            raise FileExistsError(
                f"A {label} already exists while resume=false: "
                f"{checkpoint_path}"
            )

        if not signature_path.is_file():
            raise RuntimeError(
                f"{label} has no BayMorph target signature: "
                f"{signature_path}"
            )

        with signature_path.open(
            "r",
            encoding="utf-8",
        ) as stream:
            recorded = json.load(
                stream
            )

        if recorded != expected:
            raise RuntimeError(
                f"{label} does not match the current scientific "
                f"problem: {checkpoint_path}"
            )
    elif signature_path.exists():
        with signature_path.open(
            "r",
            encoding="utf-8",
        ) as stream:
            recorded = json.load(
                stream
            )

        if recorded != expected:
            raise RuntimeError(
                f"{label} does not match the current scientific "
                f"problem: {signature_path}"
            )
    else:
        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        write_json_atomic(
            signature_path,
            expected,
        )


@dataclasses.dataclass(frozen=True)
class SamplerCheckpoint:
    """Where one backend keeps its native checkpoint in a sampler config.

    Attributes
    ----------
    section : str
        Sampler-configuration section holding the checkpoint path, ``init``
        or ``run``.
    path_key : str
        Native option naming the checkpoint path in that section.
    auto_filename : str
        File name used when the path is declared as ``auto``.
    display_name : str
        Backend name used in error messages.
    label : str
        Checkpoint identity used in signature error messages.
    resume_key : str or None
        Native option selecting resume when a checkpoint exists, or ``None``
        when the presence of the path already means resume.
    """

    section: str
    path_key: str
    auto_filename: str
    display_name: str
    label: str
    resume_key: str | None = None


SAMPLER_CHECKPOINTS = {
    "nautilus": SamplerCheckpoint(
        section="init",
        path_key="filepath",
        auto_filename="nautilus_checkpoint.hdf5",
        display_name="Nautilus",
        label="Nautilus checkpoint",
        resume_key="resume",
    ),
    "pocomc": SamplerCheckpoint(
        section="run",
        path_key="resume_state_path",
        auto_filename="pocomc_state.h5",
        display_name="pocoMC",
        label="pocoMC resume state",
    ),
}


def prepare_signed_sampler_checkpoint(
    sampler_config: Mapping[str, Any],
    *,
    schemas: Mapping[str, str],
    checkpoint_directory: str | Path | None,
    problem_signature: Callable[[], str],
) -> Mapping[str, Any]:
    """Resolve and authenticate the native checkpoint of one model fit.

    Nautilus keeps its checkpoint at ``init.filepath`` and pocoMC resumes
    from ``run.resume_state_path``. Neither format records which scientific
    problem produced it, so whenever such a path is declared the sibling
    ``.target.json`` signature is validated or created with
    :func:`resolve_json_signed_checkpoint`, and BayMorph never resumes a
    checkpoint of another problem. A path declared as ``auto`` resolves
    inside the model output directory.

    Parameters
    ----------
    sampler_config : mapping
        Resolved sampler configuration of one model fit.
    schemas : mapping of str to str
        Signature schema of each checkpointing backend, owned by the calling
        workflow so that single-observation and joint signatures are never
        compared as if they were the same contract.
    checkpoint_directory : str, Path or None
        Pipeline-managed output directory of the model, required by an
        ``auto`` path.
    problem_signature : callable
        Returns the strict signature of the scientific problem. It is called
        only when a checkpoint path is declared.

    Returns
    -------
    mapping
        ``sampler_config`` itself for a backend without a checkpoint or when
        no checkpoint path is declared; otherwise a copy with the resolved
        checkpoint path.

    Raises
    ------
    ValueError
        If an ``auto`` path is declared without a checkpoint directory.
    """
    sampler_name = str(
        sampler_config.get(
            "name",
            "",
        )
    ).strip().lower()
    checkpoint = SAMPLER_CHECKPOINTS.get(
        sampler_name
    )

    if checkpoint is None:
        return sampler_config

    runtime_config = deepcopy(
        dict(
            sampler_config
        )
    )
    section = dict(
        runtime_config.get(
            checkpoint.section,
            {},
        )
    )
    runtime_config[
        checkpoint.section
    ] = section
    configured_path = section.get(
        checkpoint.path_key
    )

    if configured_path is None:
        return runtime_config

    if (
        isinstance(
            configured_path,
            str,
        )
        and configured_path.strip().lower()
        == "auto"
    ):
        if checkpoint_directory is None:
            raise ValueError(
                f"{checkpoint.display_name} {checkpoint.path_key}='auto' "
                "requires a pipeline-managed model output directory."
            )

        configured_path = str(
            (
                Path(
                    checkpoint_directory
                )
                / checkpoint.auto_filename
            ).expanduser().resolve()
        )
        section[
            checkpoint.path_key
        ] = configured_path

    resolve_json_signed_checkpoint(
        checkpoint_path=Path(
            configured_path
        ).expanduser().resolve(),
        schema=schemas[
            sampler_name
        ],
        problem_signature=problem_signature(),
        runtime_sampler_config=runtime_config,
        label=checkpoint.label,
        require_resume_flag=(
            checkpoint.resume_key is not None
        ),
        resume_requested=(
            True
            if checkpoint.resume_key is None
            else bool(
                section.get(
                    checkpoint.resume_key,
                    True,
                )
            )
        ),
    )

    return runtime_config
