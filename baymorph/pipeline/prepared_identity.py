"""Immutable identity for persisted image-preparation products."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from baymorph.core.observation import Observation
from baymorph.pipeline.config import resolve_config, resolve_runtime_config
from baymorph.pipeline.opening import REUSE_MIN_BYTES, file_state, open_observation

PREPARED_INPUT_IDENTITY_VERSION = 2

_CONTENT_IDENTITIES: dict[tuple, dict[str, Any]] = {}


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def file_content_identity(path: Path) -> dict[str, Any]:
    """Hash a file and reject mutation during fingerprinting.

    The digest of a file of at least
    :data:`baymorph.pipeline.opening.REUSE_MIN_BYTES` is computed once per
    process and reused while the state of the file is unchanged; a smaller
    file is hashed at every call.
    """
    before_state = file_state(path)
    key = (str(path), *before_state)
    reusable = before_state[2] >= REUSE_MIN_BYTES

    if reusable and key in _CONTENT_IDENTITIES:
        return dict(_CONTENT_IDENTITIES[key])

    digest = sha256_file(path)
    after_state = file_state(path)

    if before_state[:4] != after_state[:4]:
        raise RuntimeError(
            f"File changed while its immutable identity was built: {path}"
        )

    identity = {"size_bytes": int(after_state[2]), "sha256": digest}

    if reusable:
        _CONTENT_IDENTITIES[key] = identity

    return dict(identity)


def _file_identity(path: Path) -> dict[str, Any]:
    """Return canonical path and content identity for one file."""
    return {"path": str(path), **file_content_identity(path)}


def _canonical_file(value: str | Path, *, context: str) -> Path:
    """Resolve and validate one required regular file."""
    path = Path(value).expanduser().resolve(strict=False)

    if not path.is_file():
        raise FileNotFoundError(f"{context} not found: {path}")

    return path.resolve(strict=True)


def _json_value(value: Any) -> Any:
    """Convert configuration values to canonical JSON-compatible data."""
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    raise TypeError(
        "Preparation configuration contains a non-serializable value of "
        f"type {type(value).__name__!r}."
    )


def _canonical_json(value: Any) -> bytes:
    """Serialize JSON-compatible data deterministically."""
    return json.dumps(
        _json_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    """Return the SHA-256 digest of JSON-compatible data in canonical form.

    Mappings are sorted by key and paths written as text, so equal settings
    give equal digests whatever the order in which they were written.
    """
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    """Fingerprint array shape, dtype, and canonical contiguous bytes."""
    contiguous = np.ascontiguousarray(np.asarray(array))
    descriptor = _canonical_json(
        {"shape": list(contiguous.shape), "dtype": contiguous.dtype.str}
    )
    digest = hashlib.sha256()
    digest.update(descriptor)
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _header_sha256(observation: Observation) -> str | None:
    """Fingerprint the primary FITS header loaded for preparation."""
    header = observation.image.header

    if header is None:
        return None

    serialized = header.tostring(sep="\n", endcard=False, padding=False).encode(
        "ascii", errors="strict"
    )

    return hashlib.sha256(serialized).hexdigest()


def _wcs_sha256(observation: Observation) -> str | None:
    """Fingerprint the normalized celestial WCS used by BayMorph."""
    wcs = observation.metadata.wcs

    if wcs is None:
        return None

    serialized = (
        wcs.to_header(relax=True)
        .tostring(sep="\n", endcard=False, padding=False)
        .encode("ascii", errors="strict")
    )

    return hashlib.sha256(serialized).hexdigest()


def _package_generator_identity() -> dict[str, Any]:
    """Fingerprint the exact BayMorph Python sources generating products."""
    package_root = Path(__file__).resolve().parents[1]
    source_manifest: dict[str, dict[str, Any]] = {}

    for source_path in sorted(package_root.rglob("*.py")):
        relative_path = source_path.relative_to(package_root).as_posix()
        source_manifest[relative_path] = file_content_identity(source_path)

    manifest_sha256 = hashlib.sha256(_canonical_json(source_manifest)).hexdigest()

    try:
        package_version = metadata.version("baymorph")
    except metadata.PackageNotFoundError:
        package_version = "uninstalled"

    return {
        "package": "baymorph",
        "package_version": package_version,
        "python_source_count": len(source_manifest),
        "python_source_manifest_sha256": manifest_sha256,
    }


def _mask_identity(data_config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the identity of the optional curated mask."""
    configured_path = data_config.get("mask_path") or data_config.get(
        "external_mask_path"
    )

    if configured_path is None:
        return None

    path = _canonical_file(configured_path, context="Configured image-preparation mask")

    return _file_identity(path)


def _preparation_config_identity(
    config: Mapping[str, Any],
    *,
    image_path: Path,
    mask_identity: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    """Return canonical configuration fields affecting preparation."""
    data_config = config["data"]
    data_identity = {}

    # The HDU enters the identity only when it is declared, so that products
    # of images read from the primary HDU keep their identity (AD-049).
    if data_config.get("hdu") is not None:
        data_identity["hdu"] = data_config["hdu"]

    # The contaminant rule enters the identity only when it is not the rule
    # that masks every segment, so that products prepared before the rule
    # existed keep their identity when that rule is declared.
    preparation = copy.deepcopy(dict(config.get("preparation", {})))
    masking = preparation.get("masking")
    if (
        isinstance(masking, Mapping)
        and masking.get("contaminant_rule") == "all_segments"
    ):
        preparation["masking"] = {
            key: value for key, value in masking.items() if key != "contaminant_rule"
        }

    # The growth of the contaminant masks and the sky-correlation estimator
    # enter the identity only when they differ from the behaviour that
    # preceded them, no growth and the estimator of AD-047, so that products
    # prepared before AD-051 keep their identity.
    for section, key, inert in (
        ("masking", "contaminant_growth_pixels", 0),
        ("noise", "sky_correlation", "autocovariance"),
    ):
        entries = preparation.get(section)
        if (
            isinstance(entries, Mapping)
            and key in entries
            and entries[key] == inert
            and not isinstance(entries[key], bool)
        ):
            preparation[section] = {
                name: value for name, value in entries.items() if name != key
            }

    canonical = _json_value(
        {
            "object_name": config["object"]["name"],
            "data": {
                **data_identity,
                "image_path": str(image_path),
                "mask_path": (None if mask_identity is None else mask_identity["path"]),
                "mask_mode": str(data_config.get("mask_mode", "union")).strip().lower(),
                "pixel_scale": data_config.get("pixel_scale"),
                "target_position": data_config.get("target_position"),
                "cutout": data_config.get("cutout"),
            },
            "preparation": preparation,
        }
    )

    digest = hashlib.sha256(_canonical_json(canonical)).hexdigest()

    return canonical, digest


def build_prepared_input_identity(
    config: Mapping[str, Any], *, observation: Observation | None = None
) -> dict[str, Any]:
    """Build the immutable identity required to reuse prepared products."""
    input_data = config.get("data")

    if isinstance(input_data, Mapping):
        input_mask_path = input_data.get("mask_path")
        input_external_mask_path = input_data.get("external_mask_path")

        if (
            input_mask_path is not None
            and input_external_mask_path is not None
            and Path(input_mask_path).expanduser().resolve(strict=False)
            != Path(input_external_mask_path).expanduser().resolve(strict=False)
        ):
            raise ValueError(
                "data.mask_path and data.external_mask_path identify "
                "different curated masks."
            )

    resolved = resolve_config(dict(config))
    image_path = _canonical_file(
        resolved["data"]["image_path"], context="Configured image-preparation FITS"
    )

    file_identity_before = _file_identity(image_path)
    disk_observation = open_observation(image_path, hdu=resolved["data"].get("hdu"))
    file_identity_after = _file_identity(image_path)

    if file_identity_before != file_identity_after:
        raise RuntimeError(
            "Configured FITS changed while it was being acquired for "
            f"identity validation: {image_path}"
        )

    if observation is None:
        observation = disk_observation
    else:
        if observation.image.path is None:
            raise ValueError("A supplied Observation must retain its source FITS path.")

        observation_path = (
            Path(observation.image.path).expanduser().resolve(strict=False)
        )

        if observation_path != image_path:
            raise ValueError(
                "The supplied Observation does not identify the configured "
                f"FITS input: {observation_path} != {image_path}."
            )

        loaded_data_sha256 = _array_sha256(
            np.asarray(observation.image.data, dtype=float)
        )
        disk_data_sha256 = _array_sha256(
            np.asarray(disk_observation.image.data, dtype=float)
        )

        if loaded_data_sha256 != disk_data_sha256:
            raise RuntimeError(
                "The supplied Observation array does not match the current "
                "configured FITS bytes."
            )

        if _header_sha256(observation) != _header_sha256(disk_observation):
            raise RuntimeError(
                "The supplied Observation header does not match the current "
                "configured FITS header."
            )

        if _wcs_sha256(observation) != _wcs_sha256(disk_observation):
            raise RuntimeError(
                "The supplied Observation WCS does not match the current "
                "configured FITS WCS."
            )

    runtime_config = resolve_runtime_config(resolved, observation)
    mask_identity = _mask_identity(runtime_config["data"])
    preparation_config, preparation_config_sha256 = _preparation_config_identity(
        runtime_config, image_path=image_path, mask_identity=mask_identity
    )

    fits_identity = dict(file_identity_after)
    fits_identity.update(
        {
            "loaded_data_sha256": _array_sha256(
                np.asarray(observation.image.data, dtype=float)
            ),
            "primary_header_sha256": _header_sha256(observation),
            "celestial_wcs_sha256": _wcs_sha256(observation),
        }
    )

    if observation.image.hdu_index != 0:
        fits_identity["hdu"] = {
            "index": observation.image.hdu_index,
            "name": observation.image.hdu_name,
        }

    return {
        "schema_version": PREPARED_INPUT_IDENTITY_VERSION,
        "generator": _package_generator_identity(),
        "fits": fits_identity,
        "curated_mask": mask_identity,
        "preparation_config": preparation_config,
        "preparation_config_sha256": preparation_config_sha256,
    }


def validate_prepared_input_identity(
    saved_identity: Any,
    config: Mapping[str, Any],
    *,
    observation: Observation | None = None,
) -> None:
    """Fail closed unless immutable preparation inputs still match.

    The saved generator identity records which BayMorph sources produced the
    preparation bundle.  It is provenance, not part of the scientific input
    identity consumed by later inference code.  Fit-only source changes may
    therefore use a self-consistent saved bundle without rewriting it.
    """
    if not isinstance(saved_identity, Mapping):
        raise RuntimeError(
            "Saved image-preparation products have no immutable input "
            "identity. Re-run image preparation before inference."
        )

    saved_generator = saved_identity.get("generator")

    if not isinstance(saved_generator, Mapping):
        raise RuntimeError(
            "Saved image-preparation products have no generator provenance. "
            "Re-run image preparation in a new output namespace."
        )

    required_generator_fields = {
        "package",
        "package_version",
        "python_source_count",
        "python_source_manifest_sha256",
    }

    if not required_generator_fields.issubset(saved_generator):
        raise RuntimeError(
            "Saved image-preparation generator provenance is incomplete. "
            "Re-run image preparation in a new output namespace."
        )

    current_identity = build_prepared_input_identity(config, observation=observation)

    if dict(saved_identity) == current_identity:
        return

    mismatches = [
        key
        for key in (
            "schema_version",
            "fits",
            "curated_mask",
            "preparation_config_sha256",
        )
        if saved_identity.get(key) != current_identity.get(key)
    ]

    if not mismatches:
        return

    detail = ", ".join(mismatches)

    raise RuntimeError(
        "Saved image-preparation products do not match the current immutable "
        f"inputs ({detail}). Re-run image preparation in a new output "
        "namespace; stale prepared products cannot be used for inference."
    )


def validate_generator_identity(
    saved: Any,
    current: Mapping[str, Any],
    *,
    provenance_fields: Collection[str],
    product: str,
) -> None:
    """Reject a derived product made by other computing code or runtime.

    A generator identity records what made a persisted product of the
    preparation. Its fields other than ``provenance_fields`` fix the numbers
    of the product, such as the source of the module that computes them and
    the versions of NumPy and SciPy, and must equal those of the current
    runtime. The provenance fields, such as the version label of BayMorph and
    the source of the module that persists the product, are recorded but not
    compared: a release whose computing code is unchanged makes the same
    product. The saved identity must hold exactly the fields of the current
    one.

    Parameters
    ----------
    saved : Any
        Generator identity recorded in the persisted product.
    current : mapping
        Generator identity of the current implementation and runtime.
    provenance_fields : collection of str
        Fields of ``current`` that are recorded but not compared.
    product : str
        Name of the product in error messages.

    Raises
    ------
    ValueError
        If the saved identity is not a mapping, does not hold exactly the
        fields of the current identity, or differs from it outside the
        provenance fields.
    """
    unknown = sorted(set(provenance_fields) - set(current))

    if unknown:
        raise ValueError(
            f"{product}: provenance fields {unknown} are not fields of its "
            "generator identity."
        )

    if not isinstance(saved, Mapping) or set(saved) != set(current):
        raise ValueError(
            f"{product} has an incomplete or unknown generator identity. "
            "Build it in a new output namespace."
        )

    changed = sorted(
        key
        for key in current
        if key not in provenance_fields and saved[key] != current[key]
    )

    if changed:
        raise ValueError(
            f"{product} was made by another implementation or numerical "
            f"runtime (different {', '.join(changed)}). Build it in a new "
            "output namespace."
        )


__all__ = [
    "PREPARED_INPUT_IDENTITY_VERSION",
    "build_prepared_input_identity",
    "canonical_json_sha256",
    "file_content_identity",
    "validate_generator_identity",
    "validate_prepared_input_identity",
]
