"""Identities of preparation products, computed from content, never from paths.

An identity is the SHA-256 digest of a canonical JSON document holding the
name and version of a stage, its settings, the identities of the products it
consumes and the content identities of the files it reads. A path never
enters an identity: moving or copying a project leaves every identity
unchanged, while changing one byte of an input changes it. A path is recorded
beside an identity, relative to the project folder, as information only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

IDENTITY_SCHEMA_VERSION = 1
"""Version of the document whose digest is a stage identity."""

_HASH_BLOCK_BYTES = 1024 * 1024
_DIGEST = re.compile(r"[0-9a-f]{64}")
_STAGE_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")

# Digests of files already hashed in this process, keyed by the resolved path
# and the state of the file (device, inode, size, modification time).
_FILE_DIGESTS: dict[tuple[str, int, int, int, int], str] = {}


def canonical_value(value: Any, *, where: str = "value") -> Any:
    """Return ``value`` as canonical JSON-compatible data.

    Mappings must have string keys and are returned with sorted keys; lists
    and tuples become lists; NumPy scalars become Python scalars; ``-0.0``
    becomes ``0.0``. Everything else is refused, so that the same settings
    always give the same identity.

    Parameters
    ----------
    value : Any
        Settings, records or other JSON-like data.
    where : str, default="value"
        Name of ``value``, used in error messages.

    Returns
    -------
    Any
        Canonical JSON-compatible data.

    Raises
    ------
    TypeError
        If ``value`` holds a path, an array, a set, bytes, a non-string key
        or any other type that JSON cannot hold.
    ValueError
        If ``value`` holds a number that is not finite.
    """
    if isinstance(value, Mapping):
        canonical: dict[str, Any] = {}

        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"{where} has a key of type {type(key).__name__}; the keys "
                    "of an identity are strings."
                )

            canonical[key] = canonical_value(item, where=f"{where}.{key}")

        return {key: canonical[key] for key in sorted(canonical)}

    if isinstance(value, os.PathLike):
        raise TypeError(
            f"{where} is a path ({os.fspath(value)!s}). A path never enters an "
            "identity: pass the content identity of the file instead."
        )

    if value is None or isinstance(value, (str, bool)):
        return value

    if isinstance(value, np.ndarray):
        raise TypeError(
            f"{where} is an array; an array enters an identity through "
            "array_sha256 or through the identity of the product that holds it."
        )

    if isinstance(value, np.generic):
        return canonical_value(value.item(), where=where)

    if isinstance(value, int):
        return int(value)

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"{where} is not finite ({value!r}); an identity holds finite "
                "numbers only."
            )

        return 0.0 if value == 0.0 else float(value)

    if isinstance(value, (list, tuple)):
        return [
            canonical_value(item, where=f"{where}[{index}]")
            for index, item in enumerate(value)
        ]

    raise TypeError(
        f"{where} has type {type(value).__name__}, which an identity cannot hold."
    )


def canonical_json(value: Any) -> bytes:
    """Serialise ``value`` deterministically as compact UTF-8 JSON.

    Parameters
    ----------
    value : Any
        JSON-like data accepted by :func:`canonical_value`.

    Returns
    -------
    bytes
        The canonical serialisation.
    """
    return json.dumps(
        canonical_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    """Return the hexadecimal SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def is_digest(value: Any) -> bool:
    """Return whether ``value`` is a lower-case hexadecimal SHA-256 digest."""
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _regular_file_state(path: Path) -> tuple[int, int, int, int]:
    """Return the device, inode, size and modification time of a file."""
    state = path.stat()

    if not stat.S_ISREG(state.st_mode):
        raise FileNotFoundError(f"Not a regular file: {path}")

    return (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns)


def _stream_sha256(path: Path) -> str:
    """Return the SHA-256 digest of the bytes of one file."""
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_HASH_BLOCK_BYTES), b""):
            digest.update(block)

    return digest.hexdigest()


def file_sha256(path: str | os.PathLike[str], *, reuse: bool = True) -> str:
    """Return the SHA-256 digest of the content of one regular file.

    A file is hashed once per process while its device, inode, size and
    modification time are unchanged; a file that changes while it is hashed
    is refused.

    Parameters
    ----------
    path : str or os.PathLike
        File to hash.
    reuse : bool, default=True
        Reuse the digest computed earlier in this process for the same
        unchanged file. Integrity checks pass False, so that every byte is
        read again.

    Returns
    -------
    str
        The hexadecimal digest.

    Raises
    ------
    FileNotFoundError
        If ``path`` is not a regular file.
    RuntimeError
        If the file changes while it is hashed.
    """
    resolved = Path(path).expanduser().resolve(strict=True)
    before = _regular_file_state(resolved)
    key = (str(resolved), *before)

    if reuse and key in _FILE_DIGESTS:
        return _FILE_DIGESTS[key]

    digest = _stream_sha256(resolved)

    if _regular_file_state(resolved) != before:
        raise RuntimeError(f"File changed while its content was hashed: {resolved}")

    _FILE_DIGESTS[key] = digest

    return digest


def forget_file_digests() -> None:
    """Forget the file digests computed in this process."""
    _FILE_DIGESTS.clear()


@dataclass(frozen=True, slots=True)
class FileIdentity:
    """Content identity of one input file, with its location as information.

    Parameters
    ----------
    sha256 : str
        SHA-256 digest of the content of the file.
    size_bytes : int
        Size of the file in bytes.
    path : str
        POSIX path of the file relative to the project folder, or its
        absolute path when it lies outside the project. Information only.
    inside_project : bool
        Whether the file lies inside the project folder.
    """

    sha256: str
    size_bytes: int
    path: str
    inside_project: bool

    def __post_init__(self) -> None:
        if not is_digest(self.sha256):
            raise ValueError("FileIdentity.sha256 must be a SHA-256 digest.")

        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise TypeError("FileIdentity.size_bytes must be an integer.")

        if self.size_bytes < 0:
            raise ValueError("FileIdentity.size_bytes must not be negative.")

    def payload(self) -> dict[str, Any]:
        """Return the part of this identity that enters a product identity."""
        return {"sha256": self.sha256, "size_bytes": self.size_bytes}

    def as_record(self) -> dict[str, Any]:
        """Return this identity with its location, for a product record."""
        return {
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "path": self.path,
            "inside_project": self.inside_project,
        }


def file_identity(
    path: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str],
) -> FileIdentity:
    """Return the content identity of one input file.

    Parameters
    ----------
    path : str or os.PathLike
        Input file. A relative path is resolved against the current
        directory, as the caller declared it.
    project_root : str or os.PathLike
        Project folder against which the location is recorded.

    Returns
    -------
    FileIdentity
        Digest and size of the content, with the location as information.

    Raises
    ------
    FileNotFoundError
        If ``path`` is not a regular file or ``project_root`` does not exist.
    """
    resolved = Path(path).expanduser().resolve(strict=True)
    root = Path(project_root).expanduser().resolve(strict=True)

    if not root.is_dir():
        raise FileNotFoundError(f"Project folder not found: {root}")

    digest = file_sha256(resolved)
    size = _regular_file_state(resolved)[2]

    try:
        location = resolved.relative_to(root).as_posix()
        inside = True
    except ValueError:
        location = resolved.as_posix()
        inside = False

    return FileIdentity(
        sha256=digest, size_bytes=size, path=location, inside_project=inside
    )


def array_sha256(array: ArrayLike) -> str:
    """Return a digest of the values, dtype and shape of one array.

    The byte order is normalised, so equal values give equal digests whether
    they come from a FITS file (big-endian) or from memory.

    Parameters
    ----------
    array : array_like
        Numerical or Boolean array.

    Returns
    -------
    str
        The hexadecimal digest.

    Raises
    ------
    TypeError
        If the array holds Python objects.
    """
    values = np.asarray(array)

    if values.dtype.hasobject:
        raise TypeError("array_sha256 does not accept object arrays.")

    little = np.ascontiguousarray(values.astype(values.dtype.newbyteorder("<")))
    descriptor = canonical_json(
        {"dtype": little.dtype.str, "shape": list(little.shape)}
    )
    digest = hashlib.sha256()
    digest.update(descriptor)
    digest.update(b"\0")
    digest.update(little.tobytes(order="C"))

    return digest.hexdigest()


def validate_stage_name(stage: str) -> str:
    """Return ``stage`` if it is a valid stage name, else raise ValueError."""
    if not isinstance(stage, str) or _STAGE_NAME.fullmatch(stage) is None:
        raise ValueError(
            f"Invalid stage name {stage!r}: use lower-case letters, digits and "
            "underscores, starting with a letter, at most 64 characters."
        )

    return stage


def stage_identity(
    *,
    stage: str,
    version: int,
    settings: Mapping[str, Any],
    upstream: Mapping[str, str] | None = None,
    inputs: Mapping[str, FileIdentity] | None = None,
) -> str:
    """Return the identity of the product of one stage.

    Parameters
    ----------
    stage : str
        Name of the stage.
    version : int
        Version of the rules of the stage; raising it changes every identity
        of the stage.
    settings : Mapping
        Settings of the stage, as resolved from the configuration.
    upstream : Mapping of str to str, optional
        Identities of the products the stage consumes, by role.
    inputs : Mapping of str to FileIdentity, optional
        Content identities of the files the stage reads, by role.

    Returns
    -------
    str
        The hexadecimal identity.

    Raises
    ------
    ValueError
        If the stage name, the version or an upstream identity is invalid.
    TypeError
        If the settings hold a value an identity cannot hold, or an input is
        not a FileIdentity.
    """
    validate_stage_name(stage)

    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("A stage version is a positive integer.")

    upstream = dict(upstream or {})

    for role, identity in upstream.items():
        if not is_digest(identity):
            raise ValueError(
                f"Upstream identity {role!r} of stage {stage!r} is not a SHA-256 "
                "digest."
            )

    inputs = dict(inputs or {})

    for role, identity in inputs.items():
        if not isinstance(identity, FileIdentity):
            raise TypeError(
                f"Input {role!r} of stage {stage!r} must be a FileIdentity, not "
                f"{type(identity).__name__}."
            )

    document = {
        "identity_schema": IDENTITY_SCHEMA_VERSION,
        "stage": stage,
        "version": version,
        "settings": canonical_value(settings, where=f"{stage}.settings"),
        "upstream": upstream,
        "inputs": {role: identity.payload() for role, identity in inputs.items()},
    }

    return sha256_hex(canonical_json(document))
