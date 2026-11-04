"""One store for the products of every preparation stage.

Each stage writes one product into its own directory under the root of the
store, ``<namespace>/preparation/<stage>/``:

- one ``<name>.npy`` file per array and one ``<name>.fits`` file per image
  that a user may open;
- ``record.json``, the record of what the stage did and why, with the stage
  name, version, identity and generator;
- ``MANIFEST.json``, the size and SHA-256 digest of every file above;
- ``COMPLETE.json``, written last, with the digest of the manifest.

A product is written into a hidden staging directory and published by one
rename, so a reader sees either no product or a complete one. Loading checks
the marker, the manifest and every byte of every file before returning
anything. The files of a product carry no timestamp, so the same product
gives the same bytes in any namespace and process; only ``COMPLETE.json``
records when it was completed.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from astropy.io import fits

from baymorph.preparation.identity import (
    canonical_value,
    file_sha256,
    is_digest,
    validate_stage_name,
)

STORE_SCHEMA_VERSION = 1
"""Version of the layout of a product directory."""

RECORD_NAME = "record.json"
MANIFEST_NAME = "MANIFEST.json"
COMPLETE_NAME = "COMPLETE.json"

_RESERVED_NAMES = frozenset({"record", "manifest", "complete"})
_IMAGE_DTYPES = frozenset({"uint8", "int16", "int32", "int64", "float32", "float64"})


class ProductError(RuntimeError):
    """A product of the store is missing, incomplete, corrupt or stale."""


class ProductExistsError(ProductError):
    """A product of the stage already exists and overwriting was not asked."""


class IncompleteProductError(ProductError):
    """A product directory exists without its completion marker."""


class CorruptProductError(ProductError):
    """A product fails its integrity checks."""


class StaleProductError(ProductError):
    """A complete product was built from other inputs or settings."""


class ProductStatus(str, Enum):
    """State of the product of one stage with respect to an identity."""

    MISSING = "missing"
    INCOMPLETE = "incomplete"
    STALE = "stale"
    COMPLETE = "complete"


def _frozen_array(value: Any, *, where: str) -> np.ndarray:
    """Return a read-only copy of a numerical or Boolean array."""
    array = np.array(value, copy=True)

    if array.dtype.hasobject:
        raise TypeError(f"{where} holds Python objects; store numbers only.")

    if array.dtype.kind not in "biuf":
        raise TypeError(
            f"{where} has dtype {array.dtype}; the store holds Boolean, integer "
            "and floating-point arrays."
        )

    array.setflags(write=False)

    return array


def _validate_member_name(name: Any, *, where: str) -> str:
    """Return ``name`` if it can name a file of a product."""
    validate_stage_name(name)

    if name in _RESERVED_NAMES:
        raise ValueError(f"{where} name {name!r} is reserved by the store.")

    return name


@dataclass(frozen=True, eq=False)
class StageProduct:
    """The immutable product of one preparation stage.

    Products are not compared with ``==``: compare their identities.

    Parameters
    ----------
    stage : str
        Name of the stage.
    version : int
        Version of the rules of the stage.
    identity : str
        Identity of the product, from
        :func:`baymorph.preparation.identity.stage_identity`.
    record : Mapping
        What the stage did and why, as JSON-compatible data with finite
        numbers only.
    arrays : Mapping of str to ndarray, optional
        Arrays saved as ``.npy`` files.
    images : Mapping of str to ndarray, optional
        Images saved as FITS files that a user may open, in one of the dtypes
        FITS stores exactly: uint8, int16, int32, int64, float32, float64.
    """

    stage: str
    version: int
    identity: str
    record: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray] = field(default_factory=dict)
    images: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_stage_name(self.stage)

        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, int)
            or self.version < 1
        ):
            raise ValueError("A stage version is a positive integer.")

        if not is_digest(self.identity):
            raise ValueError("A product identity is a SHA-256 digest.")

        if not isinstance(self.record, Mapping):
            raise TypeError("The record of a product is a mapping.")

        record = canonical_value(self.record, where=f"{self.stage}.record")

        arrays = {
            _validate_member_name(name, where="Array"): _frozen_array(
                value, where=f"Array {name!r} of stage {self.stage!r}"
            )
            for name, value in dict(self.arrays).items()
        }
        images = {
            _validate_member_name(name, where="Image"): _frozen_array(
                value, where=f"Image {name!r} of stage {self.stage!r}"
            )
            for name, value in dict(self.images).items()
        }

        shared = sorted(set(arrays) & set(images))

        if shared:
            raise ValueError(
                f"Stage {self.stage!r} names both an array and an image {shared}."
            )

        for name, image in images.items():
            if image.dtype.name not in _IMAGE_DTYPES:
                raise TypeError(
                    f"Image {name!r} of stage {self.stage!r} has dtype "
                    f"{image.dtype}; FITS images of the store use one of "
                    f"{sorted(_IMAGE_DTYPES)}. Save a mask as uint8."
                )

            if image.ndim < 1:
                raise ValueError(f"Image {name!r} of stage {self.stage!r} is a scalar.")

        object.__setattr__(self, "record", MappingProxyType(record))
        object.__setattr__(self, "arrays", MappingProxyType(arrays))
        object.__setattr__(self, "images", MappingProxyType(images))


def _generator() -> dict[str, str]:
    """Return the package and version that wrote a product."""
    try:
        version = metadata.version("baymorph")
    except metadata.PackageNotFoundError:
        version = "uninstalled"

    return {"package": "baymorph", "version": version}


def _write_bytes(path: Path, payload: bytes) -> None:
    """Write a new file and flush it to the disk."""
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return the readable, deterministic serialisation of a JSON document."""
    return (
        json.dumps(
            payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n"
    ).encode("utf-8")


def _sync_directory(path: Path) -> None:
    """Flush a directory entry to the disk where the platform allows it."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return

    try:
        os.fsync(descriptor)
    except OSError:
        # Some file systems cannot flush a directory; the files themselves
        # were flushed, so this only weakens durability after a crash.
        pass
    finally:
        os.close(descriptor)


def _read_json(path: Path, *, what: str) -> dict[str, Any]:
    """Read one JSON document of a product, refusing a non-mapping."""
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, ValueError) as error:
        raise CorruptProductError(f"The {what} {path} is unreadable.") from error

    if not isinstance(payload, dict):
        raise CorruptProductError(f"The {what} {path} is not a JSON object.")

    return payload


class ProductStore:
    """Save, check and reuse the products of the preparation stages.

    Parameters
    ----------
    root : str or os.PathLike
        Folder that holds one directory per stage, normally
        ``<namespace>/preparation``. It is created on the first save.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        """Folder that holds one directory per stage."""
        return self._root

    def stage_dir(self, stage: str) -> Path:
        """Return the directory of the product of ``stage``."""
        return self._root / validate_stage_name(stage)

    def status(self, stage: str, identity: str) -> ProductStatus:
        """Return the state of the product of ``stage`` for ``identity``.

        Only the completion marker is read: a product reported complete is
        checked byte by byte when :meth:`load` reads it.

        Parameters
        ----------
        stage : str
            Name of the stage.
        identity : str
            Identity the caller expects.

        Returns
        -------
        ProductStatus
            ``missing``, ``incomplete``, ``stale`` or ``complete``.
        """
        if not is_digest(identity):
            raise ValueError("A product identity is a SHA-256 digest.")

        directory = self.stage_dir(stage)

        if not directory.exists():
            return ProductStatus.MISSING

        marker = directory / COMPLETE_NAME

        if not marker.is_file():
            return ProductStatus.INCOMPLETE

        recorded = _read_json(marker, what="completion marker").get("identity")

        if recorded == identity:
            return ProductStatus.COMPLETE

        return ProductStatus.STALE

    def save(self, product: StageProduct, *, overwrite: bool = False) -> Path:
        """Write ``product`` and publish it under its stage directory.

        Parameters
        ----------
        product : StageProduct
            Product to save.
        overwrite : bool, default=False
            Replace an existing product of the stage, as
            ``outputs.overwrite: true`` asks. Without it an existing product,
            complete or not, is never touched.

        Returns
        -------
        pathlib.Path
            The directory of the published product.

        Raises
        ------
        ProductExistsError
            If a product of the stage exists and ``overwrite`` is False.
        """
        if not isinstance(product, StageProduct):
            raise TypeError("ProductStore.save expects a StageProduct.")

        target = self.stage_dir(product.stage)

        if target.exists() and not overwrite:
            raise ProductExistsError(self._exists_message(product.stage, target))

        self._root.mkdir(parents=True, exist_ok=True)
        staging = self._root / f".{product.stage}.staging-{uuid.uuid4().hex}"
        staging.mkdir()

        try:
            self._write_files(staging, product)
            self._publish(staging, target, overwrite=overwrite)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        _sync_directory(self._root)

        return target

    def load(self, stage: str, *, identity: str | None = None) -> StageProduct:
        """Read the product of ``stage`` after checking every byte of it.

        Parameters
        ----------
        stage : str
            Name of the stage.
        identity : str, optional
            Identity the caller expects; a product with another identity is
            refused as stale.

        Returns
        -------
        StageProduct
            The product, with read-only arrays.

        Raises
        ------
        FileNotFoundError
            If no product of the stage exists.
        IncompleteProductError
            If the completion marker is missing.
        StaleProductError
            If the product has another identity than ``identity``.
        CorruptProductError
            If any check of the marker, manifest, record or files fails.
        """
        if identity is not None and not is_digest(identity):
            raise ValueError("A product identity is a SHA-256 digest.")

        directory = self.stage_dir(stage)

        if not directory.is_dir():
            raise FileNotFoundError(
                f"No product of stage {stage!r} exists in {self._root}."
            )

        marker_path = directory / COMPLETE_NAME

        if not marker_path.is_file():
            raise IncompleteProductError(
                f"The product of stage {stage!r} in {directory} has no completion "
                "marker: it was not finished. Keep it for diagnosis, then rebuild "
                "it with outputs.overwrite: true or use a new output namespace."
            )

        marker = _read_json(marker_path, what="completion marker")
        recorded_identity = marker.get("identity")

        if (
            marker.get("store_schema") != STORE_SCHEMA_VERSION
            or marker.get("stage") != stage
            or not is_digest(recorded_identity)
            or not is_digest(marker.get("manifest_sha256"))
        ):
            raise CorruptProductError(
                f"The completion marker of stage {stage!r} in {directory} does not "
                "match the layout of this store."
            )

        if identity is not None and recorded_identity != identity:
            raise StaleProductError(
                f"The product of stage {stage!r} in {directory} was built from "
                f"other inputs or settings (identity {recorded_identity[:12]}..., "
                f"expected {identity[:12]}...). Rebuild it with "
                "outputs.overwrite: true or use a new output namespace."
            )

        files = self._verify_manifest(directory, stage, marker)
        record = self._verify_record(directory, stage, marker, files)

        arrays = {
            name: self._read_array(directory / f"{name}.npy", spec, stage, name)
            for name, spec in record["arrays"].items()
        }
        images = {
            name: self._read_image(directory / f"{name}.fits", spec, stage, name)
            for name, spec in record["images"].items()
        }

        return StageProduct(
            stage=stage,
            version=record["version"],
            identity=recorded_identity,
            record=record["record"],
            arrays=arrays,
            images=images,
        )

    @staticmethod
    def _exists_message(stage: str, target: Path) -> str:
        """Explain why an existing product is not replaced."""
        if (target / COMPLETE_NAME).is_file():
            return (
                f"A product of stage {stage!r} already exists in {target}. "
                "Accepted products are reused, never rewritten in place: set "
                "outputs.overwrite: true to rebuild it, or use a new output "
                "namespace."
            )

        return (
            f"An unfinished product of stage {stage!r} exists in {target}. Keep "
            "it for diagnosis, then set outputs.overwrite: true to rebuild it, "
            "or use a new output namespace."
        )

    @staticmethod
    def _write_files(staging: Path, product: StageProduct) -> None:
        """Write the files, the record, the manifest and the marker."""
        array_specs: dict[str, dict[str, Any]] = {}
        image_specs: dict[str, dict[str, Any]] = {}

        for name, array in product.arrays.items():
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            _write_bytes(staging / f"{name}.npy", buffer.getvalue())

            array_specs[name] = {"dtype": array.dtype.name, "shape": list(array.shape)}

        for name, image in product.images.items():
            header = fits.Header()
            header["BMSTAGE"] = (product.stage, "BayMorph preparation stage")
            header["BMVERS"] = (product.version, "Version of the stage rules")
            header["BMIDENT"] = product.identity

            # A writable copy, because astropy may swap the bytes of the data
            # in place while it writes them.
            buffer = io.BytesIO()
            fits.PrimaryHDU(data=np.array(image, copy=True), header=header).writeto(
                buffer
            )
            _write_bytes(staging / f"{name}.fits", buffer.getvalue())

            image_specs[name] = {"dtype": image.dtype.name, "shape": list(image.shape)}

        record_document = {
            "store_schema": STORE_SCHEMA_VERSION,
            "stage": product.stage,
            "version": product.version,
            "identity": product.identity,
            "generator": _generator(),
            "arrays": array_specs,
            "images": image_specs,
            "record": dict(product.record),
        }
        _write_bytes(staging / RECORD_NAME, _json_bytes(record_document))

        manifest = {
            "store_schema": STORE_SCHEMA_VERSION,
            "stage": product.stage,
            "identity": product.identity,
            "files": {
                path.name: {
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path, reuse=False),
                }
                for path in sorted(staging.iterdir())
            },
        }
        manifest_bytes = _json_bytes(manifest)
        _write_bytes(staging / MANIFEST_NAME, manifest_bytes)

        marker = {
            "store_schema": STORE_SCHEMA_VERSION,
            "stage": product.stage,
            "version": product.version,
            "identity": product.identity,
            "manifest_sha256": file_sha256(staging / MANIFEST_NAME, reuse=False),
            "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        _write_bytes(staging / COMPLETE_NAME, _json_bytes(marker))
        _sync_directory(staging)

    def _publish(self, staging: Path, target: Path, *, overwrite: bool) -> None:
        """Move a finished staging directory to its stage directory."""
        retired: Path | None = None

        if target.exists():
            if not overwrite:
                raise ProductExistsError(self._exists_message(target.name, target))

            retired = self._root / f".{target.name}.retired-{uuid.uuid4().hex}"
            os.rename(target, retired)

        try:
            os.rename(staging, target)
        except OSError as error:
            if retired is not None and not target.exists():
                os.rename(retired, target)

            raise ProductExistsError(
                f"Another process published a product of stage {target.name!r} "
                f"in {target} at the same time; this product was not written."
            ) from error

        if retired is not None:
            shutil.rmtree(retired)

    @staticmethod
    def _verify_manifest(
        directory: Path, stage: str, marker: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Check the manifest and every file it lists; return its entries."""
        manifest_path = directory / MANIFEST_NAME

        if not manifest_path.is_file():
            raise CorruptProductError(
                f"The product of stage {stage!r} in {directory} has no manifest."
            )

        if file_sha256(manifest_path, reuse=False) != marker["manifest_sha256"]:
            raise CorruptProductError(
                f"The manifest of stage {stage!r} in {directory} does not match "
                "its completion marker: the product was modified after it was "
                "completed."
            )

        manifest = _read_json(manifest_path, what="manifest")
        files = manifest.get("files")

        if (
            manifest.get("store_schema") != STORE_SCHEMA_VERSION
            or manifest.get("stage") != stage
            or manifest.get("identity") != marker["identity"]
            or not isinstance(files, dict)
        ):
            raise CorruptProductError(
                f"The manifest of stage {stage!r} in {directory} does not match "
                "its completion marker."
            )

        present = {
            path.name
            for path in directory.iterdir()
            if path.name not in (MANIFEST_NAME, COMPLETE_NAME)
        }

        if present != set(files):
            unexpected = sorted(present - set(files))
            missing = sorted(set(files) - present)
            raise CorruptProductError(
                f"The files of stage {stage!r} in {directory} differ from its "
                f"manifest: unexpected {unexpected}, missing {missing}."
            )

        for name in sorted(files):
            path = directory / name
            expected = files[name]

            if (
                not path.is_file()
                or not isinstance(expected, dict)
                or expected.get("size_bytes") != path.stat().st_size
                or expected.get("sha256") != file_sha256(path, reuse=False)
            ):
                raise CorruptProductError(
                    f"File {name!r} of stage {stage!r} in {directory} fails its "
                    "integrity check: it was modified after the product was "
                    "completed."
                )

        return files

    @staticmethod
    def _verify_record(
        directory: Path,
        stage: str,
        marker: Mapping[str, Any],
        files: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Check the record against the marker and the listed files."""
        if RECORD_NAME not in files:
            raise CorruptProductError(
                f"The product of stage {stage!r} in {directory} has no record."
            )

        record = _read_json(directory / RECORD_NAME, what="record")
        arrays = record.get("arrays")
        images = record.get("images")

        if (
            record.get("store_schema") != STORE_SCHEMA_VERSION
            or record.get("stage") != stage
            or record.get("identity") != marker["identity"]
            or record.get("version") != marker.get("version")
            or not isinstance(record.get("record"), dict)
            or not isinstance(arrays, dict)
            or not isinstance(images, dict)
        ):
            raise CorruptProductError(
                f"The record of stage {stage!r} in {directory} does not match its "
                "completion marker."
            )

        expected = {RECORD_NAME}
        expected |= {f"{name}.npy" for name in arrays}
        expected |= {f"{name}.fits" for name in images}

        if expected != set(files):
            raise CorruptProductError(
                f"The record of stage {stage!r} in {directory} does not list the "
                "files of its manifest."
            )

        return record

    @staticmethod
    def _check_spec(
        array: np.ndarray, spec: Any, stage: str, name: str, *, kind: str
    ) -> None:
        """Check the dtype and shape of a loaded array against its record."""
        if (
            not isinstance(spec, dict)
            or spec.get("dtype") != array.dtype.name
            or spec.get("shape") != list(array.shape)
        ):
            raise CorruptProductError(
                f"{kind} {name!r} of stage {stage!r} does not have the dtype and "
                "shape its record states."
            )

    def _read_array(self, path: Path, spec: Any, stage: str, name: str) -> np.ndarray:
        """Read one ``.npy`` array of a product."""
        try:
            array = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise CorruptProductError(
                f"Array {name!r} of stage {stage!r} is unreadable."
            ) from error

        self._check_spec(array, spec, stage, name, kind="Array")

        return array

    def _read_image(self, path: Path, spec: Any, stage: str, name: str) -> np.ndarray:
        """Read one FITS image of a product, in native byte order."""
        try:
            with fits.open(path, memmap=False) as hdul:
                data = hdul[0].data
                image = None if data is None else np.array(data, copy=True)
        except (OSError, ValueError) as error:
            raise CorruptProductError(
                f"Image {name!r} of stage {stage!r} is unreadable."
            ) from error

        if image is None:
            raise CorruptProductError(f"Image {name!r} of stage {stage!r} is empty.")

        image = image.astype(image.dtype.newbyteorder("="), copy=False)
        self._check_spec(image, spec, stage, name, kind="Image")

        return image
