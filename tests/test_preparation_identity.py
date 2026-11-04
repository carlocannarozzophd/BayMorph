"""Identities of the new preparation: from content, never from paths.

A product identity must change when an input byte, a setting, an upstream
product or the version of a stage changes, and must not change when a
project or a file is moved or copied, or when the same settings are written
in another order.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from baymorph.preparation.identity import (
    FileIdentity,
    array_sha256,
    canonical_json,
    file_identity,
    file_sha256,
    forget_file_digests,
    is_digest,
    stage_identity,
)


@pytest.fixture(autouse=True)
def _fresh_digests():
    forget_file_digests()
    yield
    forget_file_digests()


def _project(root: Path, content: bytes = b"SIMPLE  =    T") -> Path:
    """Create a small project with one data file and return the project root."""
    (root / "data").mkdir(parents=True)
    (root / "data" / "image.fits").write_bytes(content)
    return root


def _identity(inputs=None, **overrides) -> str:
    arguments = {
        "stage": "cutout",
        "version": 1,
        "settings": {"mode": "auto", "threshold": 0.05},
        "upstream": {"target": "a" * 64},
        "inputs": inputs or {},
    }
    arguments.update(overrides)
    return stage_identity(**arguments)


def test_canonical_json_ignores_key_order_and_negative_zero():
    first = canonical_json({"b": 1, "a": {"y": -0.0, "x": [1, 2.5]}})
    second = canonical_json({"a": {"x": (1, 2.5), "y": 0.0}, "b": 1})

    assert first == second
    assert first == b'{"a":{"x":[1,2.5],"y":0.0},"b":1}'


def test_canonical_json_converts_numpy_scalars():
    assert canonical_json({"n": np.int64(3), "q": np.float32(0.5)}) == canonical_json(
        {"n": 3, "q": 0.5}
    )


@pytest.mark.parametrize(
    ("value", "error"),
    [
        ({"x": float("nan")}, ValueError),
        ({"x": float("inf")}, ValueError),
        ({"x": Path("data/image.fits")}, TypeError),
        ({"x": {1, 2}}, TypeError),
        ({"x": b"bytes"}, TypeError),
        ({"x": np.zeros(3)}, TypeError),
        ({1: "integer key"}, TypeError),
    ],
)
def test_canonical_json_refuses_what_an_identity_cannot_hold(value, error):
    with pytest.raises(error):
        canonical_json(value)


def test_a_path_in_settings_is_refused_with_the_remedy():
    with pytest.raises(TypeError, match="content identity"):
        _identity(settings={"mask": Path("data/mask.fits")})


def test_file_identity_records_the_path_relative_to_the_project(tmp_path):
    root = _project(tmp_path / "project")

    identity = file_identity(root / "data" / "image.fits", project_root=root)

    assert identity.path == "data/image.fits"
    assert identity.inside_project
    assert identity.size_bytes == len(b"SIMPLE  =    T")
    assert is_digest(identity.sha256)


def test_file_identity_outside_the_project_records_the_absolute_path(tmp_path):
    root = _project(tmp_path / "project")
    outside = tmp_path / "elsewhere.fits"
    outside.write_bytes(b"outside")

    identity = file_identity(outside, project_root=root)

    assert not identity.inside_project
    assert identity.path == outside.resolve().as_posix()


def test_moving_a_project_or_a_file_keeps_the_identity(tmp_path):
    root = _project(tmp_path / "first")
    original = _identity(
        {"image": file_identity(root / "data" / "image.fits", project_root=root)}
    )

    moved_root = tmp_path / "second" / "renamed_project"
    moved_root.parent.mkdir()
    shutil.move(str(root), str(moved_root))
    (moved_root / "data" / "my_data").mkdir()
    shutil.move(
        str(moved_root / "data" / "image.fits"),
        str(moved_root / "data" / "my_data" / "image.fits"),
    )
    moved_file = moved_root / "data" / "my_data" / "image.fits"

    moved = _identity({"image": file_identity(moved_file, project_root=moved_root)})

    assert moved == original


def test_one_changed_byte_changes_the_identity(tmp_path):
    first_root = _project(tmp_path / "first", b"SIMPLE  =    T")
    second_root = _project(tmp_path / "second", b"SIMPLE  =    F")

    first = _identity(
        {
            "image": file_identity(
                first_root / "data/image.fits", project_root=first_root
            )
        }
    )
    second = _identity(
        {
            "image": file_identity(
                second_root / "data/image.fits", project_root=second_root
            )
        }
    )

    assert first != second


def test_settings_upstream_and_version_enter_the_identity():
    reference = _identity()

    assert _identity(settings={"threshold": 0.05, "mode": "auto"}) == reference
    assert _identity(settings={"mode": "auto", "threshold": 0.02}) != reference
    assert _identity(upstream={"target": "b" * 64}) != reference
    assert _identity(version=2) != reference
    assert _identity(stage="sky") != reference


@pytest.mark.parametrize(
    "overrides",
    [
        {"stage": "Cutout"},
        {"stage": "cut-out"},
        {"version": 0},
        {"version": True},
        {"upstream": {"target": "not-a-digest"}},
    ],
)
def test_invalid_stage_arguments_are_refused(overrides):
    with pytest.raises(ValueError):
        _identity(**overrides)


def test_an_input_must_be_a_file_identity():
    with pytest.raises(TypeError, match="FileIdentity"):
        _identity(inputs={"image": "data/image.fits"})


def test_file_identity_validates_its_fields():
    with pytest.raises(ValueError):
        FileIdentity(sha256="abc", size_bytes=1, path="x", inside_project=True)

    with pytest.raises(ValueError):
        FileIdentity(sha256="0" * 64, size_bytes=-1, path="x", inside_project=True)


def test_file_sha256_reads_the_file_again_after_a_change(tmp_path):
    path = tmp_path / "file.bin"
    path.write_bytes(b"first")
    first = file_sha256(path)

    path.write_bytes(b"second content")

    assert file_sha256(path) != first


def test_file_sha256_refuses_a_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        file_sha256(tmp_path)


def test_array_digest_depends_on_values_dtype_and_shape_not_byte_order():
    values = np.arange(6, dtype="<f8").reshape(2, 3)

    assert array_sha256(values) == array_sha256(values.astype(">f8"))
    assert array_sha256(values) != array_sha256(values.astype("<f4"))
    assert array_sha256(values) != array_sha256(values.reshape(3, 2))

    changed = values.copy()
    changed[0, 0] = 1.0e-12

    assert array_sha256(values) != array_sha256(changed)


def test_array_digest_refuses_object_arrays():
    with pytest.raises(TypeError):
        array_sha256(np.array([1, "a"], dtype=object))
