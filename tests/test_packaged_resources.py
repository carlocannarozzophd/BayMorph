"""Tests ensuring packaged resources match repository examples."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESOURCE_PAIRS = (
    (
        PROJECT_ROOT / "configs",
        PROJECT_ROOT / "baymorph" / "resources" / "configs",
    ),
    (
        PROJECT_ROOT / "notebooks",
        PROJECT_ROOT / "baymorph" / "resources" / "notebooks",
    ),
)


def _relative_files(directory: Path) -> list[Path]:
    """Return sorted file paths relative to a directory."""
    return sorted(
        path.relative_to(directory)
        for path in directory.rglob("*")
        if path.is_file()
    )


def test_packaged_tutorial_uses_supported_imports() -> None:
    """Ensure the packaged tutorial uses supported public interfaces."""
    notebook_path = (
        files("baymorph.resources")
        / "notebooks"
        / "01_learn_baymorph.ipynb"
    )

    with notebook_path.open("r", encoding="utf-8") as stream:
        notebook = json.load(stream)

    source = "\n".join(
        line
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        for line in cell.get("source", [])
    )

    assert "from baymorph import BayMorphRunner" in source
    assert (
        "from baymorph.pipeline import run_simulation_from_config"
        in source
    )

    assert "load_resolved_config" not in source
    assert "observation.data" not in source
    assert "observation.model_image" not in source
    assert "matplotlib" not in source
    assert "numpy" not in source
    

def test_packaged_resource_directories_exist() -> None:
    """Repository and packaged resource directories must exist."""
    for repository_directory, packaged_directory in RESOURCE_PAIRS:
        assert repository_directory.is_dir()
        assert packaged_directory.is_dir()


def test_packaged_resource_file_lists_match() -> None:
    """Packaged resources must contain the same files as the repository."""
    for repository_directory, packaged_directory in RESOURCE_PAIRS:
        assert _relative_files(repository_directory) == _relative_files(
            packaged_directory
        )


def test_packaged_resource_contents_match() -> None:
    """Packaged resources must be byte-identical to repository examples."""
    for repository_directory, packaged_directory in RESOURCE_PAIRS:
        for relative_path in _relative_files(repository_directory):
            repository_file = repository_directory / relative_path
            packaged_file = packaged_directory / relative_path

            assert repository_file.read_bytes() == packaged_file.read_bytes()