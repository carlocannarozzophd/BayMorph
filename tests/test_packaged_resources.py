"""Tests for canonical BayMorph packaged resources."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPOSITORY_CONFIG_DIRECTORY = (
    PROJECT_ROOT
    / "configs"
)

PACKAGED_CONFIG_DIRECTORY = (
    PROJECT_ROOT
    / "baymorph"
    / "resources"
    / "configs"
)

PACKAGED_NOTEBOOK_DIRECTORY = (
    PROJECT_ROOT
    / "baymorph"
    / "resources"
    / "notebooks"
)


def _relative_yaml_files(
    directory: Path,
) -> list[Path]:
    """Return sorted top-level YAML files."""
    return sorted(
        path.relative_to(directory)
        for path in directory.glob("*.yaml")
        if path.is_file()
    )


def test_packaged_config_directories_exist() -> None:
    """Repository and packaged configuration directories must exist."""
    assert REPOSITORY_CONFIG_DIRECTORY.is_dir()
    assert PACKAGED_CONFIG_DIRECTORY.is_dir()


def test_packaged_config_file_lists_match() -> None:
    """Packaged configurations must match public repository configurations."""
    assert _relative_yaml_files(
        REPOSITORY_CONFIG_DIRECTORY
    ) == _relative_yaml_files(
        PACKAGED_CONFIG_DIRECTORY
    )


def test_packaged_config_contents_match() -> None:
    """Packaged configurations must be byte-identical to public configs."""
    for relative_path in _relative_yaml_files(
        REPOSITORY_CONFIG_DIRECTORY
    ):
        repository_file = (
            REPOSITORY_CONFIG_DIRECTORY
            / relative_path
        )
        packaged_file = (
            PACKAGED_CONFIG_DIRECTORY
            / relative_path
        )

        assert (
            repository_file.read_bytes()
            == packaged_file.read_bytes()
        )


def test_notebooks_are_not_duplicated_inside_package() -> None:
    """Public notebooks must have one canonical repository location."""
    assert not PACKAGED_NOTEBOOK_DIRECTORY.exists()
