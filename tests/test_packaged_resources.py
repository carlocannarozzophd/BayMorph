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


def test_packaged_configs_have_repository_sources() -> None:
    """Every packaged configuration must have a repository source."""
    repository_configs = set(
        _relative_yaml_files(
            REPOSITORY_CONFIG_DIRECTORY
        )
    )
    packaged_configs = set(
        _relative_yaml_files(
            PACKAGED_CONFIG_DIRECTORY
        )
    )

    assert packaged_configs
    assert packaged_configs <= repository_configs


def test_packaged_config_contents_match() -> None:
    """Packaged configurations must match their repository sources."""
    for relative_path in _relative_yaml_files(
        PACKAGED_CONFIG_DIRECTORY
    ):
        repository_file = (
            REPOSITORY_CONFIG_DIRECTORY
            / relative_path
        )
        packaged_file = (
            PACKAGED_CONFIG_DIRECTORY
            / relative_path
        )

        assert repository_file.is_file()

        assert (
            repository_file.read_bytes()
            == packaged_file.read_bytes()
        )


def test_packaged_notebooks_match_repository_sources() -> None:
    """Packaged notebooks must exactly match their public sources."""
    repository_directory = (
        PROJECT_ROOT
        / "notebooks"
    )

    repository_notebooks = {
        filename.name
        for filename in repository_directory.glob(
            "*.ipynb"
        )
        if filename.is_file()
    }

    packaged_notebooks = {
        filename.name
        for filename in PACKAGED_NOTEBOOK_DIRECTORY.glob(
            "*.ipynb"
        )
        if filename.is_file()
    }

    assert packaged_notebooks == repository_notebooks

    for notebook_name in packaged_notebooks:
        assert (
            PACKAGED_NOTEBOOK_DIRECTORY
            / notebook_name
        ).read_bytes() == (
            repository_directory
            / notebook_name
        ).read_bytes()
