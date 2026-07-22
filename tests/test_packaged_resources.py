"""Tests ensuring packaged resources match repository examples."""

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


def test_packaged_resource_directories_exist():
    """Repository and packaged resource directories must exist."""
    for repository_directory, packaged_directory in RESOURCE_PAIRS:
        assert repository_directory.is_dir()
        assert packaged_directory.is_dir()


def test_packaged_resource_file_lists_match():
    """Packaged resources must contain the same files as the repository."""
    for repository_directory, packaged_directory in RESOURCE_PAIRS:
        assert _relative_files(
            repository_directory
        ) == _relative_files(
            packaged_directory
        )


def test_packaged_resource_contents_match():
    """Packaged resources must be byte-identical to repository examples."""
    for repository_directory, packaged_directory in RESOURCE_PAIRS:
        for relative_path in _relative_files(repository_directory):
            repository_file = repository_directory / relative_path
            packaged_file = packaged_directory / relative_path

            assert repository_file.read_bytes() == packaged_file.read_bytes()