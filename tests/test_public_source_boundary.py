"""Enforce the data-free and object-neutral public-source boundary."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEXT_SUFFIXES = {
    ".cff",
    ".ini",
    ".ipynb",
    ".md",
    ".py",
    ".rst",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

SCIENTIFIC_DATA_SUFFIXES = {
    ".csv",
    ".fit",
    ".fits",
    ".fits.gz",
    ".npy",
    ".npz",
}

GENERATED_RESULT_DIRECTORIES = {
    PurePosixPath("benchmarks/results"),
    PurePosixPath("docs/research/figures"),
}

OPERATIONAL_PREFIXES = (
    "baymorph/",
    "configs/",
    "notebooks/",
    "tests/",
)

OPERATIONAL_DOCUMENTS = {
    "README.md",
    "docs/API.md",
    "docs/architecture/MULTI_OBSERVATION_PUBLIC_WORKFLOW.md",
    "docs/architecture/PRODUCTION_WORKFLOW.md",
}

REAL_OBJECT_PATTERN = re.compile(
    r"\b(?:PGC|NGC|UGC|IC)\s*[-_]?\s*\d+\b",
    flags=re.IGNORECASE,
)


def _tracked_files() -> tuple[PurePosixPath, ...]:
    """Return repository files tracked by Git."""
    result = subprocess.run(
        [
            "git",
            "ls-files",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    return tuple(
        PurePosixPath(line)
        for line in result.stdout.splitlines()
        if line
    )


def _has_scientific_data_suffix(
    filename: PurePosixPath,
) -> bool:
    """Return whether a tracked path represents scientific data."""
    lowered = filename.as_posix().lower()

    return any(
        lowered.endswith(suffix)
        for suffix in SCIENTIFIC_DATA_SUFFIXES
    )


def _is_operational_surface(
    filename: PurePosixPath,
) -> bool:
    """Return whether a file belongs to the reusable public workflow."""
    relative_name = filename.as_posix()

    return (
        relative_name in OPERATIONAL_DOCUMENTS
        or relative_name.startswith(
            OPERATIONAL_PREFIXES
        )
    )


def test_public_source_contains_no_scientific_data() -> None:
    """The public repository must not distribute data products."""
    offenders = sorted(
        filename.as_posix()
        for filename in _tracked_files()
        if _has_scientific_data_suffix(
            filename
        )
    )

    assert offenders == []


def test_public_source_contains_no_generated_results() -> None:
    """Generated benchmark and research results must remain external."""
    offenders = sorted(
        filename.as_posix()
        for filename in _tracked_files()
        if any(
            filename == directory
            or directory in filename.parents
            for directory in GENERATED_RESULT_DIRECTORIES
        )
    )

    assert offenders == []


def test_operational_surface_is_real_object_neutral() -> None:
    """Reusable workflows must not depend on named real target galaxies."""
    offenders: set[str] = set()

    for filename in _tracked_files():
        if not _is_operational_surface(
            filename
        ):
            continue

        relative_name = filename.as_posix()

        if REAL_OBJECT_PATTERN.search(
            relative_name
        ):
            offenders.add(
                relative_name
            )

        if filename.suffix.lower() not in TEXT_SUFFIXES:
            continue

        source_path = PROJECT_ROOT / relative_name

        try:
            source = source_path.read_text(
                encoding="utf-8"
            )
        except UnicodeDecodeError:
            continue

        if REAL_OBJECT_PATTERN.search(
            source
        ):
            offenders.add(
                relative_name
            )

    assert sorted(
        offenders
    ) == []


def test_public_source_boundary_policy_is_documented() -> None:
    """The README must state the public distribution boundary explicitly."""
    readme = (
        PROJECT_ROOT
        / "README.md"
    ).read_text(
        encoding="utf-8"
    ).lower()

    normalized_readme = " ".join(
        readme.split()
    )

    assert "does not include observational data" in normalized_readme
    assert "does not include generated results" in normalized_readme
