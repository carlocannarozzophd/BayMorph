"""Tests for the BayMorph command-line interface."""

from __future__ import annotations

from pathlib import Path

import pytest

from baymorph.cli import main


def test_examples_command_copies_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The examples command copies official notebooks and configurations."""
    destination = tmp_path / "examples"

    monkeypatch.setattr(
        "sys.argv",
        [
            "baymorph",
            "examples",
            str(destination),
        ],
    )

    assert main() == 0

    assert (
        destination
        / "configs"
        / "simulated_galaxy.yaml"
    ).is_file()

    assert (
        destination
        / "notebooks"
        / "01_Simulated_SingleSersic_SingleBand_Fit.ipynb"
    ).is_file()


def test_init_command_creates_project_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The init command creates a complete BayMorph project."""
    project = tmp_path / "project"

    monkeypatch.setattr(
        "sys.argv",
        [
            "baymorph",
            "init",
            str(project),
        ],
    )

    assert main() == 0

    assert (project / "configs").is_dir()
    assert (project / "notebooks").is_dir()
    assert (project / "data").is_dir()
    assert (project / "runs").is_dir()

    assert (
        project
        / "configs"
        / "simulated_galaxy.yaml"
    ).is_file()

    assert (
        project
        / "notebooks"
        / "01_Simulated_SingleSersic_SingleBand_Fit.ipynb"
    ).is_file()


def test_init_rejects_nonempty_directory_without_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The init command refuses to overwrite a non-empty directory."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "existing.txt").write_text(
        "existing",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "baymorph",
            "init",
            str(project),
        ],
    )

    with pytest.raises(SystemExit):
        main()