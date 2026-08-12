"""Tests for the BayMorph command-line interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy.io import fits

import baymorph
from baymorph.cli import main


def _write_fits(
    path: Path,
) -> None:
    """Write a minimal FITS image suitable for CLI configuration tests."""
    data = np.zeros(
        (
            21,
            31,
        ),
        dtype=float,
    )

    header = fits.Header()

    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRPIX1"] = 16.0
    header["CRPIX2"] = 11.0
    header["CRVAL1"] = 150.0
    header["CRVAL2"] = 2.0
    header["CDELT1"] = -0.2 / 3600.0
    header["CDELT2"] = 0.2 / 3600.0

    fits.writeto(
        path,
        data,
        header,
    )


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

    with pytest.raises(
        SystemExit,
    ):
        main()


def test_config_command_writes_yaml_with_explicit_center(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The config command supports an explicit detector-centre seed."""
    fits_path = tmp_path / "galaxy.fits"
    output_path = tmp_path / "galaxy.yaml"

    _write_fits(
        fits_path
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "baymorph",
            "config",
            str(fits_path),
            "--target-center",
            "--output",
            str(output_path),
        ],
    )

    assert main() == 0

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        config = yaml.safe_load(
            stream
        )

    assert config[
        "data"
    ][
        "target_position"
    ] == "center"

    assert config[
        "data"
    ][
        "image_path"
    ] == str(
        fits_path
    )


def test_config_command_accepts_explicit_pixel_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The config command accepts explicit target x and y coordinates."""
    fits_path = tmp_path / "galaxy.fits"
    output_path = tmp_path / "galaxy.yaml"

    _write_fits(
        fits_path
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "baymorph",
            "config",
            str(fits_path),
            "--target-x",
            "14.5",
            "--target-y",
            "9.5",
            "--output",
            str(output_path),
        ],
    )

    assert main() == 0

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        config = yaml.safe_load(
            stream
        )

    assert config[
        "data"
    ][
        "target_position"
    ] == {
        "x": 14.5,
        "y": 9.5,
    }


def test_config_command_requires_complete_target_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI must never infer an unspecified target position."""
    fits_path = tmp_path / "galaxy.fits"

    _write_fits(
        fits_path
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "baymorph",
            "config",
            str(fits_path),
            "--target-x",
            "14.5",
        ],
    )

    with pytest.raises(
        SystemExit,
    ):
        main()


def test_config_command_rejects_conflicting_target_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detector-centre and explicit-coordinate seeds are exclusive."""
    fits_path = tmp_path / "galaxy.fits"

    _write_fits(
        fits_path
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "baymorph",
            "config",
            str(fits_path),
            "--target-center",
            "--target-x",
            "14.5",
            "--target-y",
            "9.5",
        ],
    )

    with pytest.raises(
        SystemExit,
    ):
        main()


def test_make_config_from_fits_is_top_level_public_api() -> None:
    """The FITS configuration maker is exposed by the stable public API."""
    assert callable(
        baymorph.make_config_from_fits
    )
