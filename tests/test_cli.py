"""Tests for the BayMorph command-line interface."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from astropy.io import fits

import baymorph
from baymorph import ObservationAdvisor
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


def test_examples_command_copies_configurations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The examples command copies packaged example configurations."""
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

    assert (destination / "configs" / "simulated_galaxy.yaml").is_file()

    assert not (destination / "notebooks").exists()


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

    assert (project / "configs" / "simulated_galaxy.yaml").is_file()

    assert {filename.name for filename in (project / "notebooks").glob("*.ipynb")} == {
        "01_Simulated_SingleBand.ipynb",
        "02A_Observed_Assessment_and_Config.ipynb",
        "02B_Observed_Fit.ipynb",
    }


def test_init_rejects_nonempty_directory_without_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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

    assert main() == 1
    assert "Project directory is not empty" in capsys.readouterr().err


def test_a_runtime_error_prints_one_line_and_exits_1(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A failing command prints its error on one line, without the usage."""
    missing_path = tmp_path / "missing.fits"

    assert main(["config", str(missing_path), "--target-center"]) == 1

    error = capsys.readouterr().err

    assert error.startswith("baymorph: error: ")
    assert error.count("\n") == 1
    assert "usage:" not in error


def test_a_corrupt_fits_file_is_a_runtime_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A file that is not FITS fails with exit status 1, not a traceback."""
    fits_path = tmp_path / "corrupt.fits"
    fits_path.write_bytes(b"not a FITS file")

    assert main(["config", str(fits_path), "--target-center"]) == 1
    assert capsys.readouterr().err.startswith("baymorph: error: ")


def test_a_usage_error_shows_the_usage_of_its_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An invalid combination of options exits 2 with the command usage."""
    fits_path = tmp_path / "galaxy.fits"

    _write_fits(fits_path)

    with pytest.raises(SystemExit) as raised:
        main(
            [
                "config",
                str(fits_path),
                "--target-center",
                "--target-x",
                "14.5",
                "--target-y",
                "9.5",
            ]
        )

    assert raised.value.code == 2

    error = capsys.readouterr().err

    assert "usage: baymorph config" in error
    assert "--target-center cannot be combined" in error


def test_config_command_writes_yaml_with_explicit_center(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The config command supports an explicit detector-centre seed."""
    fits_path = tmp_path / "galaxy.fits"
    output_path = tmp_path / "galaxy.yaml"

    _write_fits(fits_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "baymorph",
            "config",
            str(fits_path),
            "--target-center",
            "--uniform-prior",
            "flux",
            "1",
            "1000",
            "--output",
            str(output_path),
        ],
    )

    assert main() == 0

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        config = yaml.safe_load(stream)

    assert config["data"]["target_position"] == "center"

    assert config["data"]["image_path"] == str(fits_path)

    assert config["outputs"]["overwrite"] is False


def test_config_command_overwrite_is_explicit_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The config command must not allow overwriting without --overwrite."""
    fits_path = tmp_path / "galaxy.fits"
    output_path = tmp_path / "galaxy.yaml"

    _write_fits(fits_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "baymorph",
            "config",
            str(fits_path),
            "--target-center",
            "--uniform-prior",
            "flux",
            "1",
            "1000",
            "--output",
            str(output_path),
            "--overwrite",
        ],
    )

    assert main() == 0

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        config = yaml.safe_load(stream)

    assert config["outputs"]["overwrite"] is True


def test_config_command_accepts_explicit_pixel_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The config command accepts explicit target x and y coordinates."""
    fits_path = tmp_path / "galaxy.fits"
    output_path = tmp_path / "galaxy.yaml"

    _write_fits(fits_path)

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
            "--uniform-prior",
            "flux",
            "1",
            "1000",
            "--output",
            str(output_path),
        ],
    )

    assert main() == 0

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        config = yaml.safe_load(stream)

    assert config["data"]["target_position"] == {
        "x": 14.5,
        "y": 9.5,
    }


def test_config_command_requires_complete_target_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI must never infer an unspecified target position."""
    fits_path = tmp_path / "galaxy.fits"

    _write_fits(fits_path)

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
    ) as raised:
        main()

    assert raised.value.code == 2


def test_config_command_rejects_conflicting_target_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detector-centre and explicit-coordinate seeds are exclusive."""
    fits_path = tmp_path / "galaxy.fits"

    _write_fits(fits_path)

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
    ) as raised:
        main()

    assert raised.value.code == 2


def test_make_config_from_fits_is_top_level_public_api() -> None:
    """The FITS configuration maker is exposed by the stable public API."""
    assert callable(baymorph.make_config_from_fits)


def test_config_command_exposes_model_and_sampler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI must expose the public model and sampler choices."""
    fits_path = tmp_path / "galaxy.fits"
    output_path = tmp_path / "galaxy.yaml"

    _write_fits(fits_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "baymorph",
            "config",
            str(fits_path),
            "--target-center",
            "--model",
            "double_sersic",
            "--sampler",
            "dynesty",
            "--uniform-prior",
            "flux_bulge",
            "1",
            "1000",
            "--uniform-prior",
            "flux_disk",
            "1",
            "1000",
            "--output",
            str(output_path),
        ],
    )

    assert main() == 0

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        config = yaml.safe_load(stream)

    assert config["fit"]["models"] == [
        "double_sersic",
    ]

    assert config["fit"]["sampler"] == {
        "name": "dynesty",
    }


def test_run_command_delegates_to_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI run command must delegate to BayMorphRunner."""
    config_path = tmp_path / "analysis.yaml"

    calls = {}

    class FakeRunner:
        def __init__(
            self,
            config,
            *,
            verbose=True,
        ):
            calls["config"] = config
            calls["verbose"] = verbose

        def run(
            self,
            *,
            diagnostics=False,
        ):
            calls["diagnostics"] = diagnostics

    monkeypatch.setattr(
        "baymorph.cli.BayMorphRunner",
        FakeRunner,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "baymorph",
            "run",
            str(config_path),
        ],
    )

    assert main() == 0

    assert calls["config"] == config_path
    assert calls["verbose"] is True
    assert calls["diagnostics"] is True


def test_run_command_can_disable_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI diagnostics may be disabled without changing the workflow."""
    config_path = tmp_path / "analysis.yaml"

    calls = {}

    class FakeRunner:
        def __init__(
            self,
            config,
            *,
            verbose=True,
        ):
            calls["config"] = config

        def run(
            self,
            *,
            diagnostics=False,
        ):
            calls["diagnostics"] = diagnostics

    monkeypatch.setattr(
        "baymorph.cli.BayMorphRunner",
        FakeRunner,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "baymorph",
            "run",
            str(config_path),
            "--no-diagnostics",
        ],
    )

    assert main() == 0

    assert calls["config"] == config_path
    assert calls["diagnostics"] is False


def _write_coadd(
    path: Path,
) -> None:
    """Write a sky-subtracted coadd whose header gives a zeropoint."""
    y, x = np.indices((81, 81), dtype=float)
    image = 100.0 * np.exp(
        -0.5 * (((x - 40.0) / 4.0) ** 2 + ((y - 40.0) / 3.0) ** 2)
    )
    image += np.random.default_rng(0).normal(0.0, 2.5, image.shape)

    header = fits.Header()
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRPIX1"] = 41.0
    header["CRPIX2"] = 41.0
    header["CRVAL1"] = 150.0
    header["CRVAL2"] = 2.0
    header["CDELT1"] = -0.21 / 3600.0
    header["CDELT2"] = 0.21 / 3600.0
    header["BUNIT"] = "ADU/s"
    header["EXPTIME"] = 1.0
    header["PROCSOFT"] = "VST-tube vers. 1.8.5"
    header["SOFTNAME"] = "SWarp"
    header["PHOTZP"] = (30.0, "mag = -2.5 log10(ADU) + PHOTZP")
    header["PHOTSYS"] = "AB"
    header["BKGMED"] = (320.0, "BACKMEAN median")

    fits.writeto(path, image, header)


@pytest.fixture
def advisor_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict]:
    """Replace the preparation and PSF stages of the advisor."""
    calls: list[dict] = []

    def fake_stages(self, resolved):
        calls.append(dict(resolved))
        products = SimpleNamespace(
            noise=np.full((60, 80), 2.5),
            prepared_data=np.zeros((60, 80)),
            background=-0.4,
        )

        return products, None

    monkeypatch.setattr(
        ObservationAdvisor,
        "_run_fixed_input_stages",
        fake_stages,
    )

    return calls


def _advise_arguments(
    tmp_path: Path,
    fits_path: Path,
    output_path: Path,
) -> list[str]:
    """Return the arguments of an advised configuration with a Gaussian PSF."""
    return [
        "baymorph",
        "config",
        str(fits_path),
        "--target-x",
        "40",
        "--target-y",
        "40",
        "--advise",
        "--psf",
        "gaussian",
        "--gaussian-fwhm-pixels",
        "2.74",
        "--outputs-root",
        str(tmp_path / "runs"),
        "--output",
        str(output_path),
    ]


def test_config_command_advises_and_writes_only_with_accept(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    advisor_stages: list[dict],
) -> None:
    """An advised configuration is printed for review and written on --accept."""
    fits_path = tmp_path / "coadd.fits"
    output_path = tmp_path / "configs" / "coadd.yaml"

    _write_coadd(fits_path)

    arguments = [
        *_advise_arguments(tmp_path, fits_path, output_path),
        "--background-model",
        "plane",
    ]
    monkeypatch.setattr("sys.argv", arguments)

    assert main() == 0

    printed = capsys.readouterr().out

    assert "BayMorph Header Reading" in printed
    assert "BayMorph Configuration Proposal" in printed
    assert "--accept" in printed
    assert not output_path.exists()

    monkeypatch.setattr("sys.argv", [*arguments, "--accept"])

    assert main() == 0

    config = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    report_path = tmp_path / "configs" / "coadd.advisor.json"

    assert config["fit"]["models"] == ["single_sersic"]
    assert config["fit"]["priors"]["single_sersic"]["flux"]["type"] == "log_uniform"
    assert config["fit"]["background"]["model"] == "plane"
    assert config["fit"]["likelihood"]["effective_gain"] == {
        "source": "sky_level",
        "sky_level": 320.0,
    }
    assert Path(config["data"]["image_path"]) == fits_path.resolve()
    assert Path(config["outputs"]["root"]) == (tmp_path / "runs").resolve()
    assert report_path.is_file()
    assert str(report_path.resolve()) in capsys.readouterr().out
    assert len(advisor_stages) == 2


def test_config_command_advises_several_models_with_qualified_priors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    advisor_stages: list[dict],
) -> None:
    """Repeated --model options give one proposal with every model."""
    fits_path = tmp_path / "coadd.fits"
    output_path = tmp_path / "coadd.yaml"

    _write_coadd(fits_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            *_advise_arguments(tmp_path, fits_path, output_path),
            "--accept",
            "--model",
            "single_sersic",
            "--model",
            "double_sersic",
            "--uniform-prior",
            "double_sersic.q_bulge",
            "0.5",
            "1",
        ],
    )

    assert main() == 0

    config = yaml.safe_load(output_path.read_text(encoding="utf-8"))

    assert config["fit"]["models"] == ["single_sersic", "double_sersic"]
    assert config["fit"]["priors"]["double_sersic"]["q_bulge"] == {
        "type": "uniform",
        "lower": 0.5,
        "upper": 1.0,
    }
    assert config["fit"]["priors"]["double_sersic"]["q_disk"] == {
        "type": "uniform",
        "lower": 0.4,
        "upper": 1.0,
    }
    assert len(advisor_stages) == 1


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--accept"], "--accept requires --advise"),
        (["--offline"], "--offline requires --advise"),
        (["--background-model", "plane"], "--background-model requires --advise"),
        (
            ["--resampling-variance-factor", "0.8"],
            "--resampling-variance-factor requires --advise",
        ),
        (
            ["--model", "single_sersic", "--model", "bulge_disk"],
            "Several --model options require --advise",
        ),
        (
            ["--model", "single_sersic", "--model", "single_sersic"],
            "more than once",
        ),
        (
            [
                "--advise",
                "--model",
                "single_sersic",
                "--model",
                "bulge_disk",
                "--uniform-prior",
                "flux",
                "1",
                "2",
            ],
            "name the model of each prior",
        ),
        (
            ["--advise", "--uniform-prior", "bulge_disk.flux_disk", "1", "2"],
            "not among the requested models",
        ),
        (
            ["--photometric-system", "AB"],
            "--photometric-system qualifies a declared zeropoint",
        ),
        (["--zeropoint", "25", "--photometric-system", "ST"], "not one of AB, Vega"),
        (["--gain", "-1"], "argument --gain: '-1' is not positive"),
        (["--zeropoint", "nan"], "argument --zeropoint: 'nan' is not finite"),
        (["--image-unit", " "], "The image unit must be non-empty"),
    ],
)
def test_config_command_rejects_ambiguous_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    advisor_stages: list[dict],
    extra: list[str],
    message: str,
) -> None:
    """Options that need the advisor, or name no model, must be refused."""
    fits_path = tmp_path / "galaxy.fits"
    output_path = tmp_path / "galaxy.yaml"

    _write_fits(fits_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "baymorph",
            "config",
            str(fits_path),
            "--target-center",
            "--output",
            str(output_path),
            *extra,
        ],
    )

    with pytest.raises(SystemExit) as raised:
        main()

    assert raised.value.code == 2
    assert message in capsys.readouterr().err
    assert not output_path.exists()
    assert advisor_stages == []


def test_config_command_writes_hdu_preparation_and_psf_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The image HDU, the sky correlation and the PSF ceiling are declared."""
    fits_path = tmp_path / "galaxy.fits"
    output_path = tmp_path / "galaxy.yaml"
    data = np.zeros((21, 31), dtype=float)
    header = fits.Header()

    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRPIX1"] = 16.0
    header["CRPIX2"] = 11.0
    header["CRVAL1"] = 150.0
    header["CRVAL2"] = 2.0
    header["CDELT1"] = -0.2 / 3600.0
    header["CDELT2"] = 0.2 / 3600.0

    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data, header, name="SCI"),
            fits.ImageHDU(np.ones_like(data), name="WHT"),
        ]
    ).writeto(fits_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "baymorph",
            "config",
            str(fits_path),
            "--target-center",
            "--hdu",
            "SCI",
            "--sky-correlation",
            "field_separated",
            "--psf-max-peak",
            "20000",
            "--uniform-prior",
            "single_sersic.flux",
            "1",
            "1000",
            "--output",
            str(output_path),
        ],
    )

    assert main() == 0

    config = yaml.safe_load(output_path.read_text(encoding="utf-8"))

    assert config["data"]["hdu"] == "SCI"
    assert "calibration" not in config["data"]
    assert config["preparation"] == {
        "noise": {"sky_correlation": "field_separated"},
    }
    assert config["psf"]["empirical"]["candidate_selection"]["max_peak"] == 20000.0
    assert config["fit"]["priors"]["single_sersic"]["flux"] == {
        "type": "uniform",
        "lower": 1.0,
        "upper": 1000.0,
    }


def test_config_command_declares_the_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The calibration options must be written to data.calibration."""
    fits_path = tmp_path / "galaxy.fits"
    output_path = tmp_path / "galaxy.yaml"

    _write_fits(fits_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "baymorph",
            "config",
            str(fits_path),
            "--target-center",
            "--image-unit",
            "electron/s",
            "--zeropoint",
            "25.94",
            "--photometric-system",
            "ab",
            "--exposure-time",
            "1200",
            "--gain",
            "2",
            "--saturation",
            "50000",
            "--uniform-prior",
            "flux",
            "1",
            "1000",
            "--output",
            str(output_path),
        ],
    )

    assert main() == 0

    config = yaml.safe_load(output_path.read_text(encoding="utf-8"))

    assert config["data"]["calibration"] == {
        "image_unit": "electron/s",
        "zeropoint": 25.94,
        "photometric_system": "AB",
        "exposure_time": 1200.0,
        "gain": 2.0,
        "saturation": 50000.0,
    }


def test_config_command_advises_with_the_declared_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    advisor_stages: list[dict],
) -> None:
    """With --advise, a declared zeropoint sets the flux prior and is shown."""
    fits_path = tmp_path / "coadd.fits"
    output_path = tmp_path / "coadd.yaml"

    _write_coadd(fits_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            *_advise_arguments(tmp_path, fits_path, output_path),
            "--zeropoint",
            "28",
            "--photometric-system",
            "AB",
            "--accept",
        ],
    )

    assert main() == 0

    printed = capsys.readouterr().out
    config = yaml.safe_load(output_path.read_text(encoding="utf-8"))

    assert "data.calibration" in printed
    assert "Declared calibration (data.calibration): zeropoint 28 mag [user]" in (
        printed
    )
    assert config["data"]["calibration"] == {
        "zeropoint": 28.0,
        "photometric_system": "AB",
    }
    assert advisor_stages[0]["data"]["calibration"] == config["data"]["calibration"]
    assert config["fit"]["priors"]["single_sersic"]["flux"]["lower"] == (
        pytest.approx(10.0 ** (0.4 * (28.0 - 30.0)))
    )


def _write_acs_drizzle(path: Path, **cards) -> None:
    """Write an ACS/WFC F814W image drizzled to cps whose PHOTFLAM is 0.

    ``cards`` replace keywords of the header.
    """
    y, x = np.indices((81, 81), dtype=float)
    image = 2.0 * np.exp(-0.5 * (((x - 40.0) / 4.0) ** 2 + ((y - 40.0) / 3.0) ** 2))
    image += np.random.default_rng(0).normal(0.0, 0.01, image.shape)

    header = fits.Header()
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRPIX1"] = 41.0
    header["CRPIX2"] = 41.0
    header["CRVAL1"] = 195.0
    header["CRVAL2"] = 27.9
    header["CDELT1"] = -0.05 / 3600.0
    header["CDELT2"] = 0.05 / 3600.0
    header["TELESCOP"] = "HST"
    header["INSTRUME"] = "ACS"
    header["DETECTOR"] = "WFC"
    header["FILTER2"] = "F814W"
    header["BUNIT"] = "ELECTRONS"
    header["D001OUUN"] = "cps"
    header["PHOTFLAM"] = 0.0
    header["PHOTPLAM"] = 8045.0
    header["EXPSTART"] = 54122.6577005
    header["EXPEND"] = 54122.74369596
    header["EXPTIME"] = 1400.0
    header.update(cards)

    fits.writeto(path, image, header)


def test_config_command_writes_the_zeropoint_of_the_calculator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    advisor_stages: list[dict],
    acs_calculator,
) -> None:
    """--advise --accept writes the zeropoint the calculator gave."""
    fits_path = tmp_path / "acs.fits"
    output_path = tmp_path / "acs.yaml"

    _write_acs_drizzle(fits_path)
    monkeypatch.setattr(
        "sys.argv",
        [*_advise_arguments(tmp_path, fits_path, output_path), "--accept"],
    )

    assert main() == 0

    config = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    report = yaml.safe_load(
        (tmp_path / "acs.advisor.json").read_text(encoding="utf-8")
    )

    assert config["data"]["calibration"] == {
        "zeropoint": 25.931,
        "photometric_system": "AB",
    }
    assert report["bridges"][0]["status"] == "applied"
    assert "Asking the ACS Zeropoint Calculator of STScI" in capsys.readouterr().out
    assert len(acs_calculator.calls) == 1


def test_config_command_advises_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    advisor_stages: list[dict],
    acs_calculator,
) -> None:
    """--offline asks nothing and says what it would have asked."""
    fits_path = tmp_path / "acs.fits"

    _write_acs_drizzle(fits_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            *_advise_arguments(tmp_path, fits_path, tmp_path / "acs.yaml"),
            "--offline",
        ],
    )

    assert main() == 0

    printed = capsys.readouterr().out

    assert "[stsci_acs_zeropoint]: offline." in printed
    assert "Asking" not in printed
    assert acs_calculator.calls == []


def test_accept_writes_nothing_when_the_service_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    advisor_stages: list[dict],
    acs_calculator,
) -> None:
    """A proposal without the answer it needed is written only with --offline."""
    from baymorph.pipeline.calibration_bridges import BridgeTransportError

    fits_path = tmp_path / "acs.fits"
    output_path = tmp_path / "acs.yaml"
    arguments = [*_advise_arguments(tmp_path, fits_path, output_path), "--accept"]

    _write_acs_drizzle(fits_path)
    acs_calculator.error = BridgeTransportError("could not be reached", "no route")
    monkeypatch.setattr("sys.argv", arguments)

    report_path = tmp_path / "acs.advisor.json"

    assert main() == 1
    assert (
        "baymorph: error: The configuration and its report were not written: the "
        "ACS Zeropoint Calculator of STScI did not answer, or gave an answer"
    ) in capsys.readouterr().err
    assert not output_path.exists()
    assert not report_path.exists()

    monkeypatch.setattr("sys.argv", [*arguments, "--offline"])

    assert main() == 0
    assert "calibration" not in yaml.safe_load(output_path.read_text(encoding="utf-8"))[
        "data"
    ]
    assert yaml.safe_load(report_path.read_text(encoding="utf-8"))["bridges"][0][
        "status"
    ] == ("offline")
    assert len(acs_calculator.calls) == 1


@pytest.mark.parametrize(
    ("cards", "options", "calibration", "requests"),
    [
        ({}, ["--zeropoint", "25.94", "--photometric-system", "AB"], True, 0),
        ({"EXPTIME": 99999.0}, [], False, 0),
        ({"EXPSTART": 53906.5, "EXPEND": 53936.5}, [], False, 3),
    ],
    ids=["not_needed", "refused_before_asking", "refused_after_asking"],
)
def test_accept_writes_a_consultation_that_did_not_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    advisor_stages: list[dict],
    acs_calculator,
    cards,
    options,
    calibration,
    requests,
) -> None:
    """Only a failed request stops --accept; a refusal writes the proposal."""
    fits_path = tmp_path / "acs.fits"
    output_path = tmp_path / "acs.yaml"

    _write_acs_drizzle(fits_path, **cards)
    acs_calculator.by_date["2006-07-05"] = acs_calculator.answers["2006-07-20"]
    monkeypatch.setattr(
        "sys.argv",
        [*_advise_arguments(tmp_path, fits_path, output_path), *options, "--accept"],
    )

    assert main() == 0

    written = yaml.safe_load(output_path.read_text(encoding="utf-8"))["data"]

    assert ("calibration" in written) is calibration
    assert (tmp_path / "acs.advisor.json").is_file()
    assert len(acs_calculator.calls) == requests
