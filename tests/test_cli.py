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
    ):
        main()


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
    ):
        main()


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

    with pytest.raises(SystemExit):
        main()

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
    assert config["preparation"] == {
        "noise": {"sky_correlation": "field_separated"},
    }
    assert config["psf"]["empirical"]["candidate_selection"]["max_peak"] == 20000.0
    assert config["fit"]["priors"]["single_sersic"]["flux"] == {
        "type": "uniform",
        "lower": 1.0,
        "upper": 1000.0,
    }
