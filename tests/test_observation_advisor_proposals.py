"""Tests for the configuration proposals of the observation advisor (AD-049)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from astropy.io import fits

from baymorph import ObservationAdvisor
from baymorph.models.labels import PARAMETER_KINDS
from baymorph.pipeline.advisor import FixedInputs
from baymorph.pipeline.config import resolve_config, validate_fit_priors
from baymorph.pipeline.prior_proposals import (
    noise_equivalent_area,
    point_source_flux_prior,
)
from baymorph.pipeline.psf_construction import estimate_psf_core_fwhm
from baymorph.simulation.psf import GaussianPSF

SKY_SCALE = 2.5
SKY_LEVEL = 320.0
CUTOUT_SHAPE = (60, 80)


def _write_coadd(directory: Path, *, saturation_unit_known: bool = False) -> Path:
    """Write a small sky-subtracted coadd with a VST-Tube and SWarp header."""
    y, x = np.indices((81, 81), dtype=float)
    image = 100.0 * np.exp(-0.5 * (((x - 40.0) / 4.0) ** 2 + ((y - 40.0) / 3.0) ** 2))
    image += np.random.default_rng(0).normal(0.0, SKY_SCALE, image.shape)

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
    header["SOFTNAME"] = "SWarp"
    header["GAIN"] = (34.9, "Maximum equivalent gain (e-/ADU)")
    header["RESAMPT1"] = "LANCZOS3"
    header["SATURATE"] = (26000.0, "Saturation Level (ADU)")

    if not saturation_unit_known:
        header["PROCSOFT"] = "VST-tube vers. 1.8.5"
        header["PHOTZP"] = (30.0, "mag = -2.5 log10(ADU) + PHOTZP")
        header["PHOTSYS"] = "AB"
        header["BKGMED"] = (SKY_LEVEL, "BACKMEAN median")

    path = directory / "coadd.fits"
    fits.writeto(path, image, header)

    return path


def _psf_kernel(sigma: float = 1.2, size: int = 25) -> np.ndarray:
    """Return a normalized Gaussian kernel."""
    y, x = np.indices((size, size), dtype=float)
    centre = (size - 1) / 2
    kernel = np.exp(-0.5 * ((x - centre) ** 2 + (y - centre) ** 2) / sigma**2)

    return kernel / kernel.sum()


@pytest.fixture
def stages(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Replace the preparation and PSF stages with known fixed inputs."""
    calls: list[dict] = []

    def fake_stages(self, resolved):
        calls.append(dict(resolved))
        products = SimpleNamespace(
            noise=np.full(CUTOUT_SHAPE, SKY_SCALE),
            prepared_data=np.zeros(CUTOUT_SHAPE),
            background=-0.4,
        )
        psf = None

        if resolved["psf"]["type"] == "empirical":
            psf = _psf_kernel()

        return products, psf

    monkeypatch.setattr(
        ObservationAdvisor,
        "_run_fixed_input_stages",
        fake_stages,
    )

    return calls


def test_proposal_takes_its_priors_from_header_physics_and_fixed_inputs(
    tmp_path: Path,
    stages: list[dict],
) -> None:
    """With no declared prior, every rule of AD-049 must apply."""
    fits_path = _write_coadd(tmp_path)
    advisor = ObservationAdvisor(fits_path, target_position=(40.0, 40.0))

    proposal = advisor.propose_config(
        output_path=tmp_path / "configs" / "coadd.yaml",
        outputs_root=tmp_path / "runs",
        psf_type="gaussian",
        gaussian_fwhm_pixels=2.74,
        background_model="plane",
    )

    assert len(stages) == 1

    priors = proposal.config["fit"]["priors"]["single_sersic"]

    assert priors["flux"]["type"] == "log_uniform"
    assert priors["flux"]["lower"] == pytest.approx(1.0)
    assert priors["flux"]["upper"] == pytest.approx(10.0**8.8)
    assert priors["re"]["type"] == "log_uniform"
    assert priors["re"]["lower"] == pytest.approx(1.0)
    assert priors["re"]["upper"] == pytest.approx(120.0 / 0.21)
    assert priors["x0"]["upper"] == pytest.approx(3.0 * 2.74)
    assert priors["n"] == {"type": "uniform", "lower": 0.5, "upper": 6.0}

    background = proposal.config["fit"]["background"]

    assert background["model"] == "plane"
    assert background["priors"]["background_level"]["upper"] == pytest.approx(
        5.0 * SKY_SCALE
    )
    assert background["priors"]["background_gradient_x"]["upper"] == pytest.approx(
        5.0 * SKY_SCALE / 40.0
    )
    assert background["priors"]["background_gradient_y"]["upper"] == pytest.approx(
        5.0 * SKY_SCALE / 30.0
    )

    gain = proposal.config["fit"]["likelihood"]["effective_gain"]

    assert gain == {"source": "sky_level", "sky_level": SKY_LEVEL}

    provenance = proposal.provenance
    sources = {
        name: entry["source"]
        for name, entry in provenance["priors"]["single_sersic"].items()
    }

    assert sources == {
        "flux": "header",
        "re": "header",
        "n": "reference",
        "q": "reference",
        "theta": "reference",
        "x0": "fixed_inputs",
        "y0": "fixed_inputs",
    }
    assert provenance["decision"] == "AD-049"
    assert provenance["models"] == ["single_sersic"]
    assert provenance["fixed_inputs"]["sky_scale"] == pytest.approx(SKY_SCALE)
    assert provenance["header"]["quantities"]["zeropoint"]["value"] == 30.0
    assert provenance["effective_gain"]["candidates"]["gain_keyword"] == pytest.approx(
        34.9
    )

    summary = proposal.summary()

    assert "PRIORS AND THEIR SOURCES" in summary
    assert "FIXED INPUTS" in summary
    assert "Gain cross-check" in summary


def test_written_proposal_records_its_provenance_beside_the_yaml(
    tmp_path: Path,
    stages: list[dict],
) -> None:
    """The JSON report must be written only on acceptance, with the YAML."""
    fits_path = _write_coadd(tmp_path)
    advisor = ObservationAdvisor(fits_path, target_position=(40.0, 40.0))
    output_path = tmp_path / "configs" / "coadd.yaml"

    proposal = advisor.propose_config(
        output_path=output_path,
        outputs_root=tmp_path / "runs",
        psf_type="gaussian",
        gaussian_fwhm_pixels=2.74,
    )

    assert proposal.report_path == tmp_path / "configs" / "coadd.advisor.json"
    assert not proposal.report_path.exists()

    proposal.write(accept=True)

    report = json.loads(proposal.report_path.read_text(encoding="utf-8"))
    written = yaml.safe_load(output_path.read_text(encoding="utf-8"))

    assert report["priors"]["single_sersic"]["re"]["source"] == "header"
    assert report["fixed_inputs"]["cutout_shape"] == list(CUTOUT_SHAPE)
    assert written["fit"]["priors"] == proposal.config["fit"]["priors"]

    assert proposal.write(accept=True) == output_path.resolve()

    proposal.report_path.write_text("{}\n", encoding="utf-8")
    output_path.unlink()

    with pytest.raises(FileExistsError, match="Provenance report already exists"):
        proposal.write(accept=True)

    assert not output_path.exists()


def test_declared_values_are_kept_and_recorded_as_the_users(
    tmp_path: Path,
    stages: list[dict],
) -> None:
    """Priors, background, gain and PSF ceiling of the user must stand."""
    fits_path = _write_coadd(tmp_path)
    advisor = ObservationAdvisor(fits_path, target_position=(40.0, 40.0))
    re_prior = {"type": "uniform", "lower": 2.0, "upper": 50.0}
    background = {
        "model": "constant",
        "priors": {
            "background_level": {"type": "uniform", "lower": -3.0, "upper": 3.0},
        },
    }
    likelihood = {
        "type": "correlated_gaussian",
        "effective_gain": {"source": "declared", "gain": 40.0},
    }

    proposal = advisor.propose_config(
        output_path=tmp_path / "coadd.yaml",
        outputs_root=tmp_path / "runs",
        model_priors={"re": re_prior},
        background=background,
        likelihood=likelihood,
        psf_max_peak=12000.0,
    )

    assert proposal.config["fit"]["priors"]["single_sersic"]["re"] == re_prior
    assert proposal.config["fit"]["background"] == background
    assert proposal.config["fit"]["likelihood"]["effective_gain"] == {
        "source": "declared",
        "gain": 40.0,
    }
    assert (
        proposal.config["psf"]["empirical"]["candidate_selection"]["max_peak"]
        == 12000.0
    )

    provenance = proposal.provenance

    assert provenance["priors"]["single_sersic"]["re"]["source"] == "user"
    assert provenance["background"]["source"] == "user"
    assert provenance["effective_gain"]["source"] == "user"
    assert provenance["psf_max_peak"]["source"] == "user"
    assert provenance["fixed_inputs"]["psf_fwhm_pixels"] == pytest.approx(
        estimate_psf_core_fwhm(_psf_kernel())
    )

    with pytest.raises(ValueError, match="not both"):
        advisor.propose_config(
            output_path=tmp_path / "other.yaml",
            outputs_root=tmp_path / "runs",
            background=background,
            background_model="plane",
        )


def test_psf_ceiling_is_proposed_before_the_psf_is_built(
    tmp_path: Path,
    stages: list[dict],
) -> None:
    """A saturation level in image units must shape the PSF stage."""
    fits_path = _write_coadd(tmp_path, saturation_unit_known=True)
    advisor = ObservationAdvisor(fits_path, target_position=(40.0, 40.0))

    proposal = advisor.propose_config(
        output_path=tmp_path / "coadd.yaml",
        outputs_root=tmp_path / "runs",
        flux_prior={"type": "log_uniform", "lower": 10.0, "upper": 1.0e7},
    )

    ceiling = proposal.config["psf"]["empirical"]["candidate_selection"]["max_peak"]

    assert ceiling == pytest.approx(13000.0)
    assert stages[0]["psf"]["empirical"]["candidate_selection"]["max_peak"] == (
        pytest.approx(13000.0)
    )
    assert proposal.provenance["psf_max_peak"]["source"] == "header"
    assert proposal.flux_prior_source == "explicit user-declared scientific prior"
    assert not any("peak ceiling" in warning for warning in proposal.warnings)


def test_fixed_inputs_must_match_the_configuration_of_the_proposal(
    tmp_path: Path,
    stages: list[dict],
) -> None:
    """Fixed inputs of another cutout must not be used silently."""
    fits_path = _write_coadd(tmp_path)
    advisor = ObservationAdvisor(fits_path, target_position=(40.0, 40.0))

    inputs = advisor.measure_fixed_inputs(
        outputs_root=tmp_path / "runs",
        psf_type="gaussian",
        gaussian_fwhm_pixels=2.74,
    )

    assert isinstance(inputs, FixedInputs)
    assert inputs.sky_scale == pytest.approx(SKY_SCALE)
    assert inputs.measured_sky_level == pytest.approx(-0.4)
    assert inputs.psf_fwhm_pixels == pytest.approx(2.74)
    assert inputs.psf_noise_equivalent_area == pytest.approx(
        noise_equivalent_area(GaussianPSF(fwhm=2.74).kernel())
    )

    reused = advisor.propose_config(
        output_path=tmp_path / "coadd.yaml",
        fixed_inputs=inputs,
        outputs_root=tmp_path / "runs",
        psf_type="gaussian",
        gaussian_fwhm_pixels=2.74,
    )

    assert len(stages) == 1
    assert reused.provenance["fixed_inputs"] == inputs.as_dict()

    with pytest.raises(ValueError, match="another preparation or PSF"):
        advisor.propose_config(
            output_path=tmp_path / "coadd.yaml",
            fixed_inputs=inputs,
            outputs_root=tmp_path / "runs",
            psf_type="gaussian",
            gaussian_fwhm_pixels=2.74,
            cutout_mode="fixed",
            fixed_size=51,
        )


def test_stages_run_on_the_anchored_project_configuration(
    tmp_path: Path,
    stages: list[dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The products must land where the fit of the written YAML looks."""
    project = tmp_path / "project"
    (project / "data").mkdir(parents=True)
    (project / "notebooks").mkdir()
    (project / "configs").mkdir()
    fits_path = _write_coadd(project / "data")
    monkeypatch.chdir(project / "notebooks")

    advisor = ObservationAdvisor.from_project(
        Path("data") / fits_path.name,
        target_position=(40.0, 40.0),
    )
    preparation = {
        "noise": {"sky_correlation": "field_separated"},
    }

    proposal = advisor.propose_config(
        output_path=Path("configs") / "coadd.yaml",
        outputs_root="runs",
        psf_type="gaussian",
        gaussian_fwhm_pixels=2.74,
        preparation=preparation,
    )

    assert stages[0]["outputs"]["root"] == str((project / "runs").resolve())
    assert stages[0]["preparation"]["noise"]["sky_correlation"] == "field_separated"
    assert proposal.config["outputs"]["root"] == "runs"
    assert proposal.config["preparation"] == preparation
    assert proposal.provenance["fixed_inputs"]["run_directory"] == str(
        (project / "runs").resolve() / "coadd"
    )


def test_log_uniform_proposals_are_refused_with_pocomc(
    tmp_path: Path,
    stages: list[dict],
) -> None:
    """pocoMC cannot sample log-uniform priors (finding B2)."""
    fits_path = _write_coadd(tmp_path)
    advisor = ObservationAdvisor(fits_path, target_position=(40.0, 40.0))

    with pytest.raises(ValueError, match="pocoMC"):
        advisor.propose_config(
            output_path=tmp_path / "coadd.yaml",
            outputs_root=tmp_path / "runs",
            psf_type="gaussian",
            gaussian_fwhm_pixels=2.74,
            sampler="pocomc",
        )


def test_multi_component_model_keeps_its_component_ranges(
    tmp_path: Path,
    stages: list[dict],
) -> None:
    """A bulge and disc proposal must keep the labelling reference ranges."""
    fits_path = _write_coadd(tmp_path)
    advisor = ObservationAdvisor(fits_path, target_position=(40.0, 40.0))

    proposal = advisor.propose_config(
        output_path=tmp_path / "coadd.yaml",
        model_name="bulge_disk",
        outputs_root=tmp_path / "runs",
        psf_type="gaussian",
        gaussian_fwhm_pixels=2.74,
    )

    priors = proposal.config["fit"]["priors"]["bulge_disk"]

    assert priors["re_bulge"] == {"type": "uniform", "lower": 1.0, "upper": 30.0}
    assert priors["re_disk"] == {"type": "uniform", "lower": 10.0, "upper": 120.0}
    assert priors["flux_bulge"]["type"] == "log_uniform"
    assert proposal.flux_prior is None
    assert proposal.provenance["priors"]["bulge_disk"]["re_bulge"]["source"] == "reference"

    with pytest.raises(ValueError, match="one total flux"):
        advisor.propose_config(
            output_path=tmp_path / "coadd.yaml",
            model_name="bulge_disk",
            flux_prior={"type": "log_uniform", "lower": 1.0, "upper": 1.0e6},
            outputs_root=tmp_path / "runs",
        )


def test_fixed_inputs_measured_first_settle_the_ceiling_of_the_proposal(
    tmp_path: Path,
    stages: list[dict],
) -> None:
    """Fixed inputs measured before the proposal must carry its PSF ceiling."""
    fits_path = _write_coadd(tmp_path, saturation_unit_known=True)
    advisor = ObservationAdvisor(fits_path, target_position=(40.0, 40.0))
    options = {"outputs_root": tmp_path / "runs"}

    inputs = advisor.measure_fixed_inputs(**options)

    assert stages[0]["psf"]["empirical"]["candidate_selection"]["max_peak"] == (
        pytest.approx(13000.0)
    )
    assert inputs.psf_noise_equivalent_area == pytest.approx(
        noise_equivalent_area(_psf_kernel())
    )

    proposal = advisor.propose_config(
        output_path=tmp_path / "coadd.yaml",
        fixed_inputs=inputs,
        flux_prior={"type": "log_uniform", "lower": 10.0, "upper": 1.0e7},
        **options,
    )

    assert len(stages) == 1
    assert proposal.provenance["fixed_inputs"] == inputs.as_dict()
    assert proposal.provenance["psf_max_peak"]["source"] == "header"


def test_one_proposal_holds_several_models_with_their_priors(
    tmp_path: Path,
    stages: list[dict],
) -> None:
    """Every chosen model must receive its priors by the same rules."""
    fits_path = _write_coadd(tmp_path)
    advisor = ObservationAdvisor(fits_path, target_position=(40.0, 40.0))
    output_path = tmp_path / "configs" / "coadd.yaml"
    models = ["single_sersic", "double_sersic", "bulge_disk"]
    q_disk = {"type": "uniform", "lower": 0.2, "upper": 1.0}

    proposal = advisor.propose_config(
        output_path=output_path,
        models=models,
        model_priors={"bulge_disk": {"q_disk": q_disk}},
        outputs_root=tmp_path / "runs",
        psf_type="gaussian",
        gaussian_fwhm_pixels=2.74,
        background_model="plane",
    )

    assert len(stages) == 1

    config = proposal.config
    priors = config["fit"]["priors"]

    assert config["fit"]["models"] == models
    assert list(priors) == models

    header_flux = {
        "type": "log_uniform",
        "lower": pytest.approx(1.0),
        "upper": pytest.approx(10.0**8.8),
    }

    for model_name, name in (
        ("single_sersic", "flux"),
        ("double_sersic", "flux_bulge"),
        ("double_sersic", "flux_disk"),
        ("bulge_disk", "flux_bulge"),
        ("bulge_disk", "flux_disk"),
    ):
        assert priors[model_name][name] == header_flux

    for model_name in models:
        assert list(priors[model_name]) == list(PARAMETER_KINDS[model_name])
        assert priors[model_name]["x0"]["upper"] == pytest.approx(3.0 * 2.74)

    assert priors["single_sersic"]["re"]["type"] == "log_uniform"
    assert priors["double_sersic"]["re_bulge"] == {
        "type": "uniform",
        "lower": 1.0,
        "upper": 30.0,
    }
    assert priors["double_sersic"]["q_bulge"] == {
        "type": "uniform",
        "lower": 0.4,
        "upper": 1.0,
    }
    assert priors["bulge_disk"]["q_disk"] == q_disk
    assert config["fit"]["background"]["model"] == "plane"

    provenance = proposal.provenance

    assert provenance["models"] == models
    assert list(provenance["priors"]) == models
    assert provenance["priors"]["bulge_disk"]["q_disk"]["source"] == "user"
    assert provenance["priors"]["double_sersic"]["q_disk"]["source"] == "reference"
    assert proposal.models == tuple(models)
    assert proposal.starting_model == "single_sersic"
    assert proposal.flux_prior is None

    summary = proposal.summary()

    assert "User-selected models: single_sersic, double_sersic, bulge_disk" in summary
    assert "own evidence" in summary
    assert "double_sersic:" in summary

    proposal.write(accept=True)

    written = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    resolved = resolve_config(written)
    validate_fit_priors(resolved)

    assert resolved["fit"]["models"] == models
    assert written["fit"]["priors"] == priors


def test_choice_of_models_and_their_priors_must_be_explicit(
    tmp_path: Path,
    stages: list[dict],
) -> None:
    """Ambiguous model choices must be refused before any stage runs."""
    fits_path = _write_coadd(tmp_path)
    advisor = ObservationAdvisor(fits_path, target_position=(40.0, 40.0))
    options = {
        "output_path": tmp_path / "coadd.yaml",
        "outputs_root": tmp_path / "runs",
        "psf_type": "gaussian",
        "gaussian_fwhm_pixels": 2.74,
    }
    re_prior = {"type": "uniform", "lower": 2.0, "upper": 50.0}

    with pytest.raises(ValueError, match="not both"):
        advisor.propose_config(
            model_name="single_sersic",
            models=["bulge_disk"],
            **options,
        )

    with pytest.raises(TypeError, match="sequence of model names"):
        advisor.propose_config(models="single_sersic", **options)

    with pytest.raises(ValueError, match="at least one"):
        advisor.propose_config(models=[], **options)

    with pytest.raises(ValueError, match="more than once"):
        advisor.propose_config(models=["single_sersic", "single_sersic"], **options)

    with pytest.raises(ValueError, match="Unknown model"):
        advisor.propose_config(models=["single_sersic", "spiral"], **options)

    with pytest.raises(ValueError, match="not among the chosen models"):
        advisor.propose_config(
            models=["single_sersic"],
            model_priors={"bulge_disk": {"re_disk": re_prior}},
            **options,
        )

    with pytest.raises(ValueError, match="has no parameters"):
        advisor.propose_config(
            models=["single_sersic", "bulge_disk"],
            model_priors={"bulge_disk": {"re": re_prior}},
            **options,
        )

    with pytest.raises(ValueError, match="one model"):
        advisor.propose_config(
            models=["single_sersic", "core_sersic"],
            flux_prior={"type": "log_uniform", "lower": 1.0, "upper": 1.0e6},
            **options,
        )

    assert stages == []


def test_without_a_zeropoint_each_flux_starts_from_a_point_source_at_snr_one(
    tmp_path: Path,
    stages: list[dict],
) -> None:
    """Without a zeropoint, the fluxes come from the sky scale and the PSF."""
    fits_path = _write_coadd(tmp_path, saturation_unit_known=True)
    advisor = ObservationAdvisor(fits_path, target_position=(40.0, 40.0))
    flux = {"type": "log_uniform", "lower": 10.0, "upper": 1.0e7}

    proposal = advisor.propose_config(
        output_path=tmp_path / "coadd.yaml",
        models=["single_sersic", "bulge_disk"],
        model_priors={"single_sersic": {"flux": flux}},
        outputs_root=tmp_path / "runs",
    )

    area = noise_equivalent_area(_psf_kernel())
    expected = point_source_flux_prior(SKY_SCALE, area)
    priors = proposal.config["fit"]["priors"]
    sources = proposal.provenance["priors"]

    assert priors["single_sersic"]["flux"] == flux
    assert sources["single_sersic"]["flux"]["source"] == "user"

    for name in ("flux_bulge", "flux_disk"):
        assert priors["bulge_disk"][name]["lower"] == pytest.approx(
            SKY_SCALE * np.sqrt(area)
        )
        assert priors["bulge_disk"][name] == pytest.approx(expected)
        assert sources["bulge_disk"][name]["source"] == "fixed_inputs"
        assert sources["bulge_disk"][name]["detail"].startswith(
            "no usable zeropoint: from "
        )

    assert proposal.provenance["rules"]["point_source_snr"] == 1.0
    assert proposal.provenance["fixed_inputs"]["psf_noise_equivalent_area"] == (
        pytest.approx(area)
    )

    summary = proposal.summary()

    assert "flux_disk: log_uniform(" in summary
    assert "[fixed_inputs] no usable zeropoint: from" in summary
    assert "22 magnitudes brighter" in summary
    assert "PSF noise-equivalent area:" in summary


def test_declared_calibration_reaches_every_reading_and_the_configuration(
    tmp_path: Path,
    stages: list[dict],
) -> None:
    """Declarations made to the advisor must reach the proposal (AD-053)."""
    fits_path = _write_coadd(tmp_path, saturation_unit_known=True)
    advisor = ObservationAdvisor(
        fits_path,
        target_position=(40.0, 40.0),
        calibration={
            "zeropoint": 28.0,
            "photometric_system": "ab",
            "saturation": 20000.0,
        },
    )
    declared = {"zeropoint": 28.0, "photometric_system": "AB", "saturation": 20000.0}

    assert advisor.calibration == declared
    assert "data.calibration" in advisor.header_report()
    assert advisor.inspect().saturation == pytest.approx(20000.0)

    proposal = advisor.propose_config(
        output_path=tmp_path / "coadd.yaml",
        outputs_root=tmp_path / "runs",
    )

    assert proposal.config["data"]["calibration"] == declared
    assert stages[0]["data"]["calibration"] == declared

    flux = proposal.provenance["priors"]["single_sersic"]["flux"]

    assert flux["source"] == "declaration"
    assert flux["prior"]["lower"] == pytest.approx(10.0 ** (0.4 * (28.0 - 30.0)))
    assert "magnitudes 8 to 30 AB through the declared zeropoint 28" in flux["detail"]

    provenance = proposal.provenance

    assert provenance["calibration"]["zeropoint"] == {
        "value": 28.0,
        "unit": "mag",
        "source": "user",
    }
    assert provenance["header"]["quantities"]["zeropoint"]["status"] == "declared"
    assert provenance["psf_max_peak"] == {
        "value": pytest.approx(10000.0),
        "source": "declaration",
        "detail": "0.5 times the saturation level 20000 (declared in data.calibration)",
    }

    summary = proposal.summary()

    assert "Declared calibration (data.calibration): zeropoint 28 mag [user]" in (
        summary
    )
    assert "saturation 20000 in the units of the image [user]" in summary

    with pytest.raises(TypeError, match="when creating the advisor"):
        advisor.make_config(calibration={"zeropoint": 25.0})

    with pytest.raises(ValueError, match="cannot declare"):
        ObservationAdvisor(fits_path, calibration={"pixel_scale": 0.2})


def test_header_report_tables_what_the_header_holds(tmp_path: Path) -> None:
    """The report must show the values read, their units and their sources."""
    fits_path = _write_coadd(tmp_path)

    report = ObservationAdvisor(fits_path).header_report()

    assert report.startswith("BayMorph Header Reading")
    assert "HDU: selected by BayMorph" in report
    assert "zeropoint" in report
    assert "PHOTZP" in report
    assert "BKGMED" in report
    assert "HDU: 0" in ObservationAdvisor(fits_path, hdu=0).header_report()


def test_header_report_reads_no_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report reads the header only, so a large mosaic is never loaded."""
    fits_path = _write_coadd(tmp_path)

    def refuse_pixels(*args, **kwargs):
        raise AssertionError("the header report opened the pixels")

    monkeypatch.setattr("baymorph.pipeline.advisor.open_observation", refuse_pixels)

    assert "PHOTZP" in ObservationAdvisor(fits_path).header_report()


ACS_OPTIONS = {"psf_type": "gaussian", "gaussian_fwhm_pixels": 2.0}
SERVED = {"zeropoint": 25.931, "photometric_system": "AB"}


def _write_acs_drizzle(directory: Path) -> Path:
    """Write a small ACS/WFC F814W image drizzled to cps whose PHOTFLAM is 0.

    The dates and the exposure time are those of an image of the Coma
    Treasury Survey, so the calculator is asked for 2007-01-22.
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
    header["FILTER1"] = "CLEAR1L"
    header["FILTER2"] = "F814W"
    header["FILENAME"] = "j9ty22010_drz.fits"
    header["BUNIT"] = "ELECTRONS"
    header["D001OUUN"] = "cps"
    header["PHOTFLAM"] = 0.0
    header["PHOTPLAM"] = 8045.0
    header["EXPSTART"] = 54122.6577005
    header["EXPEND"] = 54122.74369596
    header["EXPTIME"] = 1400.0

    path = directory / "acs.fits"
    fits.writeto(path, image, header)

    return path


def test_the_calculator_gives_the_zeropoint_the_header_lacks(
    tmp_path: Path,
    stages: list[dict],
    acs_calculator,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One request settles the zeropoint for every reading of the advisor."""
    fits_path = _write_acs_drizzle(tmp_path)
    advisor = ObservationAdvisor(fits_path, target_position=(40.0, 40.0))
    options = {"outputs_root": tmp_path / "runs", **ACS_OPTIONS}

    report = advisor.header_report()
    advisor.inspect()
    fixed_inputs = advisor.measure_fixed_inputs(**options)
    proposal = advisor.propose_config(
        output_path=tmp_path / "acs.yaml", fixed_inputs=fixed_inputs, **options
    )

    assert len(acs_calculator.calls) == 1
    assert "Asking the ACS Zeropoint Calculator of STScI" in capsys.readouterr().out
    assert advisor.calibration == {}
    assert stages[0]["data"]["calibration"] == SERVED
    assert proposal.config["data"]["calibration"] == SERVED

    flux = proposal.provenance["priors"]["single_sersic"]["flux"]

    assert flux["source"] == "declaration"
    assert flux["prior"]["lower"] == pytest.approx(10.0 ** (0.4 * (25.931 - 30.0)))
    assert "through the zeropoint 25.931 given by the ACS Zeropoint" in flux["detail"]

    provenance = proposal.provenance

    assert provenance["schema_version"] == 3
    assert provenance["online"] is True
    assert provenance["calibration"] == {
        "zeropoint": {"value": 25.931, "unit": "mag", "source": "stsci_acs_zeropoint"},
        "photometric_system": {
            "value": "AB",
            "unit": None,
            "source": "stsci_acs_zeropoint",
        },
    }

    (record,) = provenance["bridges"]

    assert record["status"] == "applied"
    assert record["values"] == SERVED
    assert record["run"] == {"asked_at": acs_calculator.asked_at, "error": None}
    assert provenance["header"]["quantities"]["zeropoint"]["status"] == "declared"
    assert provenance["header"]["quantities"]["zeropoint"]["note"].startswith(
        "From the ACS Zeropoint Calculator of STScI (stsci_acs_zeropoint). "
        "Declared in data.calibration;"
    )

    summary = proposal.summary()

    assert "zeropoint 25.931 mag [stsci_acs_zeropoint]" in summary
    assert "CALIBRATION SERVICES" in summary
    assert "- ACS Zeropoint Calculator of STScI [stsci_acs_zeropoint]: applied." in (
        summary
    )
    assert f"Asked at {acs_calculator.asked_at} at https://" in summary
    assert "Calibration services" in report
    assert "- zeropoint: From the ACS Zeropoint Calculator of STScI" in report


def test_offline_the_advisor_asks_nothing_and_uses_the_rule_without_a_zeropoint(
    tmp_path: Path,
    stages: list[dict],
    acs_calculator,
) -> None:
    """Offline, the request is recorded and the fluxes need no zeropoint."""
    fits_path = _write_acs_drizzle(tmp_path)
    advisor = ObservationAdvisor(
        fits_path, target_position=(40.0, 40.0), online=False
    )

    proposal = advisor.propose_config(
        output_path=tmp_path / "acs.yaml",
        outputs_root=tmp_path / "runs",
        **ACS_OPTIONS,
    )

    assert acs_calculator.calls == []
    assert "calibration" not in proposal.config["data"]
    assert proposal.provenance["online"] is False
    assert proposal.provenance["bridges"][0]["status"] == "offline"
    assert proposal.provenance["priors"]["single_sersic"]["flux"]["source"] == (
        "fixed_inputs"
    )
    assert not any("Calculator" in warning for warning in proposal.warnings)

    with pytest.raises(TypeError, match="online must be True or False"):
        ObservationAdvisor(fits_path, online="no")


def test_a_failed_request_falls_back_with_a_stable_warning(
    tmp_path: Path,
    stages: list[dict],
    acs_calculator,
) -> None:
    """A failure is stated in words that do not change, the error apart."""
    from baymorph.pipeline.calibration_bridges import BridgeTransportError

    acs_calculator.error = BridgeTransportError(
        "did not answer within 30 s", "The read operation timed out"
    )
    advisor = ObservationAdvisor(
        _write_acs_drizzle(tmp_path), target_position=(40.0, 40.0)
    )

    proposal = advisor.propose_config(
        output_path=tmp_path / "acs.yaml",
        outputs_root=tmp_path / "runs",
        **ACS_OPTIONS,
    )

    (record,) = proposal.provenance["bridges"]
    (warning,) = [item for item in proposal.warnings if "Calculator" in item]

    assert record["status"] == "failed"
    assert record["run"]["error"] == "The read operation timed out"
    assert "did not answer within 30 s" in record["detail"]
    assert "did not answer within 30 s" in warning
    assert "timed out" not in record["detail"] and "timed out" not in warning
    assert proposal.provenance["priors"]["single_sersic"]["flux"]["source"] == (
        "fixed_inputs"
    )
    assert "(The read operation timed out)" in proposal.summary()


def test_writing_again_ignores_when_the_service_answered(
    tmp_path: Path,
    stages: list[dict],
    acs_calculator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same answer at another time is the same proposal; another is not."""
    from baymorph.pipeline import calibration_bridges

    fits_path = _write_acs_drizzle(tmp_path)
    output_path = tmp_path / "configs" / "acs.yaml"
    options = {
        "output_path": output_path,
        "outputs_root": tmp_path / "runs",
        **ACS_OPTIONS,
    }

    def propose():
        advisor = ObservationAdvisor(fits_path, target_position=(40.0, 40.0))

        return advisor.propose_config(**options)

    first = propose()
    first.write(accept=True)
    written = first.report_path.read_text(encoding="utf-8")

    monkeypatch.setattr(calibration_bridges, "_utc_now", lambda: "2026-10-21T09:00:00Z")
    again = propose()

    assert again.provenance["bridges"][0]["run"]["asked_at"] == "2026-10-21T09:00:00Z"
    assert again.write(accept=True) == output_path.resolve()
    assert first.report_path.read_text(encoding="utf-8") == written

    # Another answer with the same zeropoint leaves the YAML unchanged, but it
    # is another report, refused with the keys that differ.
    headers = acs_calculator.answers["2007-01-22"]["headers"]
    acs_calculator.answer = {
        "headers": headers,
        "rows": [
            ["WFC", "F814W", "8045.2", "7.1340e-20", "26.767", "25.500", "25.931"]
        ],
    }
    other_answer = propose()
    configuration = output_path.read_text(encoding="utf-8")

    with pytest.raises(
        FileExistsError,
        match=r"Provenance report already exists and differs at "
        r"bridges\.0\.detail, bridges\.0\.response\.rows\.0\.3, ",
    ):
        other_answer.write(accept=True)

    assert output_path.read_text(encoding="utf-8") == configuration
    assert first.report_path.read_text(encoding="utf-8") == written

    # A recalibration of STScI changes the zeropoint, and so the YAML.
    acs_calculator.answer = {
        "headers": headers,
        "rows": [
            ["WFC", "F814W", "8045.2", "7.0683e-20", "26.777", "25.510", "25.941"]
        ],
    }
    changed = propose()

    with pytest.raises(
        FileExistsError,
        match=r"Configuration already exists and differs at "
        r"data\.calibration\.zeropoint, ",
    ):
        changed.write(accept=True)


def test_writing_again_keeps_a_failure_with_the_same_reason(
    tmp_path: Path,
    stages: list[dict],
    acs_calculator,
) -> None:
    """The verbatim error of a failure may change; its reason may not."""
    from baymorph.pipeline.calibration_bridges import BridgeTransportError

    fits_path = _write_acs_drizzle(tmp_path)
    options = {
        "output_path": tmp_path / "acs.yaml",
        "outputs_root": tmp_path / "runs",
        **ACS_OPTIONS,
    }

    def propose(error):
        acs_calculator.error = error
        advisor = ObservationAdvisor(fits_path, target_position=(40.0, 40.0))

        return advisor.propose_config(**options)

    first = propose(BridgeTransportError("did not answer within 30 s", "timed out"))
    first.write(accept=True)
    written = first.report_path.read_text(encoding="utf-8")

    again = propose(
        BridgeTransportError("did not answer within 30 s", "The read timed out")
    )

    assert again.write(accept=True) == (tmp_path / "acs.yaml").resolve()
    assert first.report_path.read_text(encoding="utf-8") == written

    other = propose(BridgeTransportError("could not be reached", "no route"))

    with pytest.raises(
        FileExistsError, match=r"Provenance report already exists and differs at "
    ):
        other.write(accept=True)


def test_writing_offline_again_keeps_a_proposal_no_service_serves(
    tmp_path: Path,
    stages: list[dict],
) -> None:
    """Whether the advisor was online changes nothing a service did not do."""
    fits_path = _write_coadd(tmp_path)
    options = {
        "output_path": tmp_path / "coadd.yaml",
        "outputs_root": tmp_path / "runs",
        "psf_type": "gaussian",
        "gaussian_fwhm_pixels": 2.74,
    }

    online = ObservationAdvisor(fits_path, target_position=(40.0, 40.0))
    first = online.propose_config(**options)
    first.write(accept=True)
    written = first.report_path.read_text(encoding="utf-8")

    offline = ObservationAdvisor(fits_path, target_position=(40.0, 40.0), online=False)
    again = offline.propose_config(**options)

    assert again.provenance["online"] is False
    assert again.provenance["bridges"] == []
    assert again.write(accept=True) == (tmp_path / "coadd.yaml").resolve()
    assert first.report_path.read_text(encoding="utf-8") == written


def test_a_proposal_rests_on_one_settlement_of_the_calibration(
    tmp_path: Path,
    acs_calculator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file that changes state while the stages run stops the proposal."""
    fits_path = _write_acs_drizzle(tmp_path)

    def stages_that_touch_the_file(self, resolved):
        fits_path.chmod(0o600 if fits_path.stat().st_mode & 0o777 != 0o600 else 0o644)
        products = SimpleNamespace(
            noise=np.full(CUTOUT_SHAPE, SKY_SCALE),
            prepared_data=np.zeros(CUTOUT_SHAPE),
            background=-0.4,
        )

        return products, None

    monkeypatch.setattr(
        ObservationAdvisor, "_run_fixed_input_stages", stages_that_touch_the_file
    )
    advisor = ObservationAdvisor(fits_path, target_position=(40.0, 40.0))

    with pytest.raises(RuntimeError, match="the calibration was settled again"):
        advisor.propose_config(
            output_path=tmp_path / "acs.yaml",
            outputs_root=tmp_path / "runs",
            **ACS_OPTIONS,
        )

    assert len(acs_calculator.calls) == 2


def test_a_declared_zeropoint_is_never_asked_for(
    tmp_path: Path,
    stages: list[dict],
    acs_calculator,
) -> None:
    """A declaration of the user comes before any service."""
    advisor = ObservationAdvisor(
        _write_acs_drizzle(tmp_path),
        target_position=(40.0, 40.0),
        calibration={"zeropoint": 25.94, "photometric_system": "AB"},
    )

    proposal = advisor.propose_config(
        output_path=tmp_path / "acs.yaml",
        outputs_root=tmp_path / "runs",
        **ACS_OPTIONS,
    )

    assert acs_calculator.calls == []
    assert proposal.provenance["bridges"][0]["status"] == "not_needed"
    assert proposal.provenance["calibration"]["zeropoint"]["source"] == "user"
    assert proposal.config["data"]["calibration"] == {
        "zeropoint": 25.94,
        "photometric_system": "AB",
    }


def test_target_selection_asks_no_service(tmp_path: Path) -> None:
    """Choosing the target, by click or by number, sends no request."""
    fits_path = _write_acs_drizzle(tmp_path)
    advisor = ObservationAdvisor(fits_path, target_position="center")

    assert advisor.select_target().value is None
    assert advisor.set_target_position(40.0, 41.0) == (40.0, 41.0)


def test_a_new_calibration_or_online_state_reaches_every_reading(
    tmp_path: Path,
    acs_calculator,
) -> None:
    """A changed declaration or online flag settles the calibration again."""
    coadd = ObservationAdvisor(_write_coadd(tmp_path), target_position=(40.0, 40.0))

    assert "Saturation: not established" in coadd.summary()

    coadd.calibration = {"saturation": 5000.0}

    assert "Saturation: 5000 in the units of the image" in coadd.summary()

    advisor = ObservationAdvisor(
        _write_acs_drizzle(tmp_path), target_position=(40.0, 40.0), online=False
    )

    assert "[stsci_acs_zeropoint]: offline." in advisor.header_report()
    assert acs_calculator.calls == []

    advisor.online = True

    assert "[stsci_acs_zeropoint]: applied." in advisor.header_report()
    assert len(acs_calculator.calls) == 1


def test_a_new_declaration_settles_the_calibration_again(
    tmp_path: Path,
    acs_calculator,
) -> None:
    """Declarations made after a request replace what the service gave."""
    advisor = ObservationAdvisor(
        _write_acs_drizzle(tmp_path), target_position=(40.0, 40.0)
    )

    assert "stsci_acs_zeropoint" in advisor.header_report()
    assert "stsci_acs_zeropoint" in advisor.header_report()
    assert len(acs_calculator.calls) == 1

    advisor.calibration = {"zeropoint": 26.0, "photometric_system": "Vega"}
    report = advisor.header_report()

    assert "not_needed. The zeropoint 26 is declared in data.calibration" in report
    assert "From the ACS Zeropoint Calculator" not in report
    assert len(acs_calculator.calls) == 1

    config = advisor.make_config(
        model_priors={"flux": {"type": "log_uniform", "lower": 1.0, "upper": 1e6}},
        **ACS_OPTIONS,
    )

    assert config["data"]["calibration"] == {
        "zeropoint": 26.0,
        "photometric_system": "Vega",
    }


def test_every_reading_follows_a_new_file_or_hdu(tmp_path: Path) -> None:
    """A rewritten file or another HDU is inspected again by every reading."""
    fits_path = _write_coadd(tmp_path)
    advisor = ObservationAdvisor(fits_path, target_position=(40.0, 40.0))
    before = advisor.prior_guidance()["flux"].observed_scale

    with fits.open(fits_path) as hdus:
        data, header = hdus[0].data * 4.0, hdus[0].header

    fits.writeto(fits_path, data, header, overwrite=True)

    assert advisor.prior_guidance()["flux"].observed_scale == pytest.approx(
        4.0 * before, rel=0.05
    )

    second = np.random.default_rng(1).normal(0.0, SKY_SCALE, (61, 71))
    fits.HDUList(
        [fits.PrimaryHDU(data, header), fits.ImageHDU(second, header)]
    ).writeto(fits_path, overwrite=True)
    advisor.hdu = 0

    assert "Shape: 81 x 81 pixels" in advisor.summary()

    advisor.hdu = 1

    assert "Shape: 71 x 61 pixels" in advisor.summary()


def test_calibrate_flux_prior_reads_the_current_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A saturation declared after an inspection reaches the calibration."""
    from baymorph.pipeline import advisor as advisor_module

    advisor = ObservationAdvisor(
        _write_coadd(tmp_path),
        target_position=(40.0, 40.0),
        calibration={"saturation": 5000.0},
    )
    config = advisor.make_config(
        model_priors={"flux": {"type": "log_uniform", "lower": 1.0, "upper": 1e6}},
        psf_type="gaussian",
        gaussian_fwhm_pixels=2.0,
        outputs_root=tmp_path / "runs",
    )

    assert advisor.inspect().saturation == pytest.approx(5000.0)

    advisor.calibration = {"saturation": 8000.0}
    seen: dict = {}
    monkeypatch.setattr(
        advisor_module,
        "load_image_preparation_products",
        lambda resolved: SimpleNamespace(
            target_x=40.0,
            target_y=40.0,
            prepared_data=np.zeros(CUTOUT_SHAPE),
            noise=np.full(CUTOUT_SHAPE, SKY_SCALE),
            background=0.0,
            fitting_mask=np.zeros(CUTOUT_SHAPE, dtype=bool),
        ),
    )
    monkeypatch.setattr(advisor_module, "_load_psf_from_config", lambda resolved: None)
    monkeypatch.setattr(
        advisor_module,
        "calibrate_single_sersic_flux_prior",
        lambda **options: seen.update(options) or "calibrated",
    )

    assert advisor.calibrate_flux_prior(config) == "calibrated"
    assert seen["saturation_level"] == pytest.approx(8000.0)
