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
from baymorph.pipeline.advisor import FixedInputs
from baymorph.pipeline.psf_construction import estimate_psf_core_fwhm

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
