"""Tests for reviewable observed-image configuration proposals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy.io import fits

from baymorph import (
    ObservationAdvisor,
    ObservationConfigProposal,
)


def write_observation(
    directory: Path,
) -> Path:
    """Write a simple finite target for Advisor tests.

    The frame is large enough for the whole-image mesh background, which
    the Advisor refuses to estimate when detected sources fill most of the
    image.
    """
    y, x = np.indices(
        (
            81,
            81,
        ),
        dtype=float,
    )

    image = 10.0 + 100.0 * np.exp(
        -0.5 * (((x - 20.0) / 4.0) ** 2 + ((y - 20.0) / 3.0) ** 2)
    )

    filename = directory / "observed.fits"

    fits.writeto(
        filename,
        image,
    )

    return filename


def test_proposal_requires_explicit_acceptance_before_writing(
    tmp_path: Path,
) -> None:
    """Creating or displaying a proposal must never write its YAML."""
    fits_path = write_observation(tmp_path)
    output_path = tmp_path / "configs" / "observed.yaml"

    advisor = ObservationAdvisor(
        fits_path,
        target_position=(
            20.0,
            20.0,
        ),
    )

    proposal = advisor.propose_config(
        output_path=output_path,
        model_name="single_sersic",
        flux_prior={
            "type": "log_uniform",
            "lower": 100.0,
            "upper": 100000.0,
        },
        outputs_root=(tmp_path / "runs"),
        psf_type="gaussian",
        gaussian_fwhm_pixels=2.0,
        likelihood={
            "type": "gaussian",
            "include_normalization": True,
        },
    )

    assert isinstance(
        proposal,
        ObservationConfigProposal,
    )
    assert not output_path.exists()
    assert proposal.starting_model == "single_sersic"
    assert "user decision" in proposal.summary()

    with pytest.raises(
        ValueError,
        match="accept=True",
    ):
        proposal.write()

    written_path = proposal.write(accept=True)

    assert written_path == output_path.resolve()
    assert output_path.is_file()

    config = yaml.safe_load(output_path.read_text(encoding="utf-8"))

    assert config["fit"]["models"] == ["single_sersic"]
    assert config["data"]["target_position"] == {
        "x": 20.0,
        "y": 20.0,
    }
    assert config["fit"]["priors"]["single_sersic"]["flux"] == {
        "type": "log_uniform",
        "lower": 100.0,
        "upper": 100000.0,
    }


def test_proposal_blocks_required_flux_review(
    tmp_path: Path,
) -> None:
    """Unsafe flux coverage must block YAML persistence."""
    output_path = tmp_path / "unsafe.yaml"

    proposal = ObservationConfigProposal(
        config={
            "data": {
                "image_path": "data/observed.fits",
                "target_position": {
                    "x": 20.0,
                    "y": 20.0,
                },
            },
        },
        output_path=output_path,
        starting_model="single_sersic",
        flux_prior={
            "type": "log_uniform",
            "lower": 100.0,
            "upper": 1000.0,
        },
        flux_prior_source="observational calibration",
        requires_flux_review=True,
        warnings=(
            "Flux coverage requires review.",
        ),
    )

    assert "Writing is blocked" in proposal.summary()

    with pytest.raises(
        ValueError,
        match="flux prior requires review",
    ):
        proposal.write(
            accept=True,
        )

    assert not output_path.exists()


def test_proposal_requires_exactly_one_flux_prior_source(
    tmp_path: Path,
) -> None:
    """Ambiguous or missing flux-prior provenance must be rejected."""
    fits_path = write_observation(tmp_path)

    advisor = ObservationAdvisor(
        fits_path,
        target_position=(
            20.0,
            20.0,
        ),
    )

    with pytest.raises(
        ValueError,
        match="exactly one",
    ):
        advisor.propose_config(
            output_path=(tmp_path / "config.yaml"),
            outputs_root=(tmp_path / "runs"),
            psf_type="gaussian",
            gaussian_fwhm_pixels=2.0,
        )

    with pytest.raises(
        ValueError,
        match="exactly one",
    ):
        advisor.propose_config(
            output_path=(tmp_path / "config.yaml"),
            flux_prior={
                "type": "uniform",
                "lower": 1.0,
                "upper": 10.0,
            },
            model_priors={
                "flux": {
                    "type": "uniform",
                    "lower": 2.0,
                    "upper": 20.0,
                },
            },
            outputs_root=(tmp_path / "runs"),
            psf_type="gaussian",
            gaussian_fwhm_pixels=2.0,
        )


def test_proposal_rejects_unnormalized_likelihood(
    tmp_path: Path,
) -> None:
    """Evidence-oriented proposals must retain likelihood normalization."""
    fits_path = write_observation(tmp_path)

    advisor = ObservationAdvisor(
        fits_path,
        target_position=(
            20.0,
            20.0,
        ),
    )

    with pytest.raises(
        ValueError,
        match="normalized likelihood",
    ):
        advisor.propose_config(
            output_path=(tmp_path / "config.yaml"),
            flux_prior={
                "type": "uniform",
                "lower": 1.0,
                "upper": 1000.0,
            },
            outputs_root=(tmp_path / "runs"),
            psf_type="gaussian",
            gaussian_fwhm_pixels=2.0,
            likelihood={
                "type": "gaussian",
                "include_normalization": False,
            },
        )


def test_project_advisor_anchors_notebook_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Notebook-relative paths must resolve from the project root."""
    project_root = tmp_path / "project"
    data_directory = project_root / "data"
    notebook_directory = project_root / "notebooks"
    config_directory = project_root / "configs"

    data_directory.mkdir(parents=True)
    notebook_directory.mkdir()
    config_directory.mkdir()

    fits_path = write_observation(data_directory)

    monkeypatch.chdir(notebook_directory)

    advisor = ObservationAdvisor.from_project(
        Path("data") / fits_path.name,
        target_position=(
            20.0,
            20.0,
        ),
    )

    assert advisor.fits_path == fits_path.resolve()
    assert advisor.project_root == project_root.resolve()

    class RuntimeCaptureError(Exception):
        """Stop calibration after capturing the resolved runtime path."""

    captured = {}

    def fake_prepare(
        resolved,
        save,
    ):
        captured["outputs_root"] = resolved["outputs"]["root"]
        captured["save"] = save
        raise RuntimeCaptureError

    monkeypatch.setattr(
        "baymorph.pipeline.advisor.prepare_real_image_from_config",
        fake_prepare,
    )

    with pytest.raises(RuntimeCaptureError):
        advisor.calibrate_flux_prior(
            prepare=True,
            output_path=config_directory / "calibration.yaml",
            object_name="observed",
            outputs_root="runs",
            psf_type="gaussian",
            gaussian_fwhm_pixels=2.0,
        )

    assert captured["outputs_root"] == str(
        (project_root / "runs").resolve()
    )
    assert captured["save"] is True

    proposal = advisor.propose_config(
        output_path=Path("configs") / "observed.yaml",
        flux_prior={
            "type": "log_uniform",
            "lower": 100.0,
            "upper": 100000.0,
        },
        outputs_root="runs",
        psf_type="gaussian",
        gaussian_fwhm_pixels=2.0,
    )

    expected_config_path = project_root / "configs" / "observed.yaml"

    assert proposal.output_path == expected_config_path
    assert (
        proposal.config["data"]["image_path"]
        == (Path("data") / fits_path.name).as_posix()
    )
    assert not (notebook_directory / "configs" / "observed.yaml").exists()

    written_path = proposal.write(accept=True)

    assert written_path == expected_config_path.resolve()

    written_config = yaml.safe_load(
        written_path.read_text(encoding="utf-8")
    )
    assert proposal.config["outputs"]["root"] == "runs"
    assert written_config["outputs"]["root"] == "runs"


def test_proposal_write_refuses_to_replace_existing_configuration(
    tmp_path: Path,
) -> None:
    """A reviewed configuration must not be replaced silently."""
    fits_path = write_observation(tmp_path)
    output_path = tmp_path / "configs" / "observed.yaml"

    advisor = ObservationAdvisor(
        fits_path,
        target_position=(
            20.0,
            20.0,
        ),
    )

    def propose(upper: float) -> ObservationConfigProposal:
        return advisor.propose_config(
            output_path=output_path,
            flux_prior={
                "type": "log_uniform",
                "lower": 100.0,
                "upper": upper,
            },
            outputs_root=(tmp_path / "runs"),
            psf_type="gaussian",
            gaussian_fwhm_pixels=2.0,
        )

    propose(100000.0).write(accept=True)
    original = output_path.read_text(encoding="utf-8")

    replacement = propose(200000.0)

    assert propose(100000.0).write(accept=True) == output_path.resolve()
    assert output_path.read_text(encoding="utf-8") == original

    with pytest.raises(
        FileExistsError,
        match="already exists and differs",
    ):
        replacement.write(accept=True)

    assert output_path.read_text(encoding="utf-8") == original

    replacement.write(
        accept=True,
        overwrite=True,
    )

    config = yaml.safe_load(output_path.read_text(encoding="utf-8"))

    assert config["fit"]["priors"]["single_sersic"]["flux"]["upper"] == 200000.0


def test_proposal_warns_about_undeclared_unit_dependent_choices(
    tmp_path: Path,
) -> None:
    """Default background priors and PSF ceilings must be flagged, not hidden."""
    fits_path = write_observation(tmp_path)

    advisor = ObservationAdvisor(
        fits_path,
        target_position=(
            20.0,
            20.0,
        ),
    )

    flux_prior = {
        "type": "log_uniform",
        "lower": 100.0,
        "upper": 100000.0,
    }

    undeclared = advisor.propose_config(
        output_path=(tmp_path / "undeclared.yaml"),
        flux_prior=flux_prior,
        outputs_root=(tmp_path / "runs"),
    )

    warnings = "\n".join(undeclared.warnings)

    assert "No residual-background prior was declared" in warnings
    assert "No empirical-PSF candidate peak ceiling was declared" in warnings
    assert undeclared.config["fit"]["background"]["model"] == "constant"
    assert set(undeclared.config["fit"]["background"]["priors"]) == {
        "background_level",
    }

    declared = advisor.propose_config(
        output_path=(tmp_path / "declared.yaml"),
        flux_prior=flux_prior,
        outputs_root=(tmp_path / "runs"),
        background={
            "model": "constant",
            "priors": {
                "background_level": {
                    "type": "uniform",
                    "lower": -5.0,
                    "upper": 5.0,
                },
            },
        },
        psf_max_peak=20000.0,
    )

    warnings = "\n".join(declared.warnings)

    assert "residual-background" not in warnings
    assert "peak ceiling" not in warnings
    assert declared.config["fit"]["background"]["model"] == "constant"
    assert (
        declared.config["psf"]["empirical"]["candidate_selection"]["max_peak"]
        == 20000.0
    )


def test_gaussian_psf_proposal_does_not_request_a_peak_ceiling(
    tmp_path: Path,
) -> None:
    """The PSF-ceiling warning applies only to empirical PSFs."""
    fits_path = write_observation(tmp_path)

    advisor = ObservationAdvisor(
        fits_path,
        target_position=(
            20.0,
            20.0,
        ),
    )

    proposal = advisor.propose_config(
        output_path=(tmp_path / "gaussian.yaml"),
        flux_prior={
            "type": "log_uniform",
            "lower": 100.0,
            "upper": 100000.0,
        },
        outputs_root=(tmp_path / "runs"),
        psf_type="gaussian",
        gaussian_fwhm_pixels=2.0,
    )

    assert not any("peak ceiling" in warning for warning in proposal.warnings)


def test_proposal_records_every_sampled_prior(
    tmp_path: Path,
) -> None:
    """The written YAML must not depend on BayMorph prior defaults."""
    fits_path = write_observation(tmp_path)
    output_path = tmp_path / "configs" / "observed.yaml"

    advisor = ObservationAdvisor(
        fits_path,
        target_position=(
            20.0,
            20.0,
        ),
    )

    re_prior = {
        "type": "uniform",
        "lower": 2.0,
        "upper": 30.0,
    }
    background = {
        "model": "plane",
        "priors": {
            "background_level": {
                "type": "uniform",
                "lower": -5.0,
                "upper": 5.0,
            },
            "background_gradient_x": {
                "type": "uniform",
                "lower": -0.01,
                "upper": 0.01,
            },
            "background_gradient_y": {
                "type": "uniform",
                "lower": -0.01,
                "upper": 0.01,
            },
        },
    }

    proposal = advisor.propose_config(
        output_path=output_path,
        flux_prior={
            "type": "log_uniform",
            "lower": 100.0,
            "upper": 100000.0,
        },
        model_priors={
            "re": re_prior,
        },
        background=background,
        outputs_root=(tmp_path / "runs"),
        psf_type="gaussian",
        gaussian_fwhm_pixels=2.0,
    )

    priors = proposal.config["fit"]["priors"]

    assert set(priors) == {"single_sersic"}
    assert set(priors["single_sersic"]) == {
        "flux",
        "re",
        "n",
        "q",
        "theta",
        "x0",
        "y0",
    }
    assert priors["single_sersic"]["re"] == re_prior
    assert proposal.config["fit"]["background"] == background

    defaults_warning = next(
        warning
        for warning in proposal.warnings
        if "reference defaults" in warning
    )

    assert "n, q, theta, x0, y0" in defaults_warning
    assert "re," not in defaults_warning
    assert "flux" not in defaults_warning

    written = yaml.safe_load(
        proposal.write(accept=True).read_text(encoding="utf-8")
    )

    assert written["fit"]["priors"] == priors
    assert written["fit"]["background"] == background
