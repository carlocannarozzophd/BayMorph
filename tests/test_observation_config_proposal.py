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
    """Write a simple finite target for Advisor tests."""
    y, x = np.indices(
        (
            41,
            41,
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


def test_emcee_proposal_summary_marks_evidence_unavailable(
    tmp_path: Path,
) -> None:
    """An emcee proposal must not imply that the run produces evidence."""
    proposal = ObservationConfigProposal(
        config={
            "data": {
                "image_path": "data/observed.fits",
                "target_position": {
                    "x": 20.0,
                    "y": 20.0,
                },
            },
            "fit": {
                "sampler": {
                    "name": "emcee",
                },
            },
        },
        output_path=(
            tmp_path
            / "emcee.yaml"
        ),
        starting_model="single_sersic",
        flux_prior={
            "type": "log_uniform",
            "lower": 100.0,
            "upper": 1000.0,
        },
        flux_prior_source="explicit scientific prior",
        requires_flux_review=False,
    )

    summary = proposal.summary()

    assert "posterior only" in summary
    assert "Bayesian evidence unavailable" in summary
    assert "cannot establish a preferred model" in summary
    assert "configured inference and evidence comparison" not in summary


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
