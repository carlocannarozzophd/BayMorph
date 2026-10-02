"""Tests for preparation-free single-observation model screening."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from baymorph.analysis.model_screening import SCREENING_MODELS
from baymorph.analysis.model_screening_features import (
    SCREENING_FEATURE_NAMES,
    extract_model_screening_features,
)
from baymorph.pipeline.advisor import ObservationAdvisor
from baymorph.pipeline.model_screening_advisor import (
    SingleObservationModelAdvisor,
)
from baymorph.validation.model_screening_classifier import (
    fit_linear_discriminant_screening_classifier,
)


def write_test_fits(
    directory: Path,
) -> Path:
    """Write a smooth target with representative FITS metadata."""
    y, x = np.indices(
        (81, 81),
        dtype=float,
    )
    source = (
        150.0
        * np.exp(
            -0.5
            * (
                (
                    (
                        x - 40.0
                    )
                    / 6.0
                )
                ** 2
                + (
                    (
                        y - 40.0
                    )
                    / 4.0
                )
                ** 2
            )
        )
    )
    image = 10.0 + source

    hdu = fits.PrimaryHDU(image)
    hdu.header["FILTER"] = "r_SDSS"
    hdu.header["TELESCOP"] = "TEST"
    hdu.header["INSTRUME"] = "TESTCAM"
    hdu.header["BUNIT"] = "ADU/s"
    hdu.header["PHOTZP"] = 30.0
    hdu.header["EXPTIME"] = 1.0
    hdu.header["TEXPTIME"] = 1200.0
    hdu.header["SATURATE"] = 50000.0
    hdu.header["FWHM"] = 0.8
    hdu.header["CTYPE1"] = "RA---TAN"
    hdu.header["CTYPE2"] = "DEC--TAN"
    hdu.header["CRPIX1"] = 41.0
    hdu.header["CRPIX2"] = 41.0
    hdu.header["CRVAL1"] = 150.0
    hdu.header["CRVAL2"] = -30.0
    hdu.header["CDELT1"] = -0.2 / 3600.0
    hdu.header["CDELT2"] = 0.2 / 3600.0

    filename = directory / "screening_target.fits"
    hdu.writeto(filename)

    return filename


def make_classifier(
    image: np.ndarray,
) -> object:
    """Fit a deterministic classifier around the test feature vector."""
    target = extract_model_screening_features(
        image,
        target_x=40.0,
        target_y=40.0,
        psf_fwhm=4.0,
        background=10.0,
        noise_sigma=1.0,
    )
    generator = np.random.default_rng(15)
    feature_blocks = []
    labels = []

    for model_index, model_name in enumerate(
        SCREENING_MODELS
    ):
        feature_blocks.append(
            target.values
            + float(model_index)
            + generator.normal(
                scale=0.02,
                size=(
                    12,
                    len(SCREENING_FEATURE_NAMES),
                ),
            )
        )
        labels.extend([model_name] * 12)

    return fit_linear_discriminant_screening_classifier(
        np.vstack(feature_blocks),
        labels,
        feature_names=SCREENING_FEATURE_NAMES,
    )


def test_screening_reads_header_and_ranks_all_models(
    tmp_path: Path,
) -> None:
    """The advisor must return metadata and every screening model."""
    filename = write_test_fits(
        tmp_path
    )
    image = np.asarray(
        fits.getdata(filename),
        dtype=float,
    )
    classifier = make_classifier(
        image
    )

    result = SingleObservationModelAdvisor(
        filename,
        target_position="center",
    ).screen_models(
        classifier,
        header_fwhm_unit="arcsec",
        background=10.0,
        noise_sigma=1.0,
    )

    assert result.shape == (81, 81)
    assert result.target_x == pytest.approx(40.0)
    assert result.target_y == pytest.approx(40.0)
    assert result.metadata.filter_name == "r_SDSS"
    assert result.metadata.telescope == "TEST"
    assert result.metadata.instrument == "TESTCAM"
    assert result.metadata.bunit == "ADU/s"
    assert result.psf_fwhm_pixels == pytest.approx(
        4.0,
        rel=1e-6,
    )
    assert (
        result.psf_fwhm_source
        == "fits_header_fwhm_declared_arcsec"
    )
    assert result.report.status == "ambiguous"
    assert {
        score.model_name
        for score in result.report.scores
    } == set(
        SCREENING_MODELS
    )


def test_screening_does_not_call_production_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Screening must remain independent of production preparation."""
    filename = write_test_fits(
        tmp_path
    )
    image = np.asarray(
        fits.getdata(filename),
        dtype=float,
    )
    classifier = make_classifier(
        image
    )

    def fail_inspection(
        self: ObservationAdvisor,
    ) -> None:
        raise AssertionError(
            "Production inspection must not run during screening."
        )

    monkeypatch.setattr(
        ObservationAdvisor,
        "inspect",
        fail_inspection,
    )

    result = SingleObservationModelAdvisor(
        filename,
        target_position=(40.0, 40.0),
    ).screen_models(
        classifier,
        psf_fwhm_pixels=4.0,
        background=10.0,
        noise_sigma=1.0,
    )

    assert result.report.user_decision_required is True


def test_header_fwhm_requires_explicit_unit(
    tmp_path: Path,
) -> None:
    """A non-standard FITS FWHM keyword must never be converted silently."""
    filename = write_test_fits(
        tmp_path
    )
    image = np.asarray(
        fits.getdata(filename),
        dtype=float,
    )
    classifier = make_classifier(
        image
    )

    with pytest.raises(
        ValueError,
        match="unit is ambiguous",
    ):
        SingleObservationModelAdvisor(
            filename,
            target_position="center",
        ).screen_models(
            classifier,
            background=10.0,
            noise_sigma=1.0,
        )


def test_screening_requires_target_position(
    tmp_path: Path,
) -> None:
    """The user must identify the target before screening."""
    filename = write_test_fits(
        tmp_path
    )
    image = np.asarray(
        fits.getdata(filename),
        dtype=float,
    )
    classifier = make_classifier(
        image
    )

    with pytest.raises(
        ValueError,
        match="requires target_position",
    ):
        SingleObservationModelAdvisor(
            filename
        ).screen_models(
            classifier,
            psf_fwhm_pixels=4.0,
            background=10.0,
            noise_sigma=1.0,
        )


def test_summary_preserves_user_decision_and_stage_boundary(
    tmp_path: Path,
) -> None:
    """The summary must state the scientific and workflow limitations."""
    filename = write_test_fits(
        tmp_path
    )
    image = np.asarray(
        fits.getdata(filename),
        dtype=float,
    )
    classifier = make_classifier(
        image
    )

    result = SingleObservationModelAdvisor(
        filename,
        target_position="center",
    ).screen_models(
        classifier,
        psf_fwhm_pixels=4.0,
        background=10.0,
        noise_sigma=1.0,
    )
    summary = result.summary()

    assert "Filter: r_SDSS" in summary
    assert "Status: ambiguous" in summary
    assert "User decision required" in summary
    assert "does not perform production cleaning" in summary
    assert "The user must choose the production model" in summary
