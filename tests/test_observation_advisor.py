"""Tests for evidence-aware FITS configuration guidance."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from baymorph import ObservationAdvisor


def _write_source_fits(
    path,
) -> None:
    """Write a simple extended astronomical source."""
    y, x = np.indices(
        (
            81,
            81,
        ),
        dtype=float,
    )

    source = (
        500.0
        * np.exp(
            -0.5
            * (
                (
                    (
                        x - 40.0
                    )
                    / 5.0
                ) ** 2
                + (
                    (
                        y - 40.0
                    )
                    / 3.0
                ) ** 2
            )
        )
    )

    data = (
        20.0
        + source
    )

    header = fits.Header()

    header[
        "CTYPE1"
    ] = "RA---TAN"

    header[
        "CTYPE2"
    ] = "DEC--TAN"

    header[
        "CRPIX1"
    ] = 41.0

    header[
        "CRPIX2"
    ] = 41.0

    header[
        "CRVAL1"
    ] = 150.0

    header[
        "CRVAL2"
    ] = 2.0

    header[
        "CDELT1"
    ] = (
        -0.2
        / 3600.0
    )

    header[
        "CDELT2"
    ] = (
        0.2
        / 3600.0
    )

    header[
        "FWHM"
    ] = 0.8

    header[
        "BUNIT"
    ] = "counts"

    fits.writeto(
        path,
        data,
        header,
    )


def test_advisor_inspects_target_without_creating_outputs(
    tmp_path,
) -> None:
    """Inspection must be non-destructive."""
    fits_path = (
        tmp_path
        / "galaxy.fits"
    )

    _write_source_fits(
        fits_path
    )

    advisor = ObservationAdvisor(
        fits_path,
        target_position="center",
    )

    inspection = advisor.inspect()

    assert inspection.shape == (
        81,
        81,
    )

    assert (
        inspection.pixel_scale
        == pytest.approx(
            0.2
        )
    )

    assert (
        inspection.seeing_fwhm
        == pytest.approx(
            0.8
        )
    )

    assert (
        inspection.target_flux_scale
        is not None
    )

    assert (
        inspection.target_flux_scale
        > 0.0
    )

    assert (
        inspection.target_half_light_radius
        is not None
    )

    assert (
        inspection.target_half_light_radius
        > 0.0
    )

    assert (
        inspection.target_axis_ratio
        is not None
    )

    assert (
        0.0
        < inspection.target_axis_ratio
        <= 1.0
    )

    assert list(
        tmp_path.glob(
            "*.yaml"
        )
    ) == []

    assert not (
        tmp_path
        / "runs"
    ).exists()


def test_prior_guidance_separates_observed_scales_from_priors(
    tmp_path,
) -> None:
    """Image-derived scales must never become implicit prior bounds."""
    fits_path = (
        tmp_path
        / "galaxy.fits"
    )

    _write_source_fits(
        fits_path
    )

    advisor = ObservationAdvisor(
        fits_path,
        target_position="center",
    )

    guidance = advisor.prior_guidance()

    assert (
        guidance[
            "flux"
        ].family
        == "log_uniform"
    )

    assert (
        guidance[
            "flux"
        ].observed_scale
        is not None
    )

    assert (
        guidance[
            "flux"
        ].evidence_safe
        is True
    )

    assert (
        guidance[
            "re"
        ].observed_scale
        is not None
    )

    assert (
        "bounds"
        not in guidance[
            "flux"
        ].__dataclass_fields__
    )


def test_interactive_configuration_requires_explicit_flux_bounds(
    tmp_path,
) -> None:
    """Interactive configuration must not infer flux-prior bounds."""
    fits_path = (
        tmp_path
        / "galaxy.fits"
    )

    yaml_path = (
        tmp_path
        / "galaxy.yaml"
    )

    _write_source_fits(
        fits_path
    )

    answers = iter(
        [
            "y",
            "log_uniform",
            "10",
            "1000000",
        ]
    )

    advisor = ObservationAdvisor(
        fits_path,
        target_position="center",
    )

    config = advisor.configure(
        output_path=yaml_path,
        input_fn=lambda prompt: next(
            answers
        ),
    )

    assert config is not None

    assert config[
        "fit"
    ][
        "priors"
    ][
        "single_sersic"
    ][
        "flux"
    ] == {
        "type": "log_uniform",
        "lower": 10.0,
        "upper": 1000000.0,
    }

    assert yaml_path.exists()


def test_advisor_can_delegate_directly_to_config_maker(
    tmp_path,
) -> None:
    """Expert users may skip the interactive advisor questions."""
    fits_path = (
        tmp_path
        / "galaxy.fits"
    )

    _write_source_fits(
        fits_path
    )

    advisor = ObservationAdvisor(
        fits_path,
        target_position="center",
    )

    config = advisor.make_config(
        model_priors={
            "flux": {
                "type": "log_uniform",
                "lower": 10.0,
                "upper": 1000000.0,
            },
        },
    )

    assert config[
        "data"
    ][
        "target_position"
    ] == "center"

    assert config[
        "fit"
    ][
        "priors"
    ][
        "single_sersic"
    ][
        "flux"
    ][
        "type"
    ] == "log_uniform"


def test_advisor_marks_image_scales_as_diagnostic_only(
    tmp_path,
) -> None:
    """Data-derived scales must carry explicit evidence-safe provenance."""
    fits_path = tmp_path / "galaxy.fits"

    _write_source_fits(
        fits_path
    )

    advisor = ObservationAdvisor(
        fits_path,
        target_position="center",
    )

    guidance = advisor.prior_guidance()

    for parameter in (
        "flux",
        "re",
        "q",
        "theta",
    ):
        item = guidance[
            parameter
        ]

        assert item.observed_scale_source == "fitted_image"
        assert (
            item.observed_scale_role
            == "diagnostic_numerical_scale"
        )
        assert item.use_observed_scale_for_prior is False
        assert item.evidence_safe is True


def test_advisor_summary_distinguishes_diagnostics_from_priors(
    tmp_path,
) -> None:
    """Human-readable output must not present diagnostics as fitted parameters."""
    fits_path = tmp_path / "galaxy.fits"

    _write_source_fits(
        fits_path
    )

    advisor = ObservationAdvisor(
        fits_path,
        target_position="center",
    )

    summary = advisor.summary()

    assert "OBSERVATION" in summary
    assert "TARGET DIAGNOSTICS" in summary
    assert "EVIDENCE SAFETY" in summary

    assert "Segmented positive-flux scale" in summary
    assert "Segmented half-light radius scale" in summary
    assert "Axis-ratio moment estimate" in summary
    assert "Position-angle moment estimate" in summary

    assert (
        "does not use them automatically to define "
        "scientific prior bounds"
        in summary
    )


def test_advisor_configuration_shows_complete_prior_plan(
    tmp_path,
    capsys,
) -> None:
    """Interactive configuration must expose the prior plan explicitly."""
    fits_path = tmp_path / "galaxy.fits"

    _write_source_fits(
        fits_path
    )

    answers = iter(
        [
            "y",
            "",
            "10",
            "1000000",
        ]
    )

    advisor = ObservationAdvisor(
        fits_path,
        target_position="center",
    )

    advisor.configure(
        input_fn=lambda prompt: next(
            answers
        ),
    )

    output = capsys.readouterr().out

    assert "PRIOR PLAN" in output
    assert "flux" in output
    assert "re" in output
    assert "n" in output
    assert "q" in output
    assert "theta" in output

    assert (
        "data used to define prior: NO"
        in output
    )

    assert (
        "prior bounds: USER DECISION"
        in output
    )


def test_advisor_corrects_unnormalized_likelihood_when_confirmed(
    tmp_path,
) -> None:
    """Evidence-safe configuration must normalize the likelihood."""
    fits_path = tmp_path / "galaxy.fits"

    _write_source_fits(
        fits_path
    )

    answers = iter(
        [
            "y",
            "y",
            "",
            "10",
            "1000000",
        ]
    )

    advisor = ObservationAdvisor(
        fits_path,
        target_position="center",
    )

    config = advisor.configure(
        likelihood={
            "type": "correlated_gaussian",
            "include_normalization": False,
        },
        input_fn=lambda prompt: next(
            answers
        ),
    )

    assert config is not None

    assert (
        config[
            "fit"
        ][
            "likelihood"
        ][
            "include_normalization"
        ]
        is True
    )


def test_advisor_rejects_unnormalized_likelihood_when_declined(
    tmp_path,
) -> None:
    """Advisor mode must not silently create evidence-unsafe configs."""
    fits_path = tmp_path / "galaxy.fits"

    _write_source_fits(
        fits_path
    )

    answers = iter(
        [
            "y",
            "n",
        ]
    )

    advisor = ObservationAdvisor(
        fits_path,
        target_position="center",
    )

    with pytest.raises(
        ValueError,
        match="normalized likelihood",
    ):
        advisor.configure(
            likelihood={
                "type": "correlated_gaussian",
                "include_normalization": False,
            },
            input_fn=lambda prompt: next(
                answers
            ),
        )


def test_advisor_prior_plan_reports_reference_priors(
    tmp_path,
    capsys,
) -> None:
    """Advisor must expose actual reference priors without data-derived bounds."""
    fits_path = tmp_path / "galaxy.fits"

    _write_source_fits(
        fits_path
    )

    answers = iter(
        [
            "y",
            "",
            "10",
            "1000000",
        ]
    )

    advisor = ObservationAdvisor(
        fits_path,
        target_position="center",
    )

    advisor.configure(
        input_fn=lambda prompt: next(
            answers
        ),
    )

    output = capsys.readouterr().out

    assert "PRIOR PLAN" in output
    assert "BayMorph reference prior" in output
    assert "sky" in output
    assert "data used to define prior: NO" in output
    assert "deg" in output
