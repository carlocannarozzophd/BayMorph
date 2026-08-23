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
        is None
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
    """Data-derived scales must remain diagnostic rather than evidence claims."""
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
        assert item.evidence_safe is None


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
        "observed diagnostic used to define bounds: NO"
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
    assert "observed diagnostic used to define bounds: NO" in output
    assert "deg" in output


def test_advisor_warns_for_extremely_broad_log_uniform_flux_prior(
    tmp_path,
    capsys,
) -> None:
    """Broad logarithmic prior volume must be made explicit."""
    fits_path = tmp_path / "galaxy.fits"

    _write_source_fits(
        fits_path
    )

    answers = iter(
        [
            "y",
            "",
            "1",
            "1e10",
            "y",
        ]
    )

    advisor = ObservationAdvisor(
        fits_path,
        target_position="center",
    )

    config = advisor.configure(
        input_fn=lambda prompt: next(
            answers
        ),
    )

    output = capsys.readouterr().out

    assert config is not None
    assert "FLUX PRIOR AUDIT" in output
    assert "prior width: 10.000 dex" in output
    assert "evidence review: WARNING" in output
    assert "prior-volume sensitive" in output
    assert "observed diagnostic used to define bounds: NO" in output


def test_advisor_cancels_warned_flux_prior_cleanly(
    tmp_path,
    capsys,
) -> None:
    """Declining a warned prior must cancel without raising an exception."""
    fits_path = tmp_path / "galaxy.fits"

    _write_source_fits(
        fits_path
    )

    answers = iter(
        [
            "y",
            "",
            "1",
            "1e10",
            "n",
        ]
    )

    advisor = ObservationAdvisor(
        fits_path,
        target_position="center",
    )

    config = advisor.configure(
        input_fn=lambda prompt: next(
            answers
        ),
    )

    output = capsys.readouterr().out

    assert config is None
    assert "Configuration cancelled" in output


def test_advisor_reports_reviewed_moderate_flux_prior(
    tmp_path,
    capsys,
) -> None:
    """A moderate prior must report its logarithmic width explicitly."""
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

    assert "prior width: 5.000 dex" in output
    assert "evidence review: REVIEWED" in output


def test_configure_does_not_repeat_observation_summary(
    tmp_path,
    capsys,
) -> None:
    """Inspection summary and interactive configuration are separate concerns."""
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
    assert "OBSERVATION" not in output
    assert "TARGET DIAGNOSTICS" not in output



def test_theta_guidance_uses_axial_prior_domain(
    tmp_path,
) -> None:
    """Theta diagnostics must use the same axial domain as the prior."""
    fits_path = tmp_path / "galaxy.fits"

    _write_source_fits(
        fits_path
    )

    advisor = ObservationAdvisor(
        fits_path,
        target_position="center",
    )

    advisor.inspect()

    inspection = advisor._inspection

    assert inspection is not None

    object.__setattr__(
        inspection,
        "target_position_angle",
        np.deg2rad(
            177.492
        ),
    )

    theta = advisor.prior_guidance()[
        "theta"
    ].observed_scale

    assert theta is not None
    assert np.degrees(theta) == pytest.approx(
        -2.508,
        abs=1.0e-6,
    )
    assert -0.5 * np.pi <= theta <= 0.5 * np.pi



def _make_test_flux_calibration():
    """Return a deterministic calibrated flux-prior result for advisor tests."""
    import numpy as np

    from baymorph.analysis.prior_predictive import (
        FluxPriorCalibration,
    )

    return FluxPriorCalibration(
        detection_flux=np.array(
            [
                8.0,
                10.0,
                12.0,
            ],
            dtype=float,
        ),
        saturation_flux=np.array(
            [
                800.0,
                1000.0,
                1200.0,
            ],
            dtype=float,
        ),
        snr_threshold=5.0,
        required_detectable_fraction=0.95,
        required_unsaturated_fraction=0.95,
        saturation_level=100.0,
        subtracted_background=1.0,
        prepared_saturation_level=99.0,
        reference_flux=20.0,
        render_valid_fraction=1.0,
        chosen_flux_lower=10.0,
        chosen_flux_upper=1000.0,
        lower_confidence_interval=(
            9.0,
            11.0,
        ),
        upper_confidence_interval=(
            900.0,
            1100.0,
        ),
        confidence_level=0.95,
        unit="ADU/s",
    )


def test_advisor_flux_calibration_uses_canonical_prepared_products(
    tmp_path,
    monkeypatch,
) -> None:
    """Advisor calibration must reuse canonical preparation and PSF products."""
    from types import SimpleNamespace

    import numpy as np

    import baymorph.pipeline.advisor as advisor_module

    fits_path = tmp_path / "galaxy.fits"

    _write_source_fits(
        fits_path
    )

    advisor = ObservationAdvisor(
        fits_path,
        target_position="center",
    )

    advisor._inspection = SimpleNamespace(
        saturation=12345.0,
        bunit="ADU/s",
        target_flux_scale=9.99e99,
    )

    priors = {
        "flux": {
            "type": "log_uniform",
            "lower": 1.0,
            "upper": 1.0e6,
        },
        "re": {
            "type": "uniform",
            "lower": 1.0,
            "upper": 20.0,
        },
    }

    config = {
        "fit": {
            "priors": {
                "single_sersic": priors,
            },
        },
    }

    products = SimpleNamespace(
        prepared_data=np.zeros(
            (
                9,
                11,
            ),
            dtype=float,
        ),
        noise=np.full(
            (
                9,
                11,
            ),
            2.0,
            dtype=float,
        ),
        fitting_mask=np.zeros(
            (
                9,
                11,
            ),
            dtype=bool,
        ),
        background=3.0,
        target_x=5.0,
        target_y=4.0,
    )

    psf = object()
    expected = _make_test_flux_calibration()
    captured = {}

    monkeypatch.setattr(
        advisor_module,
        "load_image_preparation_products",
        lambda resolved: products,
    )

    monkeypatch.setattr(
        advisor_module,
        "load_empirical_psf",
        lambda resolved: psf,
    )

    def fake_calibration(
        **kwargs,
    ):
        captured.update(
            kwargs
        )
        return expected

    monkeypatch.setattr(
        advisor_module,
        "calibrate_single_sersic_flux_prior",
        fake_calibration,
    )

    result = advisor.calibrate_flux_prior(
        config,
        n_draws=32,
        bootstrap_samples=16,
    )

    assert result is expected
    assert captured["priors"] is priors
    assert captured["psf"] is psf
    assert captured["image_shape"] == (
        9,
        11,
    )
    assert captured["references"] == {
        "target_x": 5.0,
        "target_y": 4.0,
    }
    assert captured["saturation_level"] == 12345.0
    assert captured["subtracted_background"] == 3.0
    assert captured["unit"] == "ADU/s"

    assert "target_flux_scale" not in captured

    scalar_numeric_arguments = [
        float(value)
        for value in captured.values()
        if isinstance(
            value,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        )
    ]

    assert 9.99e99 not in scalar_numeric_arguments


def test_advisor_configuration_accepts_calibrated_flux_prior(
    tmp_path,
    capsys,
) -> None:
    """Accepted calibrated bounds must become the explicit flux prior."""
    fits_path = tmp_path / "galaxy.fits"

    _write_source_fits(
        fits_path
    )

    calibration = _make_test_flux_calibration()

    answers = iter(
        [
            "y",
            "",
        ]
    )

    advisor = ObservationAdvisor(
        fits_path,
        target_position="center",
    )

    config = advisor.configure(
        flux_calibration=calibration,
        input_fn=lambda prompt: next(
            answers
        ),
    )

    assert config is not None

    flux_prior = config[
        "fit"
    ][
        "priors"
    ][
        "single_sersic"
    ][
        "flux"
    ]

    assert flux_prior == {
        "type": "log_uniform",
        "lower": 10.0,
        "upper": 1000.0,
    }

    output = capsys.readouterr().out

    assert "CALIBRATED FLUX PRIOR" in output
    assert "calibration status: CALIBRATED" in output
    assert (
        "measured target diagnostic used to define bounds: NO"
        in output
    )
    assert "evidence review: CALIBRATED" in output


def test_advisor_configuration_can_decline_calibration_and_use_manual_prior(
    tmp_path,
) -> None:
    """Declining calibrated bounds must preserve the manual-prior workflow."""
    fits_path = tmp_path / "galaxy.fits"

    _write_source_fits(
        fits_path
    )

    calibration = _make_test_flux_calibration()

    answers = iter(
        [
            "y",
            "n",
            "",
            "20",
            "2000000",
        ]
    )

    advisor = ObservationAdvisor(
        fits_path,
        target_position="center",
    )

    config = advisor.configure(
        flux_calibration=calibration,
        input_fn=lambda prompt: next(
            answers
        ),
    )

    assert config is not None

    flux_prior = config[
        "fit"
    ][
        "priors"
    ][
        "single_sersic"
    ][
        "flux"
    ]

    assert flux_prior == {
        "type": "log_uniform",
        "lower": 20.0,
        "upper": 2000000.0,
    }


def test_advisor_calibrated_prior_is_not_labelled_safe(
    tmp_path,
    capsys,
) -> None:
    """Calibration must not be presented as objective evidence safety."""
    fits_path = tmp_path / "galaxy.fits"

    _write_source_fits(
        fits_path
    )

    calibration = _make_test_flux_calibration()

    answers = iter(
        [
            "y",
            "",
        ]
    )

    advisor = ObservationAdvisor(
        fits_path,
        target_position="center",
    )

    advisor.configure(
        flux_calibration=calibration,
        input_fn=lambda prompt: next(
            answers
        ),
    )

    output = capsys.readouterr().out

    assert "CALIBRATED" in output
    assert "evidence status: SAFE" not in output
    assert "evidence review: SAFE" not in output
