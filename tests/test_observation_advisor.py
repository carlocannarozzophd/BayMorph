"""Tests for evidence-aware FITS configuration guidance."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from baymorph import ObservationAdvisor
from baymorph.models.labels import PARAMETER_KINDS, parameter_names


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

    # A FWHM without a unit is not established, so it is not used.
    assert inspection.seeing_fwhm is None

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

    config = advisor.make_config(
        model_name="single_sersic",
        model_priors=priors,
    )

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
        "_load_psf_from_config",
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
    assert "flux" not in captured["priors"]
    assert captured["priors"]["re"] == priors["re"]
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


def test_advisor_flux_calibration_supports_gaussian_psf(
    tmp_path,
    monkeypatch,
) -> None:
    """A Gaussian-PSF configuration must not require an empirical PSF file."""
    from types import SimpleNamespace

    import numpy as np

    import baymorph.pipeline.advisor as advisor_module
    from baymorph.simulation.psf import GaussianPSF

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

    config = advisor.make_config(
        model_name="single_sersic",
        model_priors=priors,
    )
    config["psf"] = {
        "type": "gaussian",
        "gaussian": {
            "fwhm": 3.0,
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

    expected = _make_test_flux_calibration()
    captured = {}

    monkeypatch.setattr(
        advisor_module,
        "load_image_preparation_products",
        lambda resolved: products,
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
    assert isinstance(
        captured["psf"],
        GaussianPSF,
    )
    assert captured["psf"].fwhm == pytest.approx(
        3.0
    )


def test_advisor_build_psf_runs_candidate_selection_before_construction(
    tmp_path,
    monkeypatch,
) -> None:
    """Requested PSF building must execute the complete empirical workflow."""
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
        saturation=1000.0,
        bunit="ADU/s",
    )

    config = advisor.make_config(
        model_name="single_sersic",
        model_priors={
            "flux": {
                "type": "fixed",
                "value": 1.0,
            },
            "re": {
                "type": "uniform",
                "lower": 1.0,
                "upper": 10.0,
            },
        },
    )

    products = SimpleNamespace(
        prepared_data=np.zeros(
            (
                9,
                9,
            ),
            dtype=float,
        ),
        noise=np.ones(
            (
                9,
                9,
            ),
            dtype=float,
        ),
        fitting_mask=np.zeros(
            (
                9,
                9,
            ),
            dtype=bool,
        ),
        background=0.0,
        target_x=4.0,
        target_y=4.0,
    )

    calls = []

    monkeypatch.setattr(
        advisor_module,
        "prepare_real_image_from_config",
        lambda resolved, save: products,
    )

    monkeypatch.setattr(
        advisor_module,
        "run_psf_candidate_selection_from_config",
        lambda resolved: calls.append(
            "candidates"
        ),
    )

    monkeypatch.setattr(
        advisor_module,
        "run_psf_construction_from_config",
        lambda resolved: calls.append(
            "construction"
        ),
    )

    monkeypatch.setattr(
        advisor_module,
        "_load_psf_from_config",
        lambda resolved: object(),
    )

    monkeypatch.setattr(
        advisor_module,
        "calibrate_single_sersic_flux_prior",
        lambda **kwargs: _make_test_flux_calibration(),
    )

    advisor.calibrate_flux_prior(
        config,
        prepare=True,
        build_psf=True,
        n_draws=8,
        bootstrap_samples=0,
    )

    assert calls == [
        "candidates",
        "construction",
    ]


def test_advisor_creates_internal_flux_calibration_workspace(
    tmp_path,
    monkeypatch,
) -> None:
    """Calibration must not require a user-declared preliminary flux prior."""
    from types import SimpleNamespace

    import numpy as np

    import baymorph.pipeline.advisor as advisor_module

    fits_path = tmp_path / "galaxy.fits"
    config_path = tmp_path / "config.yaml"

    _write_source_fits(
        fits_path
    )

    advisor = ObservationAdvisor(
        fits_path,
        target_position="center",
    )

    advisor._inspection = SimpleNamespace(
        saturation=1000.0,
        bunit="ADU/s",
    )

    products = SimpleNamespace(
        prepared_data=np.zeros(
            (
                9,
                9,
            ),
            dtype=float,
        ),
        noise=np.ones(
            (
                9,
                9,
            ),
            dtype=float,
        ),
        fitting_mask=np.zeros(
            (
                9,
                9,
            ),
            dtype=bool,
        ),
        background=0.0,
        target_x=4.0,
        target_y=4.0,
    )

    captured = {}

    def fake_prepare(
        resolved,
        save,
    ):
        captured["runtime_config"] = resolved
        captured["save"] = save
        return products

    monkeypatch.setattr(
        advisor_module,
        "prepare_real_image_from_config",
        fake_prepare,
    )

    monkeypatch.setattr(
        advisor_module,
        "_load_psf_from_config",
        lambda resolved: object(),
    )

    def fake_calibration(
        **kwargs,
    ):
        captured["calibration"] = kwargs
        return _make_test_flux_calibration()

    monkeypatch.setattr(
        advisor_module,
        "calibrate_single_sersic_flux_prior",
        fake_calibration,
    )

    result = advisor.calibrate_flux_prior(
        prepare=True,
        build_psf=False,
        output_path=config_path,
        object_name="test_galaxy",
        outputs_root=tmp_path / "runs",
        psf_type="empirical",
        n_draws=8,
        bootstrap_samples=0,
    )

    assert result is not None
    assert config_path.exists()

    runtime_config = captured[
        "runtime_config"
    ]

    assert "preparation" in runtime_config
    assert "detection" in runtime_config["preparation"]
    assert captured["save"] is True

    calibration_kwargs = captured[
        "calibration"
    ]

    assert "flux" not in calibration_kwargs["priors"]


def test_flux_prior_coverage_guard_uses_logarithmic_prior_measure():
    """Bright diagnostics near a log-prior edge must require review."""
    from baymorph.pipeline.advisor import (
        _flux_prior_requires_review,
        _prior_volume_position,
    )

    calibrated_prior = {
        "type": "log_uniform",
        "lower": 12510.587338311978,
        "upper": 5830215.232335205,
    }

    observed_scale = 4.91838e6

    position = _prior_volume_position(
        calibrated_prior,
        observed_scale,
    )

    assert position == pytest.approx(
        0.9723196232653571,
    )
    assert _flux_prior_requires_review(
        calibrated_prior,
        observed_scale,
    )


def test_flux_prior_coverage_guard_accepts_interior_logarithmic_scale():
    """Diagnostics in the interior of a log prior need no edge warning."""
    from baymorph.pipeline.advisor import (
        _flux_prior_requires_review,
        _prior_volume_position,
    )

    calibrated_prior = {
        "type": "log_uniform",
        "lower": 1.0,
        "upper": 100.0,
    }

    observed_scale = 10.0

    assert _prior_volume_position(
        calibrated_prior,
        observed_scale,
    ) == pytest.approx(
        0.5
    )
    assert not _flux_prior_requires_review(
        calibrated_prior,
        observed_scale,
    )


def test_header_fwhm_formatter_preserves_unit_uncertainty() -> None:
    """Advisor output must distinguish explicit and unresolved units."""
    from baymorph.pipeline.advisor import _format_header_fwhm

    assert (
        _format_header_fwhm(
            0.78812181,
            None,
        )
        == "0.788122 (unit unresolved; not converted)"
    )
    assert (
        _format_header_fwhm(
            0.78812181,
            "arcsec",
        )
        == "0.788122 arcsec"
    )
    assert (
        _format_header_fwhm(
            3.75,
            "pixels",
        )
        == "3.75 pixels"
    )


def test_advisor_summary_reports_units_and_saturation_as_header_metadata(
    tmp_path,
) -> None:
    """Saturation is shown only when its unit is that of the image."""
    fits_path = tmp_path / "galaxy.fits"

    _write_source_fits(
        fits_path
    )

    summary = ObservationAdvisor(
        fits_path,
        target_position="center",
    ).summary()

    assert "Image units: counts" in summary
    assert "Saturation: not established" in summary
    assert "Noise sigma (whole-image diagnostic)" in summary

    fits.setval(
        fits_path,
        "SATLEVEL",
        value=30000.0,
    )

    summary = ObservationAdvisor(
        fits_path,
        target_position="center",
    ).summary()

    assert "Saturation: 30000 in the units of the image" in summary


def test_prior_guidance_covers_every_model_by_kind(
    tmp_path,
) -> None:
    """Every parameter of the ten models must have guidance from its kind."""
    fits_path = tmp_path / "galaxy.fits"

    _write_source_fits(fits_path)

    advisor = ObservationAdvisor(
        fits_path,
        target_position="center",
    )
    families = {
        "flux": {"log_uniform"},
        "length": {"positive_scale"},
        "position": {"uniform_relative_to_target"},
        "angle": {"uniform_axial", "uniform_periodic"},
        "dimensionless": {"bounded", "bounded_0_1"},
    }
    whole_target = {"flux", "re", "q", "theta"}

    for model_name, kinds in PARAMETER_KINDS.items():
        guidance = advisor.prior_guidance(model_name)

        assert list(guidance) == parameter_names(model_name)

        for name, item in guidance.items():
            assert item.parameter == name
            assert item.family in families[kinds[name]], (model_name, name)
            assert item.use_observed_scale_for_prior is False

            if name in whole_target:
                assert item.observed_scale_source == "fitted_image"
                assert "diagnostic" in item.rationale
            else:
                assert item.observed_scale is None
                assert item.observed_scale_source is None

    assert (
        advisor.prior_guidance("bulge_disk_spiral")["phase_spiral"].family
        == "uniform_periodic"
    )
    assert advisor.prior_guidance("double_sersic")["q_disk"].family == "bounded_0_1"
    assert "clips it" in advisor.prior_guidance("bulge_disk")["re_disk"].rationale
    assert "validated" in advisor.prior_guidance("core_sersic")["rb"].rationale

    with pytest.raises(ValueError, match="Unknown model"):
        advisor.prior_guidance("spiral")
