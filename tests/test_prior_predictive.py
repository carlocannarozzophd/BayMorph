"""Tests for latent prior-predictive morphology diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.analysis.prior_predictive import (
    evaluate_single_sersic_prior_predictive,
)


class _UnitPSF:
    """Minimal unit PSF satisfying the BayMorph PSF protocol."""

    includes_pixel_response = False

    def kernel(
        self,
        *_args,
        **_kwargs,
    ) -> np.ndarray:
        """Return a one-pixel normalized PSF kernel."""
        return np.ones(
            (
                1,
                1,
            ),
            dtype=float,
        )


def _priors():
    return {
        "flux": {
            "type": "log_uniform",
            "lower": 100.0,
            "upper": 1000.0,
        },
        "re": {
            "type": "uniform",
            "lower": 2.0,
            "upper": 5.0,
        },
        "n": {
            "type": "uniform",
            "lower": 1.0,
            "upper": 4.0,
        },
        "q": {
            "type": "uniform",
            "lower": 0.5,
            "upper": 1.0,
        },
        "theta": {
            "type": "uniform",
            "lower": "-pi/2",
            "upper": "pi/2",
        },
        "x0": {
            "type": "uniform",
            "lower": -1.0,
            "upper": 1.0,
            "relative_to": "target_x",
        },
        "y0": {
            "type": "uniform",
            "lower": -1.0,
            "upper": 1.0,
            "relative_to": "target_y",
        },
    }


def test_single_sersic_prior_predictive_returns_finite_diagnostics():
    report = evaluate_single_sersic_prior_predictive(
        priors=_priors(),
        image_shape=(31, 31),
        psf=_UnitPSF(),
        noise_sigma=2.0,
        references={
            "target_x": 15.0,
            "target_y": 15.0,
        },
        n_draws=16,
        random_state=42,
    )

    assert report.n_draws == 16
    assert report.n_valid == 16
    assert report.valid_fraction == pytest.approx(
        1.0
    )

    flux = report.parameter_samples[
        "flux"
    ]

    assert np.all(
        flux >= 100.0
    )
    assert np.all(
        flux <= 1000.0
    )

    assert np.all(
        np.isfinite(
            report.frame_flux_fraction
        )
    )

    assert np.all(
        report.frame_flux_fraction
        > 0.0
    )

    assert np.all(
        report.peak_snr
        > 0.0
    )

    assert np.all(
        report.quadrature_snr
        > 0.0
    )


def test_relative_centroid_priors_use_declared_references():
    report = evaluate_single_sersic_prior_predictive(
        priors=_priors(),
        image_shape=(31, 31),
        psf=_UnitPSF(),
        noise_sigma=2.0,
        references={
            "target_x": 15.0,
            "target_y": 15.0,
        },
        n_draws=32,
        random_state=7,
    )

    assert np.all(
        report.parameter_samples[
            "x0"
        ]
        >= 14.0
    )

    assert np.all(
        report.parameter_samples[
            "x0"
        ]
        <= 16.0
    )

    assert np.all(
        report.parameter_samples[
            "y0"
        ]
        >= 14.0
    )

    assert np.all(
        report.parameter_samples[
            "y0"
        ]
        <= 16.0
    )


def test_prior_predictive_sampling_is_reproducible():
    kwargs = {
        "priors": _priors(),
        "image_shape": (
            31,
            31,
        ),
        "psf": _UnitPSF(),
        "noise_sigma": 2.0,
        "references": {
            "target_x": 15.0,
            "target_y": 15.0,
        },
        "n_draws": 8,
        "random_state": 11,
    }

    first = evaluate_single_sersic_prior_predictive(
        **kwargs
    )

    second = evaluate_single_sersic_prior_predictive(
        **kwargs
    )

    for name in first.parameter_samples:
        np.testing.assert_allclose(
            first.parameter_samples[
                name
            ],
            second.parameter_samples[
                name
            ],
        )

    np.testing.assert_allclose(
        first.frame_flux_fraction,
        second.frame_flux_fraction,
    )


def test_prior_predictive_respects_inference_mask():
    mask = np.zeros(
        (
            31,
            31,
        ),
        dtype=bool,
    )

    mask[
        :,
        :15,
    ] = True

    report = evaluate_single_sersic_prior_predictive(
        priors=_priors(),
        image_shape=(31, 31),
        psf=_UnitPSF(),
        noise_sigma=2.0,
        references={
            "target_x": 15.0,
            "target_y": 15.0,
        },
        mask=mask,
        n_draws=8,
        random_state=3,
    )

    assert np.all(
        report.usable_flux_fraction
        < report.frame_flux_fraction
    )


def test_prior_predictive_rejects_invalid_noise():
    with pytest.raises(
        ValueError,
        match="finite and positive",
    ):
        evaluate_single_sersic_prior_predictive(
            priors=_priors(),
            image_shape=(31, 31),
            psf=_UnitPSF(),
            noise_sigma=0.0,
            references={
                "target_x": 15.0,
                "target_y": 15.0,
            },
            n_draws=2,
        )


def test_report_summary_is_quantitative_not_prescriptive():
    report = evaluate_single_sersic_prior_predictive(
        priors=_priors(),
        image_shape=(31, 31),
        psf=_UnitPSF(),
        noise_sigma=2.0,
        references={
            "target_x": 15.0,
            "target_y": 15.0,
        },
        n_draws=4,
        random_state=5,
    )

    summary = report.summary()

    assert "Latent Prior-Predictive Diagnostics" in summary
    assert "Frame flux fraction" in summary
    assert "Re / half-frame" in summary
    assert "Peak S/N" in summary
    assert "Quadrature S/N" in summary
    assert "No SAFE/UNSAFE classification" in summary


def test_prior_predictive_tracks_full_frame_peak():
    report = evaluate_single_sersic_prior_predictive(
        priors=_priors(),
        image_shape=(31, 31),
        psf=_UnitPSF(),
        noise_sigma=2.0,
        references={
            "target_x": 15.0,
            "target_y": 15.0,
        },
        n_draws=8,
        random_state=8,
    )

    assert np.all(
        report.frame_peak_surface_brightness
        >= report.peak_surface_brightness
    )


def test_joint_flux_prior_calibration_returns_ordered_candidate():
    from baymorph.analysis.prior_predictive import (
        calibrate_single_sersic_flux_prior,
    )

    calibration = calibrate_single_sersic_flux_prior(
        priors=_priors(),
        image_shape=(31, 31),
        psf=_UnitPSF(),
        noise_sigma=2.0,
        saturation_level=1.0e6,
        subtracted_background=0.0,
        references={
            "target_x": 15.0,
            "target_y": 15.0,
        },
        n_draws=32,
        bootstrap_samples=32,
        random_state=42,
    )

    assert calibration.chosen_flux_lower is not None
    assert calibration.chosen_flux_upper is not None

    assert (
        calibration.chosen_flux_lower
        < calibration.chosen_flux_upper
    )

    assert calibration.width_dex is not None
    assert calibration.width_dex > 0.0

    assert calibration.candidate_prior == {
        "type": "log_uniform",
        "lower": calibration.chosen_flux_lower,
        "upper": calibration.chosen_flux_upper,
    }


def test_joint_flux_prior_calibration_renders_one_ensemble(
    monkeypatch,
):
    import baymorph.analysis.prior_predictive as module

    calls = 0

    original = module.make_single_sersic_psf_model

    def counted_factory(
        **kwargs,
    ):
        nonlocal calls

        calls += 1

        return original(
            **kwargs
        )

    monkeypatch.setattr(
        module,
        "make_single_sersic_psf_model",
        counted_factory,
    )

    module.calibrate_single_sersic_flux_prior(
        priors=_priors(),
        image_shape=(31, 31),
        psf=_UnitPSF(),
        noise_sigma=2.0,
        saturation_level=1.0e6,
        subtracted_background=0.0,
        references={
            "target_x": 15.0,
            "target_y": 15.0,
        },
        n_draws=12,
        bootstrap_samples=16,
        random_state=4,
    )

    assert calls == 12


def test_joint_flux_prior_calibration_bootstrap_is_reproducible():
    from baymorph.analysis.prior_predictive import (
        calibrate_single_sersic_flux_prior,
    )

    kwargs = {
        "priors": _priors(),
        "image_shape": (
            31,
            31,
        ),
        "psf": _UnitPSF(),
        "noise_sigma": 2.0,
        "saturation_level": 1.0e6,
        "subtracted_background": 0.0,
        "references": {
            "target_x": 15.0,
            "target_y": 15.0,
        },
        "n_draws": 24,
        "bootstrap_samples": 32,
        "random_state": 7,
    }

    first = calibrate_single_sersic_flux_prior(
        **kwargs
    )

    second = calibrate_single_sersic_flux_prior(
        **kwargs
    )

    assert first.chosen_flux_lower == pytest.approx(
        second.chosen_flux_lower
    )

    assert first.chosen_flux_upper == pytest.approx(
        second.chosen_flux_upper
    )

    assert (
        first.lower_confidence_interval
        == pytest.approx(
            second.lower_confidence_interval
        )
    )

    assert (
        first.upper_confidence_interval
        == pytest.approx(
            second.upper_confidence_interval
        )
    )


def test_joint_flux_prior_calibration_summary_records_provenance():
    from baymorph.analysis.prior_predictive import (
        calibrate_single_sersic_flux_prior,
    )

    calibration = calibrate_single_sersic_flux_prior(
        priors=_priors(),
        image_shape=(31, 31),
        psf=_UnitPSF(),
        noise_sigma=2.0,
        saturation_level=1.0e6,
        subtracted_background=0.0,
        references={
            "target_x": 15.0,
            "target_y": 15.0,
        },
        n_draws=16,
        bootstrap_samples=16,
        random_state=9,
        unit="counts",
    )

    summary = calibration.summary()

    assert "Flux Prior Calibration" in summary
    assert "LOWER BOUND" in summary
    assert "UPPER BOUND" in summary
    assert "CANDIDATE PRIOR" in summary
    assert "bootstrap interval" in summary
    assert "Measured target flux used to define bounds: NO" in summary
