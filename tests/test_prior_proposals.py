"""Tests for the priors and settings proposed by the advisor (AD-049)."""

from __future__ import annotations

import math

import pytest
from astropy.io import fits

from baymorph.core.header_metadata import read_header
from baymorph.models.labels import PARAMETER_KINDS, parameter_names
from baymorph.pipeline.prior_config import prior_config_for_parameter
from baymorph.pipeline.prior_proposals import (
    PHYSICAL_LENGTH_MODELS,
    PriorProposal,
    angular_length_prior,
    background_priors,
    centre_priors,
    clipped_reference_length_prior,
    length_domain,
    magnitude_flux_prior,
    propose_effective_gain,
    propose_model_priors,
    propose_psf_max_peak,
    reference_priors,
)


def _wcs_header(scale_arcsec: float = 0.2) -> fits.Header:
    """Return a header with a celestial WCS of the given pixel scale."""
    header = fits.Header()
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRPIX1"] = 50.0
    header["CRPIX2"] = 50.0
    header["CRVAL1"] = 150.0
    header["CRVAL2"] = 2.0
    header["CDELT1"] = -scale_arcsec / 3600.0
    header["CDELT2"] = scale_arcsec / 3600.0
    return header


def _coadd_header(
    *,
    sky_level: float | None = 326.87,
    resampled: bool = True,
) -> fits.Header:
    """Return a header of a VST-Tube coadd written by SWarp."""
    header = _wcs_header(0.21)
    header["BUNIT"] = "ADU/s"
    header["EXPTIME"] = 1.0
    header["PROCSOFT"] = "VST-tube vers. 1.8.5"
    header["SOFTNAME"] = "SWarp"
    header["GAIN"] = (34.9, "Maximum equivalent gain (e-/ADU)")
    header["PHOTZP"] = (30.0, "mag = -2.5 log10(ADU) + PHOTZP")
    header["PHOTSYS"] = "AB"
    header["SATURATE"] = (26197.4, "Saturation Level (ADU)")

    if sky_level is not None:
        header["BKGMED"] = (sky_level, "BACKMEAN median")

    if resampled:
        header["RESAMPT1"] = "LANCZOS3"

    return header


def _single_exposure_header() -> fits.Header:
    """Return a header of a single exposure in ADU with a gain keyword."""
    header = _wcs_header(0.26)
    header["BUNIT"] = "adu"
    header["GAIN"] = (2.5, "e-/ADU")
    return header


def test_magnitude_flux_prior_converts_through_the_zeropoint() -> None:
    """Magnitudes 8 to 30 must map to a log-uniform flux in image units."""
    prior = magnitude_flux_prior(30.0)

    assert prior["type"] == "log_uniform"
    assert prior["lower"] == pytest.approx(1.0)
    assert prior["upper"] == pytest.approx(10.0**8.8)

    nanomaggies = magnitude_flux_prior(22.5, magnitudes=(14.0, 24.0))

    assert nanomaggies["lower"] == pytest.approx(10.0 ** (0.4 * (22.5 - 24.0)))
    assert nanomaggies["upper"] == pytest.approx(10.0 ** (0.4 * (22.5 - 14.0)))

    with pytest.raises(ValueError, match="increasing order"):
        magnitude_flux_prior(30.0, magnitudes=(30.0, 8.0))

    with pytest.raises(ValueError, match="finite"):
        magnitude_flux_prior(math.nan)


def test_angular_length_prior_respects_the_validated_domain() -> None:
    """An effective radius must stay at or above 1 px (AD-046)."""
    domain = length_domain("single_sersic", "re")

    assert domain == (1.0, None)

    coarse = angular_length_prior(0.21, domain=domain)

    assert coarse["type"] == "log_uniform"
    assert coarse["lower"] == pytest.approx(1.0)
    assert coarse["upper"] == pytest.approx(120.0 / 0.21)

    fine = angular_length_prior(0.03, domain=domain)

    assert fine["lower"] == pytest.approx(0.1 / 0.03)
    assert fine["upper"] == pytest.approx(4000.0)

    assert length_domain("core_sersic", "rb") == (0.5, None)
    assert length_domain("bulge_disk_spiral", "re_disk") == (10.0, None)

    with pytest.raises(ValueError, match="empty"):
        angular_length_prior(1000.0, domain=(1.0, None))


def test_reference_length_prior_is_clipped_to_the_angular_range() -> None:
    """Multi-component lengths keep their reference ranges within the rule."""
    reference = reference_priors("bulge_disk")["re_bulge"]

    assert reference == {"type": "uniform", "lower": 1.0, "upper": 30.0}

    unchanged = clipped_reference_length_prior(reference, 0.21)

    assert unchanged == reference

    fine = clipped_reference_length_prior(reference, 0.005)

    assert fine["type"] == "uniform"
    assert fine["lower"] == pytest.approx(20.0)
    assert fine["upper"] == pytest.approx(30.0)

    with pytest.raises(ValueError, match="empty"):
        clipped_reference_length_prior(reference, 0.001)


def test_centre_priors_span_three_psf_widths() -> None:
    """The centre must lie within three FWHM of the target position."""
    priors = centre_priors(2.74)

    assert priors["x0"] == {
        "type": "uniform",
        "lower": pytest.approx(-8.22),
        "upper": pytest.approx(8.22),
        "relative_to": "target_x",
    }
    assert priors["y0"]["relative_to"] == "target_y"


def test_background_priors_scale_with_the_prepared_sky() -> None:
    """The level and a plane must stay within five sky scales."""
    constant = background_priors("constant", sky_scale=2.6, cutout_shape=(400, 600))

    assert constant == {
        "background_level": {
            "type": "uniform",
            "lower": pytest.approx(-13.0),
            "upper": pytest.approx(13.0),
        },
    }

    plane = background_priors("plane", sky_scale=2.6, cutout_shape=(400, 600))

    assert plane["background_gradient_x"]["upper"] == pytest.approx(13.0 / 300.0)
    assert plane["background_gradient_y"]["upper"] == pytest.approx(13.0 / 200.0)
    assert plane["background_gradient_x"]["lower"] == pytest.approx(-13.0 / 300.0)

    with pytest.raises(ValueError, match="positive"):
        background_priors("constant", sky_scale=0.0, cutout_shape=(10, 10))


def test_single_sersic_priors_record_their_sources() -> None:
    """Every parameter of a single-component model must have its rule."""
    proposals = propose_model_priors(
        "single_sersic",
        zeropoint=30.0,
        pixel_scale=0.21,
        psf_fwhm_pixels=2.74,
        photometric_system="AB",
    )

    assert list(proposals) == parameter_names("single_sersic")
    assert {name: proposal.source for name, proposal in proposals.items()} == {
        "flux": "header",
        "re": "header",
        "n": "reference",
        "q": "reference",
        "theta": "reference",
        "x0": "fixed_inputs",
        "y0": "fixed_inputs",
    }
    assert proposals["flux"].prior["upper"] == pytest.approx(10.0**8.8)
    assert "AB" in proposals["flux"].detail
    assert proposals["re"].prior["type"] == "log_uniform"
    assert proposals["re"].prior["lower"] == pytest.approx(1.0)
    assert proposals["n"].prior == reference_priors("single_sersic")["n"]
    assert proposals["x0"].prior["upper"] == pytest.approx(3.0 * 2.74)


def test_decided_priors_are_kept_as_declared() -> None:
    """A prior of the user or of a calibration must not be replaced."""
    declared = PriorProposal(
        parameter="re",
        kind="length",
        prior={"type": "uniform", "lower": 2.0, "upper": 40.0},
        source="user",
        detail="declared by the user",
    )

    proposals = propose_model_priors(
        "single_sersic",
        zeropoint=30.0,
        pixel_scale=0.21,
        psf_fwhm_pixels=2.74,
        decided={"re": declared},
    )

    assert proposals["re"] is declared

    with pytest.raises(ValueError, match="has no parameters"):
        propose_model_priors(
            "single_sersic",
            zeropoint=30.0,
            pixel_scale=0.21,
            psf_fwhm_pixels=2.74,
            decided={"c0": declared},
        )


def test_missing_inputs_are_reported_instead_of_defaulted() -> None:
    """Without a zeropoint the user must declare the flux prior."""
    with pytest.raises(ValueError, match="no usable zeropoint"):
        propose_model_priors(
            "single_sersic",
            zeropoint=None,
            pixel_scale=0.21,
            psf_fwhm_pixels=2.74,
        )

    without_scale = propose_model_priors(
        "single_sersic",
        zeropoint=30.0,
        pixel_scale=None,
        psf_fwhm_pixels=2.74,
    )

    assert without_scale["re"].source == "reference"
    assert without_scale["re"].prior == reference_priors("single_sersic")["re"]
    assert "no pixel scale" in without_scale["re"].detail

    with pytest.raises(ValueError, match="FWHM of the PSF is unknown"):
        propose_model_priors(
            "single_sersic",
            zeropoint=30.0,
            pixel_scale=0.21,
            psf_fwhm_pixels=None,
        )


def test_multi_component_lengths_keep_their_reference_ranges() -> None:
    """Bulge and disc must stay distinguishable through their lengths."""
    proposals = propose_model_priors(
        "bulge_disk",
        zeropoint=30.0,
        pixel_scale=0.21,
        psf_fwhm_pixels=2.74,
    )

    assert proposals["re_bulge"].source == "reference"
    assert proposals["re_bulge"].prior == {"type": "uniform", "lower": 1.0, "upper": 30.0}
    assert proposals["re_disk"].prior == {
        "type": "uniform",
        "lower": 10.0,
        "upper": 120.0,
    }
    assert proposals["flux_bulge"].source == "header"
    assert proposals["flux_disk"].prior == proposals["flux_bulge"].prior


def test_every_model_receives_a_complete_proposal() -> None:
    """All ten models must be proposable without a declared prior."""
    for model_name in PARAMETER_KINDS:
        proposals = propose_model_priors(
            model_name,
            zeropoint=30.0,
            pixel_scale=0.21,
            psf_fwhm_pixels=2.74,
        )

        assert list(proposals) == parameter_names(model_name)

        for name, proposal in proposals.items():
            if PARAMETER_KINDS[model_name][name] == "length":
                expected = (
                    "header" if model_name in PHYSICAL_LENGTH_MODELS else "reference"
                )
                assert proposal.source == expected, (model_name, name)


def test_double_sersic_axes_take_the_defaults_the_fit_shares() -> None:
    """The axes of both components take the q and theta defaults, as in a fit."""
    proposals = propose_model_priors(
        "double_sersic",
        zeropoint=30.0,
        pixel_scale=0.21,
        psf_fwhm_pixels=2.74,
    )

    for name in ("q_bulge", "q_disk"):
        assert proposals[name].prior == {"type": "uniform", "lower": 0.4, "upper": 1.0}
        assert proposals[name].source == "reference"
        assert "components of the model share" in proposals[name].detail

    for name in ("theta_bulge", "theta_disk"):
        assert proposals[name].prior == {
            "type": "uniform",
            "lower": "-pi/2",
            "upper": "pi/2",
        }

    assert proposals["n_bulge"].detail == "reference prior of BayMorph"

    # The fit reads the prior of each axis through the same shared defaults.
    fit_priors = reference_priors("double_sersic")

    for name in ("q_bulge", "q_disk", "theta_bulge", "theta_disk"):
        assert proposals[name].prior == dict(
            prior_config_for_parameter(name, fit_priors)
        )


def test_sky_level_of_the_header_gives_the_gain_of_a_coadd() -> None:
    """A subtracted sky level that passes the guard comes first."""
    reading = read_header(_coadd_header())

    proposal = propose_effective_gain(reading, sky_scale=2.598)

    assert proposal.declaration == {"source": "sky_level", "sky_level": 326.87}
    assert proposal.value == pytest.approx(326.87 / 2.598**2)
    assert proposal.candidates["gain_keyword"] == pytest.approx(34.9)
    assert proposal.ratio == pytest.approx(326.87 / 2.598**2 / 34.9)
    assert proposal.expected_ratio is None
    assert proposal.warnings == ()
    assert any("without a prediction" in note for note in proposal.notes)


def test_declared_resampling_factor_predicts_the_ratio_of_the_gains() -> None:
    """A ratio far from the declared resampling must be flagged."""
    reading = read_header(_coadd_header())

    consistent = propose_effective_gain(
        reading,
        sky_scale=2.598,
        resampling_variance_factor=0.8,
    )

    assert consistent.expected_ratio == pytest.approx(1.25)
    assert consistent.warnings == ()
    assert consistent.source == "sky_level"

    inconsistent = propose_effective_gain(
        reading,
        sky_scale=2.598,
        resampling_variance_factor=0.5,
    )

    assert inconsistent.expected_ratio == pytest.approx(2.0)
    assert len(inconsistent.warnings) == 1
    assert "resampling predicts" in inconsistent.warnings[0]

    with pytest.raises(ValueError, match="must not exceed 1"):
        propose_effective_gain(
            reading,
            sky_scale=2.598,
            resampling_variance_factor=1.5,
        )


def test_gain_keyword_of_a_resampled_image_needs_a_declared_factor() -> None:
    """Without a sky, the keyword of a resampled coadd is not proposed."""
    reading = read_header(_coadd_header(sky_level=None))

    undeclared = propose_effective_gain(reading, sky_scale=2.598)

    assert undeclared.declaration is None
    assert "No source of the effective gain qualifies" in undeclared.warnings[0]
    assert undeclared.candidates == {"gain_keyword": pytest.approx(34.9)}

    declared = propose_effective_gain(
        reading,
        sky_scale=2.598,
        resampling_variance_factor=0.8,
    )

    assert declared.declaration == {
        "source": "declared",
        "gain": pytest.approx(34.9 / 0.8),
    }
    assert declared.value == pytest.approx(34.9 / 0.8)


def test_gain_keyword_of_a_single_exposure_is_proposed() -> None:
    """An image in ADU that was not resampled takes its gain keyword."""
    reading = read_header(_single_exposure_header())

    proposal = propose_effective_gain(reading, sky_scale=5.0)

    assert proposal.declaration == {"source": "declared", "gain": 2.5}


def test_measured_sky_gives_the_gain_when_the_sky_is_in_the_image() -> None:
    """A measured sky that passes the guard is proposed after the header."""
    reading = read_header(_wcs_header())

    proposal = propose_effective_gain(
        reading,
        sky_scale=10.0,
        measured_sky_level=1000.0,
    )

    assert proposal.declaration == {"source": "measured_sky"}
    assert proposal.value == pytest.approx(10.0)

    subtracted = propose_effective_gain(
        reading,
        sky_scale=10.0,
        measured_sky_level=0.3,
    )

    assert subtracted.declaration is None


def test_psf_ceiling_is_proposed_only_for_an_established_unit() -> None:
    """Half the saturation level, and only in the units of the image."""
    vst = read_header(_coadd_header())
    peak, detail = propose_psf_max_peak(vst)

    assert peak is None
    assert "ambiguous" in detail

    swarp_header = _wcs_header()
    swarp_header["BUNIT"] = "ADU"
    swarp_header["SOFTNAME"] = "SWarp"
    swarp_header["SATURATE"] = 40000.0
    peak, detail = propose_psf_max_peak(read_header(swarp_header))

    assert peak == pytest.approx(20000.0)
    assert "SATURATE" in detail

    peak, detail = propose_psf_max_peak(read_header(_wcs_header()))

    assert peak is None
    assert "missing" in detail
