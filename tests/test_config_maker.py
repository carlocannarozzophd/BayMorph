"""Tests for FITS-to-YAML BayMorph configuration generation."""

from __future__ import annotations

from numbers import Real

import numpy as np
import pytest
import yaml
from astropy.io import fits

from baymorph.pipeline.config import (
    resolve_config,
)
from baymorph.pipeline.config_maker import (
    make_config_from_fits as _make_config_from_fits,
)



_TEST_FLUX_PARAMETERS = {
    "single_sersic": (
        "flux",
    ),
    "double_sersic": (
        "flux_bulge",
        "flux_disk",
    ),
    "sersic_point_source": (
        "flux_sersic",
        "flux_point_source",
    ),
    "bulge_disk": (
        "flux_bulge",
        "flux_disk",
    ),
}


def make_config_from_fits(
    *args,
    **kwargs,
):
    """Call the production maker with explicit scientific test flux priors."""
    if "model_priors" not in kwargs:
        model_name = kwargs.get(
            "model_name",
            "single_sersic",
        )

        flux_parameters = _TEST_FLUX_PARAMETERS.get(
            model_name
        )

        if flux_parameters is not None:
            kwargs[
                "model_priors"
            ] = {
                parameter_name: {
                    "type": "uniform",
                    "lower": 1.0,
                    "upper": 1000.0,
                }
                for parameter_name in flux_parameters
            }

    return _make_config_from_fits(
        *args,
        **kwargs,
    )


def _write_test_fits(
    path,
    *,
    with_metadata: bool = True,
):
    data = np.zeros(
        (
            21,
            31,
        ),
        dtype=float,
    )

    header = fits.Header()

    if with_metadata:
        header[
            "CTYPE1"
        ] = "RA---TAN"
        header[
            "CTYPE2"
        ] = "DEC--TAN"
        header[
            "CRPIX1"
        ] = 16.0
        header[
            "CRPIX2"
        ] = 11.0
        header[
            "CRVAL1"
        ] = 150.0
        header[
            "CRVAL2"
        ] = 2.0
        header[
            "CDELT1"
        ] = -0.2 / 3600.0
        header[
            "CDELT2"
        ] = 0.2 / 3600.0
        header[
            "FWHM"
        ] = 0.8
        header[
            "FWHMUNIT"
        ] = "arcsec"
        header[
            "SATLEVEL"
        ] = 50000.0

    fits.writeto(
        path,
        data,
        header,
    )


def test_config_maker_builds_minimal_runnable_config(
    tmp_path,
):
    fits_path = (
        tmp_path
        / "galaxy_A.fits"
    )

    yaml_path = (
        tmp_path
        / "configs"
        / "galaxy_A.yaml"
    )

    _write_test_fits(
        fits_path
    )

    config = make_config_from_fits(
        fits_path,
        target_position=(
            15.0,
            10.0,
        ),
        output_path=yaml_path,
    )

    assert config[
        "object"
    ][
        "name"
    ] == "galaxy_A"

    assert config[
        "data"
    ][
        "image_path"
    ] == str(
        fits_path
    )

    assert config[
        "data"
    ][
        "pixel_scale"
    ] == pytest.approx(
        0.2
    )

    assert config[
        "data"
    ][
        "target_position"
    ] == {
        "x": 15.0,
        "y": 10.0,
    }

    assert config[
        "data"
    ][
        "cutout"
    ][
        "mode"
    ] == "auto"

    assert config[
        "psf"
    ][
        "type"
    ] == "empirical"

    candidate = config[
        "psf"
    ][
        "empirical"
    ][
        "candidate_selection"
    ]

    assert candidate[
        "auto"
    ] is True

    assert "max_peak" not in candidate

    assert "min_half_size" not in candidate
    assert "max_half_size" not in candidate

    resolved = resolve_config(
        config
    )

    assert resolved[
        "fit"
    ][
        "models"
    ]

    assert resolved[
        "diagnostics"
    ][
        "corner"
    ] is True

    assert yaml_path.exists()

    with yaml_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        loaded = yaml.safe_load(
            stream
        )

    assert loaded == config


def test_config_maker_defaults_to_no_overwrite(
    tmp_path,
):
    """Generated configs must not silently allow overwriting run products."""
    fits_path = (
        tmp_path
        / "galaxy_A.fits"
    )

    _write_test_fits(
        fits_path
    )

    config = make_config_from_fits(
        fits_path,
        target_position=(
            15.0,
            10.0,
        ),
    )

    assert config[
        "outputs"
    ][
        "overwrite"
    ] is False


def test_config_maker_overwrite_is_explicit_opt_in(
    tmp_path,
):
    """Overwrite must remain available as an explicit opt-in."""
    fits_path = (
        tmp_path
        / "galaxy_A.fits"
    )

    _write_test_fits(
        fits_path
    )

    config = make_config_from_fits(
        fits_path,
        target_position=(
            15.0,
            10.0,
        ),
        overwrite=True,
    )

    assert config[
        "outputs"
    ][
        "overwrite"
    ] is True


def test_config_maker_empirical_psf_resolves_numeric_candidate_limits(
    tmp_path,
):
    """Generated empirical-PSF thresholds must be numerically consumable."""
    fits_path = tmp_path / "galaxy.fits"

    _write_test_fits(
        fits_path
    )

    config = make_config_from_fits(
        fits_path,
        target_position="center",
    )

    resolved = resolve_config(
        config
    )

    candidate = resolved[
        "psf"
    ][
        "empirical"
    ][
        "candidate_selection"
    ]

    numeric_keys = (
        "target_n_candidates",
        "min_area",
        "max_area",
        "min_peak",
        "max_peak",
        "max_eccentricity",
        "max_elongation",
        "edge_margin",
        "sigma_factor",
        "size_margin",
        "min_half_size",
        "max_half_size",
        "isolation_margin",
    )

    for key in numeric_keys:
        assert isinstance(
            candidate[key],
            Real,
        ), (
            f"Resolved PSF candidate field {key!r} must be numeric, "
            f"got {candidate[key]!r}."
        )

    assert candidate["max_peak"] == pytest.approx(
        80000.0
    )

    assert candidate["min_half_size"] > 0
    assert (
        candidate["max_half_size"]
        >= candidate["min_half_size"]
    )


def test_config_maker_center_is_explicit(
    tmp_path,
):
    fits_path = (
        tmp_path
        / "centered.fits"
    )

    _write_test_fits(
        fits_path
    )

    config = make_config_from_fits(
        fits_path,
        target_position="center",
    )

    assert config[
        "data"
    ][
        "target_position"
    ] == "center"


def test_config_maker_requires_target_seed(
    tmp_path,
):
    fits_path = (
        tmp_path
        / "galaxy.fits"
    )

    _write_test_fits(
        fits_path
    )

    with pytest.raises(
        ValueError,
        match="target_position is required",
    ):
        make_config_from_fits(
            fits_path
        )


def test_config_maker_fixed_cutout_requires_size(
    tmp_path,
):
    fits_path = (
        tmp_path
        / "galaxy.fits"
    )

    _write_test_fits(
        fits_path
    )

    with pytest.raises(
        ValueError,
        match="fixed_size is required",
    ):
        make_config_from_fits(
            fits_path,
            target_position="center",
            cutout_mode="fixed",
        )


def test_config_maker_builds_fixed_cutout(
    tmp_path,
):
    fits_path = (
        tmp_path
        / "galaxy.fits"
    )

    _write_test_fits(
        fits_path
    )

    config = make_config_from_fits(
        fits_path,
        target_position="center",
        cutout_mode="fixed",
        fixed_size=17,
    )

    assert config[
        "data"
    ][
        "cutout"
    ] == {
        "mode": "fixed",
        "size": 17,
    }


def test_config_maker_can_derive_gaussian_psf_pixels_from_header(
    tmp_path,
):
    fits_path = (
        tmp_path
        / "galaxy.fits"
    )

    _write_test_fits(
        fits_path
    )

    config = make_config_from_fits(
        fits_path,
        target_position="center",
        psf_type="gaussian",
    )

    assert config[
        "psf"
    ][
        "gaussian"
    ][
        "fwhm"
    ] == pytest.approx(
        4.0
    )


def test_config_maker_gaussian_psf_requires_scale_information(
    tmp_path,
):
    fits_path = (
        tmp_path
        / "galaxy.fits"
    )

    _write_test_fits(
        fits_path,
        with_metadata=False,
    )

    with pytest.raises(
        ValueError,
        match="Gaussian PSF generation requires",
    ):
        make_config_from_fits(
            fits_path,
            target_position="center",
            psf_type="gaussian",
        )


def test_config_maker_accepts_explicit_gaussian_fwhm(
    tmp_path,
):
    fits_path = (
        tmp_path
        / "galaxy.fits"
    )

    _write_test_fits(
        fits_path,
        with_metadata=False,
    )

    config = make_config_from_fits(
        fits_path,
        target_position="center",
        psf_type="gaussian",
        gaussian_fwhm_pixels=2.75,
    )

    assert config[
        "psf"
    ][
        "gaussian"
    ][
        "fwhm"
    ] == pytest.approx(
        2.75
    )


def test_config_maker_exposes_model_and_sampler(
    tmp_path,
):
    """Public scientific model and sampler choices must be explicit."""
    fits_path = tmp_path / "galaxy.fits"

    _write_test_fits(
        fits_path
    )

    config = make_config_from_fits(
        fits_path,
        target_position="center",
        model_name="single_sersic",
        sampler="dynesty",
    )

    assert config["fit"]["models"] == [
        "single_sersic",
    ]

    assert config["fit"]["sampler"] == {
        "name": "dynesty",
    }

    resolved = resolve_config(
        config
    )

    assert resolved["fit"]["models"] == [
        "single_sersic",
    ]

    assert (
        resolved["fit"]["sampler"]["name"]
        == "dynesty"
    )


def test_config_maker_defaults_to_single_sersic_and_nautilus(
    tmp_path,
):
    """The simple Config Maker workflow must expose stable defaults."""
    fits_path = tmp_path / "galaxy.fits"

    _write_test_fits(
        fits_path
    )

    config = make_config_from_fits(
        fits_path,
        target_position="center",
    )

    assert config["fit"]["models"] == [
        "single_sersic",
    ]

    assert config["fit"]["sampler"] == {
        "name": "nautilus",
    }


def test_config_maker_accepts_advanced_sampler_mapping(
    tmp_path,
):
    """Advanced sampler configuration must remain available."""
    fits_path = tmp_path / "galaxy.fits"

    _write_test_fits(
        fits_path
    )

    config = make_config_from_fits(
        fits_path,
        target_position="center",
        sampler={
            "name": "dynesty",
            "mode": "dynamic",
        },
    )

    assert config["fit"]["sampler"] == {
        "name": "dynesty",
        "mode": "dynamic",
    }


def test_config_maker_rejects_invalid_sampler_type(
    tmp_path,
):
    """Sampler specifications must use the public string-or-mapping contract."""
    fits_path = tmp_path / "galaxy.fits"

    _write_test_fits(
        fits_path
    )

    with pytest.raises(
        TypeError,
        match="sampler must be a string, mapping, or None",
    ):
        make_config_from_fits(
            fits_path,
            target_position="center",
            sampler=3,
        )


def test_config_maker_accepts_explicit_likelihood(
    tmp_path,
):
    """Public config generation must support explicit likelihood choices."""
    fits_path = tmp_path / "galaxy.fits"

    _write_test_fits(
        fits_path
    )

    config = make_config_from_fits(
        fits_path,
        target_position="center",
        likelihood={
            "type": "correlated_gaussian",
            "include_normalization": False,
        },
    )

    assert config["fit"]["likelihood"] == {
        "type": "correlated_gaussian",
        "include_normalization": False,
    }

    assert "region" not in config["fit"]

    resolved = resolve_config(
        config
    )

    assert (
        resolved["fit"]["likelihood"]["type"]
        == "correlated_gaussian"
    )

    assert "region" not in resolved["fit"]


def test_config_maker_uses_pixel_header_fwhm_directly(tmp_path):
    """A pixel-valued FITS FWHM must not be divided by pixel scale."""
    fits_path = tmp_path / "gaussian_pixels.fits"
    _write_test_fits(fits_path)

    with fits.open(fits_path, mode="update") as hdul:
        hdul[0].header["FWHM"] = 4.0
        hdul[0].header["FWHMUNIT"] = "pixels"

    config = make_config_from_fits(
        fits_path,
        target_position="center",
        psf_type="gaussian",
    )

    assert config["psf"]["gaussian"]["fwhm"] == pytest.approx(4.0)


def test_config_maker_rejects_ambiguous_header_fwhm(tmp_path):
    """A unitless FITS FWHM must require an explicit pixel override."""
    fits_path = tmp_path / "gaussian_ambiguous.fits"
    _write_test_fits(fits_path)

    with fits.open(fits_path, mode="update") as hdul:
        del hdul[0].header["FWHMUNIT"]

    with pytest.raises(
        ValueError,
        match="FITS FWHM unit is unresolved",
    ):
        make_config_from_fits(
            fits_path,
            target_position="center",
            psf_type="gaussian",
        )


def test_config_maker_gaussian_override_has_priority(tmp_path):
    """An explicit Gaussian FWHM must override FITS metadata."""
    fits_path = tmp_path / "gaussian_override.fits"
    _write_test_fits(fits_path)

    config = make_config_from_fits(
        fits_path,
        target_position="center",
        psf_type="gaussian",
        gaussian_fwhm_pixels=2.75,
    )

    assert config["psf"]["gaussian"]["fwhm"] == pytest.approx(2.75)


_CONSTANT_BACKGROUND = {
    "model": "constant",
    "priors": {
        "background_level": {
            "type": "uniform",
            "lower": -5.0,
            "upper": 5.0,
        },
    },
}


def test_config_maker_writes_declared_background(tmp_path):
    """A declared background must reach fit.background unchanged."""
    fits_path = tmp_path / "galaxy.fits"
    _write_test_fits(fits_path)

    config = make_config_from_fits(
        fits_path,
        target_position="center",
        background=_CONSTANT_BACKGROUND,
    )

    assert config["fit"]["background"] == _CONSTANT_BACKGROUND

    resolved = resolve_config(config)

    assert resolved["fit"]["background"]["model"] == "constant"
    assert resolved["fit"]["background"]["priors"]["background_level"] == (
        _CONSTANT_BACKGROUND["priors"]["background_level"]
    )


def test_config_maker_accepts_complete_plane_background(tmp_path):
    """A plane background declares its level and both gradients."""
    fits_path = tmp_path / "galaxy.fits"
    _write_test_fits(fits_path)

    gradient = {
        "type": "uniform",
        "lower": -0.01,
        "upper": 0.01,
    }
    background = {
        "model": "Plane",
        "priors": {
            "background_level": _CONSTANT_BACKGROUND["priors"][
                "background_level"
            ],
            "background_gradient_x": gradient,
            "background_gradient_y": gradient,
        },
    }

    config = make_config_from_fits(
        fits_path,
        target_position="center",
        background=background,
    )

    assert config["fit"]["background"]["model"] == "plane"
    assert set(config["fit"]["background"]["priors"]) == {
        "background_level",
        "background_gradient_x",
        "background_gradient_y",
    }


@pytest.mark.parametrize(
    ("background", "error", "match"),
    [
        (
            [("model", "constant")],
            TypeError,
            "background must be a mapping",
        ),
        (
            {"model": "constant"},
            ValueError,
            "both 'model' and 'priors'",
        ),
        (
            {**_CONSTANT_BACKGROUND, "region": "all"},
            ValueError,
            "Unsupported background entries",
        ),
        (
            {"model": "polynomial", "priors": {}},
            ValueError,
            "Unsupported background model",
        ),
        (
            {
                "model": "constant",
                "priors": {
                    "sky": {"type": "uniform", "lower": -1.0, "upper": 1.0},
                },
            },
            ValueError,
            "'background_level'",
        ),
        (
            {
                "model": "plane",
                "priors": _CONSTANT_BACKGROUND["priors"],
            },
            ValueError,
            "requires priors for exactly",
        ),
        (
            {
                "model": "constant",
                "priors": {
                    "background_level": {
                        "type": "uniform",
                        "lower": 5.0,
                        "upper": -5.0,
                    },
                },
            },
            ValueError,
            None,
        ),
    ],
)
def test_config_maker_rejects_incomplete_or_invalid_background(
    tmp_path,
    background,
    error,
    match,
):
    """Background declarations must be complete, typed, and non-legacy."""
    fits_path = tmp_path / "galaxy.fits"
    _write_test_fits(fits_path)

    with pytest.raises(error, match=match):
        make_config_from_fits(
            fits_path,
            target_position="center",
            background=background,
        )


def test_config_maker_writes_declared_psf_max_peak(tmp_path):
    """A declared PSF peak ceiling must replace the default and serialize."""
    fits_path = tmp_path / "galaxy.fits"
    _write_test_fits(fits_path)
    output_path = tmp_path / "configs" / "galaxy.yaml"

    config = make_config_from_fits(
        fits_path,
        target_position="center",
        psf_max_peak=np.float64(20000.0),
        output_path=output_path,
    )

    candidate = config["psf"]["empirical"]["candidate_selection"]

    assert candidate["max_peak"] == pytest.approx(20000.0)
    assert type(candidate["max_peak"]) is float

    written = yaml.safe_load(output_path.read_text(encoding="utf-8"))

    assert written["psf"]["empirical"]["candidate_selection"]["max_peak"] == (
        pytest.approx(20000.0)
    )
    assert resolve_config(config)["psf"]["empirical"]["candidate_selection"][
        "max_peak"
    ] == pytest.approx(20000.0)


def test_config_maker_rejects_psf_max_peak_for_gaussian_psf(tmp_path):
    """A Gaussian PSF selects no candidates, so a peak ceiling is an error."""
    fits_path = tmp_path / "galaxy.fits"
    _write_test_fits(fits_path)

    with pytest.raises(ValueError, match="only to psf_type='empirical'"):
        make_config_from_fits(
            fits_path,
            target_position="center",
            psf_type="gaussian",
            psf_max_peak=20000.0,
        )


@pytest.mark.parametrize(
    ("max_peak", "error", "match"),
    [
        (True, TypeError, "real number"),
        ("20000", TypeError, "real number"),
        (0.0, ValueError, "finite positive"),
        (float("inf"), ValueError, "finite positive"),
    ],
)
def test_config_maker_rejects_invalid_psf_max_peak(
    tmp_path,
    max_peak,
    error,
    match,
):
    """Only finite positive numbers are valid PSF peak ceilings."""
    fits_path = tmp_path / "galaxy.fits"
    _write_test_fits(fits_path)

    with pytest.raises(error, match=match):
        make_config_from_fits(
            fits_path,
            target_position="center",
            psf_max_peak=max_peak,
        )
