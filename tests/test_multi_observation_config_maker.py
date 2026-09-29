"""Tests for multi-observation configuration generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy.io import fits

from baymorph.pipeline.config_maker import (
    make_multi_observation_config,
)
from baymorph.pipeline.multi_observation_config import (
    build_multi_observation_parameter_layout,
    resolve_multi_observation_observation_configs,
)
from baymorph.runner import BayMorphRunner


def _write_fits(
    path: Path,
    *,
    with_wcs: bool = True,
) -> None:
    """Write a small astronomical FITS image."""
    data = np.zeros(
        (
            31,
            37,
        ),
        dtype=float,
    )

    header = fits.Header()

    if with_wcs:
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
        header["CRPIX1"] = 19.0
        header["CRPIX2"] = 16.0
        header["CRVAL1"] = 150.0
        header["CRVAL2"] = 2.0
        header["CDELT1"] = -0.2 / 3600.0
        header["CDELT2"] = 0.2 / 3600.0

    fits.writeto(
        path,
        data,
        header,
    )


def _uniform(
    lower: float,
    upper: float,
) -> dict:
    """Return one uniform-prior configuration."""
    return {
        "type": "uniform",
        "lower": lower,
        "upper": upper,
    }


def _sky_parameters(
    observation_names,
) -> dict:
    """Return an explicit single-Sersic sky-frame parameter contract."""
    names = tuple(
        observation_names
    )

    return {
        "flux": {
            "coupling": "independent",
            "priors": {
                name: _uniform(
                    1.0,
                    1.0e6,
                )
                for name in names
            },
        },
        "re_arcsec": {
            "coupling": "independent",
            "priors": {
                name: _uniform(
                    0.05,
                    30.0,
                )
                for name in names
            },
        },
        "n": {
            "coupling": "shared",
            "prior": _uniform(
                0.1,
                8.0,
            ),
        },
        "q": {
            "coupling": "shared",
            "prior": _uniform(
                0.1,
                1.0,
            ),
        },
        "position_angle": {
            "coupling": "shared",
            "prior": _uniform(
                -np.pi / 2.0,
                np.pi / 2.0,
            ),
        },
        "ra": {
            "coupling": "shared",
            "prior": _uniform(
                149.99,
                150.01,
            ),
        },
        "dec": {
            "coupling": "shared",
            "prior": _uniform(
                1.99,
                2.01,
            ),
        },
        "background_level": {
            "coupling": "independent",
            "priors": {
                name: _uniform(
                    -100.0,
                    100.0,
                )
                for name in names
            },
        },
    }


def test_multi_observation_config_maker_writes_complete_workspace(
    tmp_path: Path,
) -> None:
    """The maker writes observation YAMLs and one runnable joint YAML."""
    first_fits = (
        tmp_path
        / "image_a.fits"
    )

    second_fits = (
        tmp_path
        / "image_b.fits"
    )

    _write_fits(
        first_fits
    )
    _write_fits(
        second_fits
    )

    observation_names = (
        "image_A",
        "repeat 02",
    )

    output_dir = (
        tmp_path
        / "configs"
    )

    joint_path = (
        output_dir
        / "target_joint.yaml"
    )

    joint = make_multi_observation_config(
        {
            observation_names[
                0
            ]: first_fits,
            observation_names[
                1
            ]: second_fits,
        },
        target_positions={
            observation_names[
                0
            ]: (
                18.0,
                15.0,
            ),
            observation_names[
                1
            ]: "center",
        },
        parameters=_sky_parameters(
            observation_names
        ),
        output_dir=output_dir,
        joint_output_path=joint_path,
        analysis_name="target_joint",
        outputs_root=(
            tmp_path
            / "runs"
        ),
        observation_options={
            "repeat 02": {
                "psf_type": "gaussian",
                "gaussian_fwhm_pixels": 2.5,
            },
        },
    )

    first_config_path = (
        output_dir
        / "image_A.yaml"
    )

    second_config_path = (
        output_dir
        / "repeat_02.yaml"
    )

    assert first_config_path.is_file()
    assert second_config_path.is_file()
    assert joint_path.is_file()

    assert tuple(
        joint[
            "observations"
        ]
    ) == observation_names

    assert joint[
        "observations"
    ][
        "image_A"
    ][
        "config_path"
    ] == "image_A.yaml"

    assert joint[
        "observations"
    ][
        "repeat 02"
    ][
        "config_path"
    ] == "repeat_02.yaml"

    assert joint[
        "fit"
    ][
        "multi_observation"
    ][
        "parameter_frame"
    ] == "sky"

    assert isinstance(
        joint[
            "fit"
        ][
            "sampler"
        ],
        dict,
    )

    with first_config_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        first_config = yaml.safe_load(
            stream
        )

    with second_config_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        second_config = yaml.safe_load(
            stream
        )

    assert first_config[
        "psf"
    ][
        "type"
    ] == "empirical"

    assert second_config[
        "psf"
    ][
        "type"
    ] == "gaussian"

    assert second_config[
        "psf"
    ][
        "gaussian"
    ][
        "fwhm"
    ] == pytest.approx(
        2.5
    )

    resolved_observations = resolve_multi_observation_observation_configs(
        joint,
        base_dir=joint_path.parent,
    )

    assert tuple(
        resolved_observations
    ) == observation_names

    layout = build_multi_observation_parameter_layout(
        joint
    )

    assert layout.observations == observation_names

    assert "flux__image_A" in layout.parameter_set.names
    assert "flux__repeat 02" in layout.parameter_set.names
    assert "re_arcsec__image_A" in layout.parameter_set.names
    assert "re_arcsec__repeat 02" in layout.parameter_set.names

    runner = BayMorphRunner(
        joint_path,
        verbose=False,
    )

    assert runner.is_multi_observation is True


def test_multi_observation_config_maker_defaults_to_no_overwrite(
    tmp_path: Path,
) -> None:
    """Per-observation configs must not silently allow overwriting products."""
    first_fits = (
        tmp_path
        / "image_a.fits"
    )
    second_fits = (
        tmp_path
        / "image_b.fits"
    )

    _write_fits(
        first_fits
    )
    _write_fits(
        second_fits
    )

    observation_names = (
        "a",
        "b",
    )

    output_dir = (
        tmp_path
        / "configs"
    )

    make_multi_observation_config(
        {
            observation_names[
                0
            ]: first_fits,
            observation_names[
                1
            ]: second_fits,
        },
        target_positions={
            name: "center"
            for name in observation_names
        },
        parameters=_sky_parameters(
            observation_names
        ),
        output_dir=output_dir,
        outputs_root=(
            tmp_path
            / "runs"
        ),
    )

    for name in observation_names:
        with (
            output_dir
            / f"{name}.yaml"
        ).open(
            "r",
            encoding="utf-8",
        ) as stream:
            observation_config = yaml.safe_load(
                stream
            )

        assert observation_config[
            "outputs"
        ][
            "overwrite"
        ] is False


def test_multi_observation_config_requires_exact_target_mapping(
    tmp_path: Path,
) -> None:
    """Every observation must receive an explicit target seed."""
    first_fits = (
        tmp_path
        / "a.fits"
    )

    second_fits = (
        tmp_path
        / "b.fits"
    )

    _write_fits(
        first_fits
    )
    _write_fits(
        second_fits
    )

    output_dir = (
        tmp_path
        / "configs"
    )

    with pytest.raises(
        ValueError,
        match="target_positions",
    ):
        make_multi_observation_config(
            {
                "a": first_fits,
                "b": second_fits,
            },
            target_positions={
                "a": "center",
            },
            parameters=_sky_parameters(
                (
                    "a",
                    "b",
                )
            ),
            output_dir=output_dir,
        )

    assert not output_dir.exists()


def test_multi_observation_config_rejects_incomplete_parameter_contract(
    tmp_path: Path,
) -> None:
    """The maker must not invent missing priors or coupling decisions."""
    first_fits = (
        tmp_path
        / "a.fits"
    )

    second_fits = (
        tmp_path
        / "b.fits"
    )

    _write_fits(
        first_fits
    )
    _write_fits(
        second_fits
    )

    parameters = _sky_parameters(
        (
            "a",
            "b",
        )
    )

    del parameters[
        "background_level"
    ]

    output_dir = (
        tmp_path
        / "configs"
    )

    with pytest.raises(
        ValueError,
        match="Missing",
    ):
        make_multi_observation_config(
            {
                "a": first_fits,
                "b": second_fits,
            },
            target_positions={
                "a": "center",
                "b": "center",
            },
            parameters=parameters,
            output_dir=output_dir,
        )

    assert not output_dir.exists()


def test_sky_frame_config_maker_requires_wcs(
    tmp_path: Path,
) -> None:
    """Sky-frame joint configurations fail early when WCS is unavailable."""
    first_fits = (
        tmp_path
        / "a.fits"
    )

    second_fits = (
        tmp_path
        / "b.fits"
    )

    _write_fits(
        first_fits
    )
    _write_fits(
        second_fits,
        with_wcs=False,
    )

    output_dir = (
        tmp_path
        / "configs"
    )

    with pytest.raises(
        ValueError,
        match="valid WCS.*'b'",
    ):
        make_multi_observation_config(
            {
                "a": first_fits,
                "b": second_fits,
            },
            target_positions={
                "a": "center",
                "b": "center",
            },
            parameters=_sky_parameters(
                (
                    "a",
                    "b",
                )
            ),
            output_dir=output_dir,
        )

    assert not output_dir.exists()


def test_multi_observation_config_rejects_filename_collisions(
    tmp_path: Path,
) -> None:
    """Opaque identifiers must not silently overwrite YAML files."""
    first_fits = (
        tmp_path
        / "a.fits"
    )

    second_fits = (
        tmp_path
        / "b.fits"
    )

    _write_fits(
        first_fits
    )
    _write_fits(
        second_fits
    )

    observation_names = (
        "visit 01",
        "visit_01",
    )

    with pytest.raises(
        ValueError,
        match="colliding",
    ):
        make_multi_observation_config(
            {
                observation_names[
                    0
                ]: first_fits,
                observation_names[
                    1
                ]: second_fits,
            },
            target_positions={
                observation_names[
                    0
                ]: "center",
                observation_names[
                    1
                ]: "center",
            },
            parameters=_sky_parameters(
                observation_names
            ),
            output_dir=(
                tmp_path
                / "configs"
            ),
        )


def test_multi_observation_config_accepts_sampler_name(
    tmp_path: Path,
) -> None:
    """The joint Config Maker must share the single-observation sampler API."""
    first_fits = tmp_path / "first.fits"
    second_fits = tmp_path / "second.fits"

    _write_fits(
        first_fits
    )
    _write_fits(
        second_fits
    )

    observations = {
        "first": first_fits,
        "second": second_fits,
    }

    joint = make_multi_observation_config(
        observations,
        target_positions={
            "first": "center",
            "second": "center",
        },
        parameters=_sky_parameters(
            observations
        ),
        output_dir=tmp_path / "configs",
        model_name="single_sersic",
        sampler="dynesty",
        outputs_root=tmp_path / "runs",
    )

    assert joint["fit"]["models"] == [
        "single_sersic",
    ]

    assert joint["fit"]["sampler"] == {
        "name": "dynesty",
    }

    for name in observations:
        config_path = (
            tmp_path
            / "configs"
            / f"{name}.yaml"
        )

        with config_path.open(
            "r",
            encoding="utf-8",
        ) as stream:
            observation_config = yaml.safe_load(
                stream
            )

        assert observation_config["fit"]["models"] == [
            "single_sersic",
        ]

        assert observation_config["fit"]["sampler"] == {
            "name": "dynesty",
        }


def test_multi_observation_config_resolves_likelihood_per_observation(
    tmp_path: Path,
) -> None:
    """Global likelihood applies unless an observation overrides it."""
    first_fits = tmp_path / "g.fits"
    second_fits = tmp_path / "r.fits"

    _write_fits(
        first_fits
    )
    _write_fits(
        second_fits
    )

    observations = {
        "g": first_fits,
        "r": second_fits,
    }

    global_likelihood = {
        "type": "correlated_gaussian",
        "include_normalization": False,
    }
    observation_override = {
        "type": "gaussian",
        "include_normalization": True,
    }

    output_dir = tmp_path / "configs"

    joint = make_multi_observation_config(
        observations,
        target_positions={
            "g": "center",
            "r": "center",
        },
        parameters=_sky_parameters(
            observations
        ),
        output_dir=output_dir,
        likelihood=global_likelihood,
        outputs_root=tmp_path / "runs",
        observation_options={
            "r": {
                "likelihood": observation_override,
            },
        },
    )

    assert joint["fit"]["likelihood"] == global_likelihood

    resolved = resolve_multi_observation_observation_configs(
        joint,
        base_dir=output_dir,
    )

    assert resolved["g"]["fit"]["likelihood"] == global_likelihood
    assert resolved["r"]["fit"]["likelihood"] == observation_override
