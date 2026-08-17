"""Tests for typed default priors in multi-observation parameter construction."""

from __future__ import annotations

from copy import deepcopy

import baymorph.pipeline.multi_parameters as multi_parameters


def _typed_defaults() -> dict:
    """Return typed detector-space defaults for a Single-Sérsic model."""
    return {
        "flux": {
            "type": "uniform",
            "lower": 100.0,
            "upper": 1000.0,
        },
        "re": {
            "type": "uniform",
            "lower": 1.0,
            "upper": 40.0,
        },
        "n": {
            "type": "uniform",
            "lower": 0.5,
            "upper": 8.0,
        },
        "q": {
            "type": "beta",
            "alpha": 2.0,
            "beta": 2.0,
            "lower": 0.1,
            "upper": 1.0,
        },
        "theta": {
            "type": "uniform",
            "lower": -1.5707963267948966,
            "upper": 1.5707963267948966,
        },
        "x0": {
            "type": "uniform",
            "lower": -8.0,
            "upper": 8.0,
            "relative_to": "target_x",
        },
        "y0": {
            "type": "uniform",
            "lower": -6.0,
            "upper": 6.0,
            "relative_to": "target_y",
        },
        "sky": {
            "type": "uniform",
            "lower": -10.0,
            "upper": 10.0,
        },
    }


def test_detector_frame_accepts_typed_default_priors(
    monkeypatch,
) -> None:
    """Detector-frame construction must consume typed model defaults."""
    config = deepcopy(
        multi_parameters.DEFAULT_CONFIG
    )

    config["fit"]["priors"]["single_sersic"] = (
        _typed_defaults()
    )

    monkeypatch.setattr(
        multi_parameters,
        "DEFAULT_CONFIG",
        config,
    )

    parameters = multi_parameters.make_multi_parameters(
        "single_sersic",
        {
            "g": "unused_g.fits",
            "r": "unused_r.fits",
        },
        target_centers={
            "g": (
                100.0,
                200.0,
            ),
            "r": (
                110.0,
                210.0,
            ),
        },
        shared=[
            "q",
        ],
        parameter_frame="detector",
    )

    assert parameters["q"] == {
        "coupling": "shared",
        "prior": {
            "type": "beta",
            "alpha": 2.0,
            "beta": 2.0,
            "lower": 0.1,
            "upper": 1.0,
        },
    }

    assert parameters["x0"] == {
        "coupling": "independent",
        "priors": {
            "g": {
                "type": "uniform",
                "lower": 92.0,
                "upper": 108.0,
            },
            "r": {
                "type": "uniform",
                "lower": 102.0,
                "upper": 118.0,
            },
        },
    }

    assert parameters["y0"] == {
        "coupling": "independent",
        "priors": {
            "g": {
                "type": "uniform",
                "lower": 194.0,
                "upper": 206.0,
            },
            "r": {
                "type": "uniform",
                "lower": 204.0,
                "upper": 216.0,
            },
        },
    }


def test_shared_relative_detector_prior_spans_all_target_seeds(
    monkeypatch,
) -> None:
    """A shared detector centre must cover every explicit target seed."""
    config = deepcopy(
        multi_parameters.DEFAULT_CONFIG
    )

    config["fit"]["priors"]["single_sersic"] = (
        _typed_defaults()
    )

    monkeypatch.setattr(
        multi_parameters,
        "DEFAULT_CONFIG",
        config,
    )

    parameters = multi_parameters.make_multi_parameters(
        "single_sersic",
        {
            "g": "unused_g.fits",
            "r": "unused_r.fits",
        },
        target_centers={
            "g": (
                100.0,
                200.0,
            ),
            "r": (
                110.0,
                210.0,
            ),
        },
        shared=[
            "x0",
        ],
        parameter_frame="detector",
    )

    assert parameters["x0"] == {
        "coupling": "shared",
        "prior": {
            "type": "uniform",
            "lower": 92.0,
            "upper": 118.0,
        },
    }
