"""Tests for typed priors in single-observation inference."""

from __future__ import annotations

import numpy as np

from baymorph.fitting.priors import UniformPrior
from baymorph.pipeline.real_fit import _build_parameter_set


def _typed_single_sersic_priors() -> dict:
    """Return explicit data-independent Single-Sérsic priors."""
    return {
        "single_sersic": {
            "flux": {
                "type": "uniform",
                "lower": 100.0,
                "upper": 1000.0,
            },
            "re": {
                "type": "uniform",
                "lower": 1.0,
                "upper": 20.0,
            },
            "n": {
                "type": "uniform",
                "lower": 0.5,
                "upper": 8.0,
            },
            "q": {
                "type": "uniform",
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
                "lower": -4.0,
                "upper": 4.0,
                "relative_to": "target_x",
            },
            "y0": {
                "type": "uniform",
                "lower": -5.0,
                "upper": 5.0,
                "relative_to": "target_y",
            },
        },
    }


def _typed_background() -> dict:
    """Return an explicit constant-background prior."""
    return {
        "model": "constant",
        "priors": {
            "background_level": {
                "type": "uniform",
                "lower": -10.0,
                "upper": 10.0,
            },
        },
    }


def test_typed_single_band_priors_are_independent_of_image_values() -> None:
    """Image intensities must not alter explicitly configured typed priors."""
    shape = (
        21,
        21,
    )

    common = {
        "model_name": "single_sersic",
        "noise_fit": np.ones(
            shape,
            dtype=float,
        ),
        "mask_fit": np.zeros(
            shape,
            dtype=bool,
        ),
        "target_x": 10.0,
        "target_y": 11.0,
        "prior_cfg": _typed_single_sersic_priors(),
        "background_cfg": _typed_background(),
    }

    _, parameters_low, _, _ = _build_parameter_set(
        data_fit=np.zeros(
            shape,
            dtype=float,
        ),
        **common,
    )

    _, parameters_high, _, _ = _build_parameter_set(
        data_fit=np.full(
            shape,
            1.0e9,
            dtype=float,
        ),
        **common,
    )

    assert [
        parameter.prior
        for parameter in parameters_low.all_parameters
    ] == [
        parameter.prior
        for parameter in parameters_high.all_parameters
    ]


def test_relative_centroid_priors_resolve_from_explicit_reference() -> None:
    """Relative centroid priors must resolve around the supplied reference."""
    shape = (
        21,
        21,
    )

    _, parameters, _, _ = _build_parameter_set(
        "single_sersic",
        data_fit=np.ones(
            shape,
            dtype=float,
        ),
        noise_fit=np.ones(
            shape,
            dtype=float,
        ),
        mask_fit=np.zeros(
            shape,
            dtype=bool,
        ),
        target_x=100.0,
        target_y=200.0,
        prior_cfg=_typed_single_sersic_priors(),
        background_cfg=_typed_background(),
    )

    by_name = {
        parameter.name: parameter
        for parameter in parameters.all_parameters
    }

    assert by_name["x0"].prior == UniformPrior(
        lower=96.0,
        upper=104.0,
    )

    assert by_name["y0"].prior == UniformPrior(
        lower=195.0,
        upper=205.0,
    )


def test_typed_flux_prior_is_absolute() -> None:
    """Typed total-flux priors must not depend on an image-derived scale."""
    shape = (
        21,
        21,
    )

    _, parameters, _, _ = _build_parameter_set(
        "single_sersic",
        data_fit=np.full(
            shape,
            1.0e12,
            dtype=float,
        ),
        noise_fit=np.ones(
            shape,
            dtype=float,
        ),
        mask_fit=np.zeros(
            shape,
            dtype=bool,
        ),
        target_x=10.0,
        target_y=10.0,
        prior_cfg=_typed_single_sersic_priors(),
        background_cfg=_typed_background(),
    )

    flux = next(
        parameter
        for parameter in parameters.all_parameters
        if parameter.name == "flux"
    )

    assert flux.prior == UniformPrior(
        lower=100.0,
        upper=1000.0,
    )
