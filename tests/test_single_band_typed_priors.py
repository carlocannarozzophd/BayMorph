"""Tests for typed priors in single-observation inference."""

from __future__ import annotations

import inspect

import pytest

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
                "lower": "-pi/2",
                "upper": "pi/2",
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


def _references() -> dict[str, float]:
    """Return deterministic target-seed references in cutout coordinates."""
    return {
        "target_x": 100.0,
        "target_y": 200.0,
    }


def test_parameter_builder_has_no_image_inputs() -> None:
    """Scientific prior construction must be structurally independent of pixels."""
    signature = inspect.signature(
        _build_parameter_set
    )

    forbidden = {
        "data_fit",
        "noise_fit",
        "mask_fit",
        "flux_total",
        "background_guess",
        "target_x",
        "target_y",
    }

    assert forbidden.isdisjoint(
        signature.parameters
    )


def test_relative_centroid_priors_resolve_from_explicit_reference() -> None:
    """Relative centroid priors must resolve around the supplied seed."""
    _, parameters = _build_parameter_set(
        "single_sersic",
        prior_cfg=_typed_single_sersic_priors(),
        background_cfg=_typed_background(),
        references=_references(),
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
    """Total-flux support must be exactly the explicitly configured support."""
    _, parameters = _build_parameter_set(
        "single_sersic",
        prior_cfg=_typed_single_sersic_priors(),
        background_cfg=_typed_background(),
        references=_references(),
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


def test_missing_flux_prior_fails_explicitly() -> None:
    """A missing non-universal flux prior must not be silently synthesized."""
    priors = _typed_single_sersic_priors()
    del priors[
        "single_sersic"
    ][
        "flux"
    ]

    with pytest.raises(
        ValueError,
        match="Missing prior configuration for parameter: flux",
    ):
        _build_parameter_set(
            "single_sersic",
            prior_cfg=priors,
            background_cfg=_typed_background(),
            references=_references(),
        )
