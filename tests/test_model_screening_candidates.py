"""Tests that the model screening keeps its four candidates as the registry grows."""

from __future__ import annotations

import pytest

from baymorph.analysis.model_screening import (
    SCREENING_MODELS,
    ModelScreeningReport,
)
from baymorph.models import registered_models
from baymorph.models.definition import ModelDefinition
from baymorph.models.registry import _MODEL_REGISTRY
from baymorph.validation.model_screening_dataset import (
    controlled_parameters_from_latents,
)
from baymorph.validation.model_screening_simulation import (
    simulate_model_screening_case,
)


LATENTS = {
    "total_flux": 2000.0,
    "outer_re_pixels": 6.0,
    "sersic_n": 2.5,
    "inner_n": 4.0,
    "outer_n": 1.7,
    "outer_q": 0.7,
    "inner_q": 0.9,
    "position_angle_radians": 0.2,
    "component_angle_offset_radians": 0.1,
    "inner_flux_fraction": 0.35,
    "inner_re_fraction": 0.4,
    "point_source_fraction": 0.15,
    "centroid_x_offset_pixels": 0.25,
    "centroid_y_offset_pixels": -0.25,
    "background_level": 3.0,
    "psf_fwhm_pixels": 2.0,
    "noise_sigma": 1.5,
}


@pytest.fixture
def extra_registered_model(monkeypatch) -> str:
    """Register one model the screening was not built for."""
    name = "test_only_model"
    monkeypatch.setitem(
        _MODEL_REGISTRY,
        name,
        ModelDefinition(
            name=name,
            parameter_names=["flux"],
            parameter_labels=[r"$F$"],
            factory_name="SingleSersicModelFactory",
        ),
    )

    return name


def test_screening_models_are_the_four_trained_models() -> None:
    """The candidates are the four models the screening was trained on."""
    assert SCREENING_MODELS == (
        "bulge_disk",
        "double_sersic",
        "sersic_point_source",
        "single_sersic",
    )
    assert set(SCREENING_MODELS) <= set(registered_models())


def test_screening_ignores_models_registered_later(
    extra_registered_model: str,
) -> None:
    """A model added to the registry changes no screening contract."""
    assert extra_registered_model in registered_models()

    report = ModelScreeningReport.from_scores(
        {
            "single_sersic": 0.4,
            "bulge_disk": 0.3,
            "double_sersic": 0.2,
            "sersic_point_source": 0.1,
        },
        status="ambiguous",
        method="test_classifier_v1",
    )

    assert {
        item.model_name
        for item in report.scores
    } == set(SCREENING_MODELS)

    for model_name in SCREENING_MODELS:
        parameters = controlled_parameters_from_latents(
            model_name=model_name,
            latents=LATENTS,
            image_shape=(32, 32),
        )

        assert parameters

    with pytest.raises(
        ValueError,
        match="Unknown model name",
    ):
        controlled_parameters_from_latents(
            model_name=extra_registered_model,
            latents=LATENTS,
            image_shape=(32, 32),
        )

    with pytest.raises(
        ValueError,
        match="Not a model-screening model",
    ):
        simulate_model_screening_case(
            model_name=extra_registered_model,
            parameters={"flux": 1.0},
            image_shape=(16, 16),
            psf_fwhm=2.0,
            noise_sigma=1.0,
            random_seed=0,
        )
