"""Tests for the registered Sérsic plus point-source definition."""

from __future__ import annotations

from baymorph.models import (
    get_model,
    parameter_labels,
    parameter_names,
    registered_models,
)


EXPECTED_PARAMETER_NAMES = [
    "flux_sersic",
    "re",
    "n",
    "q",
    "theta",
    "flux_point_source",
    "x0",
    "y0",
    "sky",
]


def test_sersic_point_source_parameter_order() -> None:
    """The public parameter order must remain explicit and stable."""
    assert parameter_names(
        "sersic_point_source"
    ) == EXPECTED_PARAMETER_NAMES


def test_sersic_point_source_labels_match_parameters() -> None:
    """Every production parameter must have one display label."""
    labels = parameter_labels(
        "sersic_point_source"
    )

    assert len(labels) == len(EXPECTED_PARAMETER_NAMES)
    assert labels[0] == r"$F_{\mathrm{Sersic}}$"
    assert labels[5] == r"$F_{\mathrm{PS}}$"


def test_sersic_point_source_is_registered() -> None:
    """The named model must be discoverable through the registry."""
    assert "sersic_point_source" in registered_models()


def test_sersic_point_source_definition_uses_expected_factory() -> None:
    """The registry must reference the future production factory."""
    definition = get_model(
        "sersic_point_source"
    )

    assert definition.parameter_names == EXPECTED_PARAMETER_NAMES
    assert (
        definition.factory_name
        == "SersicPointSourceModelFactory"
    )
