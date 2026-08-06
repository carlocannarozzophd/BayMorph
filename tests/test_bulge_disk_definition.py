"""Tests for the registered bulge-disc model definition."""

from __future__ import annotations

from baymorph.models import (
    get_model,
    parameter_labels,
    parameter_names,
    registered_models,
)


EXPECTED_PARAMETER_NAMES = [
    "flux_bulge",
    "re_bulge",
    "n_bulge",
    "q_bulge",
    "theta_bulge",
    "flux_disk",
    "re_disk",
    "q_disk",
    "theta_disk",
    "x0",
    "y0",
    "sky",
]


def test_bulge_disk_parameter_order() -> None:
    """The production parameter order must remain explicit and stable."""
    assert parameter_names(
        "bulge_disk"
    ) == EXPECTED_PARAMETER_NAMES


def test_bulge_disk_omits_disk_sersic_index() -> None:
    """The exponential disc must not expose a free Sérsic index."""
    assert "n_disk" not in parameter_names(
        "bulge_disk"
    )


def test_bulge_disk_labels_match_parameters() -> None:
    """Every production parameter must have one display label."""
    labels = parameter_labels(
        "bulge_disk"
    )

    assert len(labels) == len(
        EXPECTED_PARAMETER_NAMES
    )
    assert labels[0] == r"$F_{\mathrm{b}}$"
    assert labels[5] == r"$F_{\mathrm{d}}$"


def test_bulge_disk_is_registered() -> None:
    """The named model must be discoverable through the registry."""
    assert "bulge_disk" in registered_models()


def test_bulge_disk_definition_uses_expected_factory() -> None:
    """The registry must reference the production factory."""
    definition = get_model(
        "bulge_disk"
    )

    assert (
        definition.parameter_names
        == EXPECTED_PARAMETER_NAMES
    )
    assert (
        definition.factory_name
        == "BulgeDiskModelFactory"
    )
