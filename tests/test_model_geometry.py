"""Tests for declarative model geometry contracts."""

from __future__ import annotations

import pytest

from baymorph.models import get_model
from baymorph.models.geometry import (
    get_model_geometry,
)


@pytest.mark.parametrize(
    "model_name",
    [
        "single_sersic",
        "double_sersic",
        "bulge_disk",
        "sersic_point_source",
    ],
)
def test_geometry_is_owned_by_registered_model_definition(
    model_name,
):
    definition = get_model(
        model_name
    )
    geometry = get_model_geometry(
        model_name
    )

    assert definition.geometry is geometry


@pytest.mark.parametrize(
    "model_name",
    [
        "single_sersic",
        "double_sersic",
        "bulge_disk",
        "sersic_point_source",
    ],
)
def test_geometry_contract_uses_registered_model_parameters(
    model_name,
):
    definition = get_model(
        model_name
    )
    geometry = get_model_geometry(
        model_name
    )

    parameter_names = set(
        definition.parameter_names
    )

    assert geometry.centre_x in parameter_names
    assert geometry.centre_y in parameter_names

    for structure in geometry.elliptical_structures:
        assert structure.radius in parameter_names
        assert structure.axis_ratio in parameter_names
        assert structure.position_angle in parameter_names


def test_single_sersic_has_one_elliptical_structure():
    geometry = get_model_geometry(
        "single_sersic"
    )

    assert len(
        geometry.elliptical_structures
    ) == 1


@pytest.mark.parametrize(
    "model_name",
    [
        "double_sersic",
        "bulge_disk",
    ],
)
def test_two_structure_models_declare_two_ellipses(
    model_name,
):
    geometry = get_model_geometry(
        model_name
    )

    assert len(
        geometry.elliptical_structures
    ) == 2


def test_sersic_point_source_declares_only_extended_ellipse():
    geometry = get_model_geometry(
        "sersic_point_source"
    )

    assert len(
        geometry.elliptical_structures
    ) == 1

    structure = geometry.elliptical_structures[
        0
    ]

    assert structure.radius == "re"
    assert structure.axis_ratio == "q"
    assert structure.position_angle == "theta"


def test_unknown_model_geometry_is_rejected():
    with pytest.raises(
        ValueError,
        match="Unknown model",
    ):
        get_model_geometry(
            "unknown_model"
        )


@pytest.mark.parametrize(
    (
        "model_name",
        "expected",
    ),
    [
        (
            "single_sersic",
            (
                "flux",
                "re_arcsec",
                "n",
                "q",
                "position_angle",
                "ra",
                "dec",
            ),
        ),
        (
            "double_sersic",
            (
                "flux_bulge",
                "re_arcsec_bulge",
                "n_bulge",
                "q_bulge",
                "position_angle_bulge",
                "flux_disk",
                "re_arcsec_disk",
                "n_disk",
                "q_disk",
                "position_angle_disk",
                "ra",
                "dec",
            ),
        ),
        (
            "bulge_disk",
            (
                "flux_bulge",
                "re_arcsec_bulge",
                "n_bulge",
                "q_bulge",
                "position_angle_bulge",
                "flux_disk",
                "re_arcsec_disk",
                "q_disk",
                "position_angle_disk",
                "ra",
                "dec",
            ),
        ),
        (
            "sersic_point_source",
            (
                "flux_sersic",
                "re_arcsec",
                "n",
                "q",
                "position_angle",
                "flux_point_source",
                "ra",
                "dec",
            ),
        ),
    ],
)
def test_sky_parameter_contract_is_derived_from_model_geometry(
    model_name,
    expected,
):
    from baymorph.models.geometry import (
        sky_parameter_names,
    )

    assert sky_parameter_names(
        model_name
    ) == expected
