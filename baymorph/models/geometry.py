"""Declarative geometry contracts for registered morphology models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EllipticalStructureGeometry:
    """Parameter names defining one elliptical structure within a model.

    Parameters
    ----------
    radius : str
        Detector-frame semi-major-axis scale parameter.
    axis_ratio : str
        Detector-frame axis-ratio parameter.
    position_angle : str
        Detector-frame position-angle parameter.

    Notes
    -----
    A structure is part of one morphological representation of one physical
    galaxy. It is not an independent astronomical object.
    """

    radius: str
    axis_ratio: str
    position_angle: str


@dataclass(frozen=True)
class ModelGeometry:
    """Geometry contract of one production galaxy-model parameterization.

    Parameters
    ----------
    centre_x, centre_y : str
        Detector-frame source-centre parameter names.
    elliptical_structures : tuple of EllipticalStructureGeometry
        Elliptical structures transformed between scientific and detector
        coordinate frames.

    Notes
    -----
    This contract describes geometric structures within one adopted
    morphological representation of one physical galaxy.

    It does not impose physical coupling requirements on the underlying
    intrinsic morphology family. Shared centres or other geometric relations
    remain properties of the production parameterization and probabilistic
    model.
    """

    centre_x: str
    centre_y: str
    elliptical_structures: tuple[
        EllipticalStructureGeometry,
        ...,
    ]


def get_model_geometry(
    model_name: str,
) -> ModelGeometry:
    """Return the geometry contract stored by a registered model.

    Raises
    ------
    ValueError
        If the model is unknown or does not define a geometry contract.
    """
    from baymorph.models.registry import get_model

    definition = get_model(
        model_name
    )

    if definition.geometry is None:
        raise ValueError(
            f"No geometry contract is registered for model "
            f"'{model_name}'."
        )

    return definition.geometry


def sky_parameter_names(
    model_name: str,
) -> tuple[str, ...]:
    """Return the sky-frame parameter contract of a registered model.

    Detector-frame centre coordinates are replaced by ``ra`` and ``dec``.
    Elliptical radii and position angles are replaced by their sky-frame
    counterparts. Non-geometric parameters preserve their registered names.
    """
    from baymorph.models.registry import get_model

    definition = get_model(
        model_name
    )
    geometry = get_model_geometry(
        model_name
    )

    radius_names = {
        structure.radius
        for structure in geometry.elliptical_structures
    }
    angle_names = {
        structure.position_angle
        for structure in geometry.elliptical_structures
    }

    names: list[str] = []

    for name in definition.parameter_names:
        if name == geometry.centre_x:
            names.append(
                "ra"
            )
            continue

        if name == geometry.centre_y:
            names.append(
                "dec"
            )
            continue

        if name in radius_names:
            if name == "re":
                names.append(
                    "re_arcsec"
                )
            else:
                names.append(
                    "re_arcsec_"
                    + name.removeprefix(
                        "re_"
                    )
                )
            continue

        if name in angle_names:
            if name == "theta":
                names.append(
                    "position_angle"
                )
            else:
                names.append(
                    "position_angle_"
                    + name.removeprefix(
                        "theta_"
                    )
                )
            continue

        names.append(
            name
        )

    return tuple(
        names
    )
