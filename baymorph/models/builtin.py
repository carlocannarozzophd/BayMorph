"""Registration of built-in BayMorph model definitions."""

from __future__ import annotations

from baymorph.models.definition import ModelDefinition
from baymorph.models.geometry import (
    EllipticalStructureGeometry,
    ModelGeometry,
)
from baymorph.models.labels import parameter_labels, parameter_names
from baymorph.models.registry import register_model


_BUILTIN_MODELS = {
    "single_sersic": {
        "factory_name": "SingleSersicModelFactory",
        "geometry": ModelGeometry(
            centre_x="x0",
            centre_y="y0",
            elliptical_structures=(
                EllipticalStructureGeometry(
                    radius="re",
                    axis_ratio="q",
                    position_angle="theta",
                ),
            ),
        ),
    },
    "double_sersic": {
        "factory_name": "DoubleSersicModelFactory",
        "geometry": ModelGeometry(
            centre_x="x0",
            centre_y="y0",
            elliptical_structures=(
                EllipticalStructureGeometry(
                    radius="re_bulge",
                    axis_ratio="q_bulge",
                    position_angle="theta_bulge",
                ),
                EllipticalStructureGeometry(
                    radius="re_disk",
                    axis_ratio="q_disk",
                    position_angle="theta_disk",
                ),
            ),
        ),
    },
    "sersic_point_source": {
        "factory_name": "SersicPointSourceModelFactory",
        "geometry": ModelGeometry(
            centre_x="x0",
            centre_y="y0",
            elliptical_structures=(
                EllipticalStructureGeometry(
                    radius="re",
                    axis_ratio="q",
                    position_angle="theta",
                ),
            ),
        ),
    },
    "bulge_disk": {
        "factory_name": "BulgeDiskModelFactory",
        "geometry": ModelGeometry(
            centre_x="x0",
            centre_y="y0",
            elliptical_structures=(
                EllipticalStructureGeometry(
                    radius="re_bulge",
                    axis_ratio="q_bulge",
                    position_angle="theta_bulge",
                ),
                EllipticalStructureGeometry(
                    radius="re_disk",
                    axis_ratio="q_disk",
                    position_angle="theta_disk",
                ),
            ),
        ),
    },
    # The families of AD-048 have no geometry contract yet: the joint path
    # rejects them before writing any output, because their further lengths
    # and shape parameters would otherwise pass to every observation
    # untransformed.
    "boxy_sersic": {
        "factory_name": "BoxySersicModelFactory",
        "geometry": None,
    },
    "core_sersic": {
        "factory_name": "CoreSersicModelFactory",
        "geometry": None,
    },
    "bulge_broken_disk": {
        "factory_name": "BulgeBrokenDiskModelFactory",
        "geometry": None,
    },
    "bulge_disk_bar": {
        "factory_name": "BulgeDiskBarModelFactory",
        "geometry": None,
    },
    "bulge_disk_ring": {
        "factory_name": "BulgeDiskRingModelFactory",
        "geometry": None,
    },
    "bulge_disk_spiral": {
        "factory_name": "BulgeDiskSpiralModelFactory",
        "geometry": None,
    },
}


def register_builtin_models() -> None:
    """Register the built-in BayMorph model definitions.

    Each registered definition is the canonical source of truth for the
    model parameter contract, display labels, production factory, and
    supported geometry transformation.
    """
    for model_name, specification in _BUILTIN_MODELS.items():
        register_model(
            ModelDefinition(
                name=model_name,
                parameter_names=parameter_names(
                    model_name
                ),
                parameter_labels=parameter_labels(
                    model_name
                ),
                factory_name=specification[
                    "factory_name"
                ],
                factory=None,
                geometry=specification[
                    "geometry"
                ],
            )
        )


register_builtin_models()
