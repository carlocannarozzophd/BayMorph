"""Geometry definitions for Sérsic morphology models."""

from __future__ import annotations

import numpy as np

from .geometry_model import GeometryModel
from .parameter_geometry import ParameterGeometry
from .transforms import IdentityTransform, LogTransform


SINGLE_SERSIC_GEOMETRY = GeometryModel(
    (
        ParameterGeometry(
            name="x_0",
            transform=IdentityTransform(),
            block="position",
        ),
        ParameterGeometry(
            name="y_0",
            transform=IdentityTransform(),
            block="position",
        ),
        ParameterGeometry(
            name="flux",
            transform=LogTransform(),
            block="photometry",
        ),
        ParameterGeometry(
            name="r_eff",
            transform=LogTransform(),
            block="structure",
        ),
        ParameterGeometry(
            name="n",
            transform=LogTransform(),
            block="structure",
        ),
        ParameterGeometry(
            name="ellip",
            transform=IdentityTransform(),
            block="shape",
        ),
        ParameterGeometry(
            name="theta",
            transform=IdentityTransform(),
            block="shape",
            period=float(np.pi),
        ),
    )
)


DOUBLE_SERSIC_GEOMETRY = GeometryModel(
    (
        ParameterGeometry(
            name="x_0",
            transform=IdentityTransform(),
            block="position",
        ),
        ParameterGeometry(
            name="y_0",
            transform=IdentityTransform(),
            block="position",
        ),
        ParameterGeometry(
            name="flux_1",
            transform=LogTransform(),
            block="photometry_1",
        ),
        ParameterGeometry(
            name="r_eff_1",
            transform=LogTransform(),
            block="structure_1",
        ),
        ParameterGeometry(
            name="n_1",
            transform=LogTransform(),
            block="structure_1",
        ),
        ParameterGeometry(
            name="ellip_1",
            transform=IdentityTransform(),
            block="shape_1",
        ),
        ParameterGeometry(
            name="theta_1",
            transform=IdentityTransform(),
            block="shape_1",
            period=float(np.pi),
        ),
        ParameterGeometry(
            name="flux_2",
            transform=LogTransform(),
            block="photometry_2",
        ),
        ParameterGeometry(
            name="r_eff_2",
            transform=LogTransform(),
            block="structure_2",
        ),
        ParameterGeometry(
            name="n_2",
            transform=LogTransform(),
            block="structure_2",
        ),
        ParameterGeometry(
            name="ellip_2",
            transform=IdentityTransform(),
            block="shape_2",
        ),
        ParameterGeometry(
            name="theta_2",
            transform=IdentityTransform(),
            block="shape_2",
            period=float(np.pi),
        ),
    )
)


__all__ = [
    "DOUBLE_SERSIC_GEOMETRY",
    "SINGLE_SERSIC_GEOMETRY",
]
