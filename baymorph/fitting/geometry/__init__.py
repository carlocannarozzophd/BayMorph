"""Geometry-aware sampling infrastructure.

This package contains the building blocks used by the experimental BayMorph
geometry-aware nested sampler. Components are promoted to the production
workflow only after dedicated statistical validation.
"""

from .blocks import ProposalBlock
from .geometry_model import GeometryModel
from .parameter_geometry import ParameterGeometry
from .transforms import IdentityTransform, LogTransform, Transform

__all__ = [
    "GeometryModel",
    "IdentityTransform",
    "LogTransform",
    "ParameterGeometry",
    "ProposalBlock",
    "Transform",
]
