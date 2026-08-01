"""Synthetic Sérsic galaxy model."""

from dataclasses import dataclass

from baymorph.models.sersic import SersicModel


@dataclass
class SyntheticSersic(SersicModel):
    """Sérsic model used to generate synthetic observations.

    This class intentionally reuses the production Sérsic forward model.
    Therefore, ``flux`` denotes the continuous total flux integrated over the
    full elliptical profile, and rendered images contain only the fraction of
    that flux falling inside the requested footprint.

    The separate class name is retained for API compatibility and to express
    the role of the object in simulation workflows.
    """
