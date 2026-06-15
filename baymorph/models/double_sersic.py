"""Double Sérsic model."""

from .composite import CompositeModel
from .sersic import SersicModel


class DoubleSersicModel(
    CompositeModel,
):
    """Convenience wrapper for a bulge+disk Sérsic model."""

    def __init__(
        self,
        bulge: SersicModel,
        disk: SersicModel,
    ):
        self.bulge = bulge
        self.disk = disk

        super().__init__(
            [
                bulge,
                disk,
            ]
        )