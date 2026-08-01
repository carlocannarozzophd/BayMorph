"""Double-Sersic morphology model."""

from dataclasses import dataclass

import numpy as np


@dataclass
class DoubleSersicModel:
    """Two-component Sérsic model."""

    bulge: object
    disk: object
    sky: float = 0.0

    def render_components(
        self,
        shape: tuple[int, int],
        *,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> dict[str, np.ndarray]:
        """Render the bulge and disk components."""
        return {
            "bulge": self.bulge.render(
                shape=shape,
                origin=origin,
            ),
            "disk": self.disk.render(
                shape=shape,
                origin=origin,
            ),
        }

    def render(
        self,
        shape: tuple[int, int],
        *,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        """Render the complete double-Sérsic model."""
        components = self.render_components(
            shape=shape,
            origin=origin,
        )

        return (
            components["bulge"]
            + components["disk"]
            + self.sky
        )
