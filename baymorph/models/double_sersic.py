"""Double-Sersic morphology model."""

from dataclasses import dataclass

import numpy as np

from baymorph.models.composite import common_centre


@dataclass
class DoubleSersicModel:
    """Two-component Sérsic model."""

    bulge: object
    disk: object
    sky: float = 0.0

    def centre(self) -> tuple[float, float]:
        """Return the centre shared by the two components."""
        return common_centre((self.bulge, self.disk))

    def scaled(self, factor: int) -> "DoubleSersicModel":
        """Return this model on a pixel grid finer by ``factor``.

        The constant level is per pixel, so it is divided by the number of
        fine cells in a pixel.
        """
        return DoubleSersicModel(
            bulge=self.bulge.scaled(factor),
            disk=self.disk.scaled(factor),
            sky=self.sky / factor**2,
        )

    def render_components(
        self,
        shape: tuple[int, int],
        *,
        origin: tuple[float, float] = (0.0, 0.0),
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """Render the bulge and disk components."""
        return {
            "bulge": self.bulge.render(
                shape=shape,
                origin=origin,
                **kwargs,
            ),
            "disk": self.disk.render(
                shape=shape,
                origin=origin,
                **kwargs,
            ),
        }

    def render(
        self,
        shape: tuple[int, int],
        *,
        origin: tuple[float, float] = (0.0, 0.0),
        **kwargs,
    ) -> np.ndarray:
        """Render the complete double-Sérsic model."""
        components = self.render_components(
            shape=shape,
            origin=origin,
            **kwargs,
        )

        return (
            components["bulge"]
            + components["disk"]
            + self.sky
        )
