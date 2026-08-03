"""Geometric integration cells for adaptive image rendering."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IntegrationCell:
    """Represent a rectangular domain used for numerical integration."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    level: int = 0

    def __post_init__(self) -> None:
        """Validate cell bounds and subdivision level."""
        if self.xmax <= self.xmin:
            raise ValueError("xmax must be greater than xmin.")

        if self.ymax <= self.ymin:
            raise ValueError("ymax must be greater than ymin.")

        if self.level < 0:
            raise ValueError("level must be non-negative.")

    @property
    def width(self) -> float:
        """Return the cell width."""
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        """Return the cell height."""
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        """Return the cell area."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Return the cell centre as ``(x, y)``."""
        return (
            0.5 * (self.xmin + self.xmax),
            0.5 * (self.ymin + self.ymax),
        )

    def subdivide(
        self,
    ) -> tuple[
        IntegrationCell,
        IntegrationCell,
        IntegrationCell,
        IntegrationCell,
    ]:
        """Split the cell isotropically into four equal children."""
        xmid, ymid = self.center
        child_level = self.level + 1

        return (
            IntegrationCell(
                xmin=self.xmin,
                xmax=xmid,
                ymin=self.ymin,
                ymax=ymid,
                level=child_level,
            ),
            IntegrationCell(
                xmin=xmid,
                xmax=self.xmax,
                ymin=self.ymin,
                ymax=ymid,
                level=child_level,
            ),
            IntegrationCell(
                xmin=self.xmin,
                xmax=xmid,
                ymin=ymid,
                ymax=self.ymax,
                level=child_level,
            ),
            IntegrationCell(
                xmin=xmid,
                xmax=self.xmax,
                ymin=ymid,
                ymax=self.ymax,
                level=child_level,
            ),
        )
