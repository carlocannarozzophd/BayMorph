from __future__ import annotations

from dataclasses import dataclass



@dataclass(frozen=True)
class CutoutDefinition:
    """
    Fully resolved cutout definition.
    """

    xmin: int
    xmax: int

    ymin: int
    ymax: int

    width: int
    height: int

    mode: str


def resolve_fixed_cutout(
    *,
    image_shape: tuple[int, int],
    center_x: float,
    center_y: float,
    size: int,
) -> CutoutDefinition:
    """
    Resolve a fixed-size cutout centred on (center_x, center_y).
    """

    if size <= 0:
        raise ValueError(
            "cutout size must be positive."
        )

    ny, nx = image_shape

    half = size // 2

    cx = int(round(center_x))
    cy = int(round(center_y))

    xmin = cx - half
    xmax = xmin + size

    ymin = cy - half
    ymax = ymin + size

    if xmin < 0:
        xmax -= xmin
        xmin = 0

    if ymin < 0:
        ymax -= ymin
        ymin = 0

    if xmax > nx:
        xmin -= xmax - nx
        xmax = nx

    if ymax > ny:
        ymin -= ymax - ny
        ymax = ny

    xmin = max(
        xmin,
        0,
    )

    ymin = max(
        ymin,
        0,
    )

    return CutoutDefinition(
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        width=xmax - xmin,
        height=ymax - ymin,
        mode="fixed",
    )


def resolve_cutout(
    *,
    image_shape: tuple[int, int],
    center_x: float,
    center_y: float,
    config: dict,
) -> CutoutDefinition:
    """
    Resolve the requested cutout strategy.
    """

    mode = config["mode"]

    if mode == "fixed":
        return resolve_fixed_cutout(
            image_shape=image_shape,
            center_x=center_x,
            center_y=center_y,
            size=int(
                config["size"]
            ),
        )

    raise ValueError(
        f"Unsupported cutout mode: {mode}"
    )