"""Inference-region construction from prepared astronomical images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class InferenceRegion:
    """Image footprint used by one likelihood evaluation.

    An inference region is derived from a larger prepared observation. It does
    not alter the preparation products and defines its own local pixel
    coordinate system.

    Parameters
    ----------
    data : ndarray
        Background-subtracted image data used by the likelihood.
    noise : ndarray
        Noise map corresponding to ``data``.
    mask : ndarray of bool
        Boolean exclusion mask. ``True`` pixels are excluded.
    target_x, target_y : float
        Target-reference coordinates in the local inference-region frame.
    xmin, xmax, ymin, ymax : int
        Half-open bounds of the inference region in preparation-cutout pixel
        coordinates.
    mode : {"full", "circular"}
        Region-selection mode.
    radius : float, optional
        Circular selection radius in preparation-cutout pixels. It is ``None``
        for the full preparation footprint.
    """

    data: FloatArray
    noise: FloatArray
    mask: BoolArray

    target_x: float
    target_y: float

    xmin: int
    xmax: int
    ymin: int
    ymax: int

    mode: str
    radius: float | None = None

    @property
    def shape(self) -> tuple[int, int]:
        """Return the inference-region image shape."""
        return self.data.shape

    @property
    def bounds(self) -> dict[str, int]:
        """Return half-open bounds in preparation-cutout coordinates."""
        return {
            "xmin": self.xmin,
            "xmax": self.xmax,
            "ymin": self.ymin,
            "ymax": self.ymax,
        }

    @property
    def n_rendered_pixels(self) -> int:
        """Return the number of pixels in the rectangular render footprint."""
        return int(self.data.size)

    @property
    def n_usable_pixels(self) -> int:
        """Return the number of unmasked pixels in the likelihood footprint."""
        return int(
            np.count_nonzero(
                ~self.mask
            )
        )

    def to_local(
        self,
        *,
        x: float,
        y: float,
    ) -> tuple[float, float]:
        """Transform preparation-cutout coordinates into local coordinates."""
        return (
            float(x) - self.xmin,
            float(y) - self.ymin,
        )

    def to_preparation(
        self,
        *,
        x: float,
        y: float,
    ) -> tuple[float, float]:
        """Transform local coordinates into preparation-cutout coordinates."""
        return (
            float(x) + self.xmin,
            float(y) + self.ymin,
        )


def _prepared_arrays(
    products: Any,
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """Validate and return canonical preparation arrays."""
    data = np.asarray(
        products.prepared_data,
        dtype=float,
    )
    noise = np.asarray(
        products.noise,
        dtype=float,
    )
    mask = np.asarray(
        products.fitting_mask,
        dtype=bool,
    )

    if data.ndim != 2:
        raise ValueError(
            "Prepared image data must be two-dimensional."
        )

    if noise.shape != data.shape:
        raise ValueError(
            "Prepared noise map must match prepared image shape."
        )

    if mask.shape != data.shape:
        raise ValueError(
            "Prepared fitting mask must match prepared image shape."
        )

    return data, noise, mask


def _finite_coordinate(
    value: float,
    *,
    name: str,
) -> float:
    """Validate one finite detector coordinate."""
    value = float(value)

    if not np.isfinite(value):
        raise ValueError(
            f"{name} must be finite."
        )

    return value


def _nearest_pixel(
    value: float,
) -> int:
    """Return the nearest detector-pixel index for a non-negative coordinate."""
    return int(
        np.floor(
            float(value) + 0.5
        )
    )


def build_inference_region(
    products: Any,
    *,
    mode: str = "full",
    radius: float | None = None,
    reference_x: float | None = None,
    reference_y: float | None = None,
) -> InferenceRegion:
    """Build one likelihood footprint from prepared observation products.

    Parameters
    ----------
    products : object
        Prepared-observation object exposing ``prepared_data``, ``noise``,
        ``fitting_mask``, ``target_x``, and ``target_y``.
    mode : {"full", "circular"}, default="full"
        Region-selection mode.
    radius : float, optional
        Circular radius in preparation-cutout pixels. Required for
        ``mode="circular"`` and forbidden for ``mode="full"``.
    reference_x, reference_y : float, optional
        Target-reference coordinates in preparation-cutout pixels. If omitted,
        the prepared target coordinates are used.

    Returns
    -------
    InferenceRegion
        Local data, noise, mask, coordinate transform, and region metadata.

    Raises
    ------
    ValueError
        If the requested region is invalid or does not fit inside the prepared
        image.

    Notes
    -----
    Circular regions use the smallest odd rectangular footprint centred on
    the detector pixel nearest the supplied reference coordinate. Pixels
    outside the requested Euclidean radius are excluded through the returned
    mask.

    Region construction never modifies the preparation products.
    """
    data, noise, mask = _prepared_arrays(
        products
    )

    mode = str(mode).strip().lower()

    if mode not in {
        "full",
        "circular",
    }:
        raise ValueError(
            "Inference-region mode must be 'full' or 'circular'."
        )

    if reference_x is None:
        reference_x = products.target_x

    if reference_y is None:
        reference_y = products.target_y

    reference_x = _finite_coordinate(
        reference_x,
        name="reference_x",
    )
    reference_y = _finite_coordinate(
        reference_y,
        name="reference_y",
    )

    ny, nx = data.shape

    if not (
        0.0 <= reference_x <= nx - 1
        and 0.0 <= reference_y <= ny - 1
    ):
        raise ValueError(
            "Inference-region reference coordinates must lie inside "
            "the prepared image."
        )

    if mode == "full":
        if radius is not None:
            raise ValueError(
                "radius must be omitted when inference-region mode is 'full'."
            )

        return InferenceRegion(
            data=data,
            noise=noise,
            mask=mask,
            target_x=reference_x,
            target_y=reference_y,
            xmin=0,
            xmax=nx,
            ymin=0,
            ymax=ny,
            mode="full",
            radius=None,
        )

    if radius is None:
        raise ValueError(
            "radius is required when inference-region mode is 'circular'."
        )

    radius = float(radius)

    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(
            "Circular inference-region radius must be finite and positive."
        )

    half_size = int(
        np.ceil(radius)
    )

    center_pixel_x = _nearest_pixel(
        reference_x
    )
    center_pixel_y = _nearest_pixel(
        reference_y
    )

    xmin = center_pixel_x - half_size
    xmax = center_pixel_x + half_size + 1
    ymin = center_pixel_y - half_size
    ymax = center_pixel_y + half_size + 1

    if (
        xmin < 0
        or ymin < 0
        or xmax > nx
        or ymax > ny
    ):
        raise ValueError(
            "Circular inference region does not fit inside the prepared image."
        )

    local_data = np.asarray(
        data[
            ymin:ymax,
            xmin:xmax,
        ],
        dtype=float,
    )

    local_noise = np.asarray(
        noise[
            ymin:ymax,
            xmin:xmax,
        ],
        dtype=float,
    )

    local_mask = np.asarray(
        mask[
            ymin:ymax,
            xmin:xmax,
        ],
        dtype=bool,
    ).copy()

    local_target_x = reference_x - xmin
    local_target_y = reference_y - ymin

    yy, xx = np.indices(
        local_data.shape,
        dtype=float,
    )

    outside_circle = (
        (xx - local_target_x) ** 2
        + (yy - local_target_y) ** 2
        > radius**2
    )

    local_mask |= outside_circle

    return InferenceRegion(
        data=local_data,
        noise=local_noise,
        mask=local_mask,
        target_x=local_target_x,
        target_y=local_target_y,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        mode="circular",
        radius=radius,
    )
