"""The masks of the preparation: the curated mask.

The curated mask, ``data.mask_path``, is a FITS image of the shape of the
whole image whose non-zero pixels are excluded, ``True`` = excluded: 1, -1
or any other finite non-zero value marks a pixel to leave out. It is read
once, on the working region of the opening, so it reaches the sky far from
the target as well as the cutout. A relative path is relative to the folder
of the project, never to the current directory.

The HDU is chosen by the rule of the image: the primary HDU when it holds an
image, otherwise the single extension named ``SCI`` or ``IMAGE``, or the single
two-dimensional image extension. A mask with a pixel that is not finite, of
another shape than the image, or without a two-dimensional image is refused.
A mask that excludes most of the working region is warned about, since a
mask that marks the good pixels with 1 must be inverted first.

The stages that detect sources leave the curated pixels out of the detection,
as the pixels without data, and out of their measurements.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from baymorph.preparation.identity import FileIdentity, file_identity, stage_identity
from baymorph.preparation.opening import (
    OpenedObservation,
    WorkingRegion,
    image_hdu,
    read_hdu_pixels,
)

CURATED_STAGE = "curated_mask"
"""Name of the curated mask among the inputs of the stages."""

CURATED_VERSION = 1
"""Version of the rules of the reading; raising it changes every identity."""

MOST_OF_THE_REGION = 0.5
"""Fraction of the working region above which a curated mask is warned about."""


@dataclass(frozen=True, eq=False)
class CuratedMask:
    """The curated mask of an observation, on its working region.

    Parameters
    ----------
    file : FileIdentity
        Identity of the content of the file, with its path relative to the
        project as information.
    hdu_index : int
        The HDU that holds the mask.
    region : WorkingRegion or None
        The working region of the opening, or None for the whole image.
    pixels : ndarray of bool
        The excluded pixels of the working region, ``True`` = excluded;
        read-only.
    identity : str
        Identity of the mask on the region.
    warnings : tuple of str
        What the stages that use the mask record and show.
    record : Mapping
        What the reading found, JSON-serializable.
    """

    file: FileIdentity
    hdu_index: int
    region: WorkingRegion | None
    pixels: NDArray[np.bool_] = field(repr=False)
    identity: str
    warnings: tuple[str, ...]
    record: Mapping[str, Any] = field(repr=False)

    @property
    def excluded_pixels(self) -> int:
        """int : The pixels of the working region that the mask excludes."""
        return int(np.count_nonzero(self.pixels))


def read_curated_mask(
    path: str | Path, opened: OpenedObservation, *, project_root: str | Path
) -> CuratedMask:
    """Read the curated mask of an observation on its working region.

    Parameters
    ----------
    path : str or pathlib.Path
        ``data.mask_path``; a relative path is relative to ``project_root``.
    opened : OpenedObservation
        The product of the opening, whose image and working region the mask
        must match.
    project_root : str or pathlib.Path
        The folder of the project.

    Returns
    -------
    CuratedMask
        The excluded pixels of the working region, with their identity.

    Raises
    ------
    TypeError
        If ``opened`` is not the product of the opening.
    FileNotFoundError
        If the file or the project folder does not exist.
    ValueError
        If the file holds no usable two-dimensional image, the mask does not
        have the shape of the whole image, or a pixel of the working region
        is not finite.
    """
    if not isinstance(opened, OpenedObservation):
        raise TypeError(
            "read_curated_mask takes the OpenedObservation of the opening stage, "
            f"not {type(opened).__name__}."
        )

    root = Path(project_root).expanduser().resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"Project folder not found: {root}")

    location = Path(path).expanduser()

    if not location.is_absolute():
        location = root / location

    location = location.resolve()

    if not location.is_file():
        raise FileNotFoundError(
            f"The curated mask (data.mask_path) was not found: {location}. A "
            f"relative path is relative to the project folder, {root}."
        )

    try:
        chosen = image_hdu(location)
    except ValueError as error:
        raise ValueError(
            f"The curated mask {location} (data.mask_path) cannot be used: {error}"
        ) from error

    shape = (int(chosen.header["NAXIS2"]), int(chosen.header["NAXIS1"]))

    if shape != tuple(opened.image_shape):
        raise ValueError(
            f"The curated mask {location} (data.mask_path), HDU {chosen.index}, has "
            f"the shape {shape} and the image {tuple(opened.image_shape)}: a "
            "curated mask has the shape of the whole image, pixel for pixel."
        )

    values, _ = read_hdu_pixels(location, chosen.index, shape, opened.region)
    bad = int(np.count_nonzero(~np.isfinite(values)))

    if bad:
        raise ValueError(
            f"The curated mask {location} (data.mask_path) holds {bad} pixels that "
            "are not finite in the working region: a curated mask holds 0 for the "
            "pixels used and a finite non-zero value for the pixels excluded."
        )

    pixels = values != 0
    pixels.flags.writeable = False
    identity_of_file = file_identity(location, project_root=root)
    region = opened.region
    region_record = None if region is None else region.as_dict()
    identity = stage_identity(
        stage=CURATED_STAGE,
        version=CURATED_VERSION,
        settings={"hdu_index": chosen.index, "region": region_record},
        inputs={"mask": identity_of_file},
    )
    excluded = int(np.count_nonzero(pixels))
    fraction = excluded / pixels.size
    warnings = []

    if fraction > MOST_OF_THE_REGION:
        warnings.append(
            f"The curated mask (data.mask_path) excludes {100.0 * fraction:.0f} % "
            "of the working region. Its non-zero pixels are excluded: a mask "
            "whose 1 marks the good pixels must be inverted."
        )

    record = {
        "stage": CURATED_STAGE,
        "version": CURATED_VERSION,
        "file": identity_of_file.as_record(),
        "hdu": {"index": chosen.index, "name": chosen.extension_name},
        "region": region_record,
        "excluded_pixels": excluded,
        "excluded_fraction": fraction,
        "rule": "non-zero pixels are excluded",
        "warnings": list(warnings),
        "identity": identity,
    }

    return CuratedMask(
        file=identity_of_file,
        hdu_index=chosen.index,
        region=region,
        pixels=pixels,
        identity=identity,
        warnings=tuple(warnings),
        record=MappingProxyType(record),
    )


def curated_from_settings(
    settings: Any, opened: OpenedObservation, *, project_root: str | Path
) -> CuratedMask | None:
    """Read the curated mask of a configuration, or None when it declares none.

    Parameters
    ----------
    settings : PreparationSettings
        The configuration as :func:`baymorph.preparation.schema.
        read_preparation_settings` reads it.
    opened : OpenedObservation
        The product of the opening.
    project_root : str or pathlib.Path
        The folder of the project; ``data.mask_path`` is relative to it.

    Returns
    -------
    CuratedMask or None
        As :func:`read_curated_mask`.
    """
    path = settings.get("data.mask_path")

    if path is None:
        return None

    return read_curated_mask(path, opened, project_root=project_root)


__all__ = [
    "CURATED_STAGE",
    "CURATED_VERSION",
    "MOST_OF_THE_REGION",
    "CuratedMask",
    "curated_from_settings",
    "read_curated_mask",
]
