"""Opening of the image of an observation.

Every stage of the pipeline that needs the image of an observation opens it
through :func:`open_observation`, so every stage reads the same HDU and the
array that one stage sees is the array that the others see. The array is
read-only: a stage that changes pixel values works on its own copy.

A large file is read once per process and reused while it is unchanged, so a
mosaic of several gigabytes is not read again at every stage. A small file is
read at every opening, as reading it costs less than tracking it.

An image with a side larger than :data:`WHOLE_IMAGE_MAX_SIDE` pixels is read
only within a working region about the target, ``data.region`` (AD-053).
Every position, of the target, of a cutout or of a star, stays in the pixels
of the whole image.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

from baymorph.core.header_metadata import read_header
from baymorph.core.image import ImageHeader, WorkingRegion, read_image_headers
from baymorph.core.observation import Observation, ObservationMetadata

REUSE_MIN_BYTES = 16 * 1024**2
"""Size from which a file is read once per process and then reused."""

MAX_OPEN_OBSERVATIONS = 4
"""Number of large observations kept for reuse within one process."""

WHOLE_IMAGE_MAX_SIDE = 4096
"""Largest side, in pixels, of an image read whole by default (AD-053)."""

_OPENED: OrderedDict[tuple, Observation] = OrderedDict()
_READ_METADATA: OrderedDict[tuple, ObservationMetadata] = OrderedDict()


def file_state(path: Path) -> tuple[int, int, int, int, int]:
    """Return the state that changes whenever the content of a file changes.

    Parameters
    ----------
    path : pathlib.Path
        Existing file.

    Returns
    -------
    tuple of int
        Device, inode, size, and the times of the last modification and of
        the last change of status, in nanoseconds. The second time is set by
        the system on every write and cannot be restored by copying tools,
        so a file replaced by another with the same size and modification
        time still changes state.
    """
    status = path.stat()

    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def validate_region(value: Any) -> str | dict[str, float] | None:
    """Validate a ``data.region`` setting and return it in canonical form.

    Parameters
    ----------
    value : None, str or mapping
        ``None``, the default: the whole image when no side exceeds
        :data:`WHOLE_IMAGE_MAX_SIDE` pixels, otherwise a square of that side
        about the target. ``"whole"``: always the whole image. ``{"size":
        side}``: a square of ``side`` arcsec about the target.

    Returns
    -------
    None, str or dict
        ``None``, ``"whole"``, or ``{"size": side}`` with the side as a float.

    Raises
    ------
    ValueError
        If the setting has none of these forms, or the side is not a finite
        positive number.
    """
    if value is None:
        return None

    if isinstance(value, str) and value.strip().lower() == "whole":
        return "whole"

    if isinstance(value, Mapping) and set(value) == {"size"}:
        size = value["size"]

        if (
            isinstance(size, Real)
            and not isinstance(size, bool)
            and math.isfinite(size)
            and size > 0
        ):
            return {"size": float(size)}

    raise ValueError(
        "data.region must be null, the default (the whole image up to "
        f"{WHOLE_IMAGE_MAX_SIDE} pixels on a side, otherwise a square of "
        f"{WHOLE_IMAGE_MAX_SIDE} pixels about the target), 'whole', or a "
        "mapping {size: <side in arcsec>} with a finite positive side; got "
        f"{value!r}."
    )


@dataclass(frozen=True)
class RegionRequest:
    """The settings that decide the working region of an image (AD-053).

    The region itself depends also on the size and the pixel scale of the
    image, so it is resolved from the header, by :meth:`resolve`.

    Parameters
    ----------
    setting : None, str or tuple
        The canonical ``data.region``, as :func:`validate_region` returns it,
        with a side in arcsec given as ``("size", side)``.
    center : str or tuple of float, optional
        Position about which the region is taken, in pixels of the whole
        image, or ``"center"`` for the centre of the image. None when no
        position is known.
    pixel_scale : float, optional
        Declared pixel scale, in arcsec per pixel. The scale of the celestial
        WCS of the header is used when it is None.
    """

    setting: str | tuple[str, float] | None = None
    center: str | tuple[float, float] | None = None
    pixel_scale: float | None = None

    @classmethod
    def of(
        cls, region: Any = None, *, target: Any = None, pixel_scale: Any = None
    ) -> RegionRequest:
        """Build a request from the values of a configuration.

        Parameters
        ----------
        region : None, str or mapping, optional
            ``data.region``.
        target : str, mapping or sequence, optional
            ``data.target_position``: ``"center"``, ``{"x": ..., "y": ...}``
            or ``(x, y)``.
        pixel_scale : float or str, optional
            ``data.pixel_scale``; ``None`` or ``"auto"`` take the scale of
            the header.

        Returns
        -------
        RegionRequest
            The request.

        Raises
        ------
        ValueError
            If ``region`` is not valid, or ``target`` is neither ``"center"``
            nor two finite coordinates.
        """
        setting = validate_region(region)

        if isinstance(setting, dict):
            setting = ("size", setting["size"])

        scale = None

        if isinstance(pixel_scale, Real) and not isinstance(pixel_scale, bool):
            scale = float(pixel_scale)

        return cls(setting=setting, center=_region_center(target), pixel_scale=scale)

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> RegionRequest:
        """Build the request of the ``data`` section of a configuration."""
        return cls.of(
            data.get("region"),
            target=data.get("target_position"),
            pixel_scale=data.get("pixel_scale"),
        )

    def resolve(self, metadata: ObservationMetadata) -> WorkingRegion | None:
        """Return the working region of an image.

        Parameters
        ----------
        metadata : ObservationMetadata
            Metadata of the whole image, read from its header.

        Returns
        -------
        WorkingRegion or None
            The region, or None when the whole image is read.

        Raises
        ------
        ValueError
            If a region about the target is needed but no target is known,
            or a side in arcsec is given for an image without a pixel scale.
        """
        width, height = int(metadata.width), int(metadata.height)

        if self.setting == "whole":
            return None

        if self.setting is None:
            if max(width, height) <= WHOLE_IMAGE_MAX_SIDE:
                return None

            side = WHOLE_IMAGE_MAX_SIDE
        else:
            side = _side_in_pixels(self.setting[1], self._pixel_scale(metadata))

        if side >= width and side >= height:
            return None

        center = self._center(metadata)

        if center is None:
            raise ValueError(
                f"The image of {width} x {height} pixels is read within a "
                f"working region of {side} pixels about the target, and no "
                "target position is set: set data.target_position, or declare "
                "data.region: whole."
            )

        return WorkingRegion.about(
            (height, width), center_x=center[0], center_y=center[1], side=side
        )

    def _pixel_scale(self, metadata: ObservationMetadata) -> float:
        """Return the pixel scale that converts a side in arcsec."""
        scale = (
            self.pixel_scale if self.pixel_scale is not None else metadata.pixel_scale
        )

        if scale is None or not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                "data.region gives its side in arcsec, which needs the pixel "
                "scale of the image: its header has no celestial WCS, so "
                "declare data.pixel_scale."
            )

        return float(scale)

    def _center(self, metadata: ObservationMetadata) -> tuple[float, float] | None:
        """Return the position about which the region is taken."""
        if self.center == "center":
            return (float(metadata.center_x), float(metadata.center_y))

        return self.center


def describe_working_region(
    region: WorkingRegion | None, metadata: ObservationMetadata
) -> str:
    """Describe in one sentence how much of an image BayMorph reads.

    Parameters
    ----------
    region : WorkingRegion or None
        Working region, or None for the whole image.
    metadata : ObservationMetadata
        Metadata of the whole image.

    Returns
    -------
    str
        The size of the image and, when there is one, the region with its
        side in arcsec when the pixel scale is known.
    """
    size = f"{metadata.width} x {metadata.height} pixels"

    if region is None:
        return f"The image of {size} is read whole."

    height, width = region.shape
    side = f"{width} x {height} pixels"
    scale = metadata.pixel_scale

    if scale is not None and math.isfinite(scale) and scale > 0.0:
        side += f" ({width * scale:.4g} x {height * scale:.4g} arcsec)"

    return (
        f"The image of {size} is read within a working region of {side}, x "
        f"{region.xmin}-{region.xmax} and y {region.ymin}-{region.ymax} of "
        "the whole image (data.region)."
    )


def region_request_message(
    region: WorkingRegion,
    *,
    what: str,
    xmin: int,
    xmax: int,
    ymin: int,
    ymax: int,
    pixel_scale: float | None = None,
) -> str:
    """Explain that a box leaves the working region, and what to declare.

    Parameters
    ----------
    region : WorkingRegion
        Working region of the image.
    what : str
        What the box is, such as ``"The cutout"``.
    xmin, xmax, ymin, ymax : int
        Half-open bounds of the box, in pixels of the whole image.
    pixel_scale : float, optional
        Pixel scale in arcsec per pixel, to state the side to declare in
        arcsec.

    Returns
    -------
    str
        The message, which names a side that holds the box when the region
        is centred where it is now.
    """
    center_x = 0.5 * (region.xmin + region.xmax)
    center_y = 0.5 * (region.ymin + region.ymax)
    reach = max(center_x - xmin, xmax - center_x, center_y - ymin, ymax - center_y)
    side = 2 * int(math.ceil(reach)) + 2

    if pixel_scale is not None and math.isfinite(pixel_scale) and pixel_scale > 0:
        declare = f"data.region: {{size: {math.ceil(side * pixel_scale)}}}"
    else:
        declare = f"data.region: {{size: <{side} pixels times the pixel scale>}}"

    return (
        f"{what}, x {xmin}-{xmax} and y {ymin}-{ymax} of the image, leaves the "
        f"working region that BayMorph reads, x {region.xmin}-{region.xmax} "
        f"and y {region.ymin}-{region.ymax}. Declare a larger region, "
        f"{declare} or more, or data.region: whole."
    )


def read_observation_metadata(
    path: str | Path,
    *,
    hdu: int | str | None = None,
    declarations: Mapping[str, Any] | None = None,
) -> ObservationMetadata:
    """Read the metadata of the image of an observation from its header.

    No pixel is read. Without declarations, the metadata of a file of at
    least :data:`REUSE_MIN_BYTES` are kept while the file is unchanged, and
    every opening of the file keeps them too.

    Parameters
    ----------
    path : str or pathlib.Path
        FITS file of the observation.
    hdu : int or str, optional
        Index or ``EXTNAME`` of the HDU that holds the image. See
        :meth:`baymorph.core.image.ImageData.from_fits` for the default.
    declarations : mapping, optional
        Values of ``data.calibration``, which take the place of the readings
        of the header.

    Returns
    -------
    ObservationMetadata
        Metadata of the whole image.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the HDU cannot be chosen, the pixels are not linear, or the
        declarations are not valid.
    """
    resolved = _existing_file(path)

    if declarations:
        return _metadata_of_headers(
            read_image_headers(resolved, hdu, declarations=declarations)
        )

    state = file_state(resolved)
    key = (str(resolved), hdu, *state)
    metadata = _READ_METADATA.get(key)

    if metadata is None:
        metadata = _metadata_of_headers(read_image_headers(resolved, hdu))

        if state[2] >= REUSE_MIN_BYTES:
            _keep_metadata(key, metadata)

    return metadata


def working_region(data: Mapping[str, Any]) -> WorkingRegion | None:
    """Return the working region of the image of a configuration.

    Parameters
    ----------
    data : mapping
        The ``data`` section of a configuration.

    Returns
    -------
    WorkingRegion or None
        The region, or None when the whole image is read.

    Raises
    ------
    ValueError
        As :meth:`RegionRequest.resolve`.
    """
    metadata = read_observation_metadata(data["image_path"], hdu=data.get("hdu"))

    return RegionRequest.from_data(data).resolve(metadata)


def open_observation(
    path: str | Path,
    *,
    hdu: int | str | None = None,
    declarations: Mapping[str, Any] | None = None,
    region: RegionRequest | WorkingRegion | None = None,
) -> Observation:
    """Open the image of an observation.

    A file of at least :data:`REUSE_MIN_BYTES` is read once per process and
    reused while its state, :func:`file_state`, is unchanged; a smaller file
    is read at every call. The image array is read-only.

    With declarations, the metadata of the observation come from the reading
    of its header with them (AD-053); the image is the same.

    Parameters
    ----------
    path : str or pathlib.Path
        FITS file of the observation.
    hdu : int or str, optional
        Index or ``EXTNAME`` of the HDU that holds the image, the
        ``data.hdu`` of a configuration. See
        :meth:`baymorph.core.image.ImageData.from_fits` for the default.
    declarations : mapping, optional
        Values of ``data.calibration``, which take the place of the readings
        of the header.
    region : RegionRequest or WorkingRegion, optional
        Working region to read (AD-053): the settings that decide it, which
        are resolved within the same reading of the header, or the region
        itself. By default the whole image is read.

    Returns
    -------
    Observation
        Image of the selected HDU, with its header and metadata.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the declarations are not valid, or the working region cannot be
        resolved.
    """
    resolved = _existing_file(path)
    state = file_state(resolved)
    reusable = state[2] >= REUSE_MIN_BYTES

    if isinstance(region, RegionRequest):
        known = _READ_METADATA.get((str(resolved), hdu, *state))

        if known is not None:
            region = region.resolve(known)

    if isinstance(region, RegionRequest):
        request = region
        observation = Observation.from_fits(
            resolved,
            hdu=hdu,
            region=lambda headers: request.resolve(_metadata_of_headers(headers)),
        )
        key = (str(resolved), hdu, observation.image.region, *state)
        observation = _keep_observation(key, observation, reusable=reusable)
    else:
        key = (str(resolved), hdu, region, *state)

        if reusable and key in _OPENED:
            _OPENED.move_to_end(key)
            observation = _OPENED[key]
        else:
            observation = _keep_observation(
                key,
                Observation.from_fits(resolved, hdu=hdu, region=region),
                reusable=reusable,
            )

    if not declarations:
        return observation

    image = observation.image
    reading = read_header(
        image.header, primary_header=image.primary_header, declarations=declarations
    )

    return Observation(
        image=image,
        metadata=ObservationMetadata.from_header(
            image.header,
            image_shape=image.full_shape,
            primary_header=image.primary_header,
            reading=reading,
        ),
    )


def open_configured_observation(
    data: Mapping[str, Any], *, declarations: bool = True
) -> Observation:
    """Open the image of a configuration, within its working region.

    Parameters
    ----------
    data : mapping
        The ``data`` section of a configuration: ``image_path``, ``hdu``,
        ``calibration``, ``region``, ``target_position`` and ``pixel_scale``.
    declarations : bool, default=True
        Whether the metadata take ``data.calibration``; the pixels are the
        same either way.

    Returns
    -------
    Observation
        The image, read within its working region.

    Raises
    ------
    FileNotFoundError
        If the image does not exist.
    ValueError
        As :func:`open_observation`.
    """
    return open_observation(
        data["image_path"],
        hdu=data.get("hdu"),
        declarations=data.get("calibration") if declarations else None,
        region=RegionRequest.from_data(data),
    )


def forget_opened_observations() -> None:
    """Release every observation and every metadata kept for reuse.

    The next call of :func:`open_observation` reads its file again.
    """
    _OPENED.clear()
    _READ_METADATA.clear()


def _existing_file(path: str | Path) -> Path:
    """Resolve the path of an existing FITS file."""
    resolved = Path(path).expanduser().resolve()

    if not resolved.is_file():
        raise FileNotFoundError(f"FITS file not found: {resolved}")

    return resolved


def _metadata_of_headers(headers: ImageHeader) -> ObservationMetadata:
    """Return the metadata of the whole image from its headers."""
    return ObservationMetadata.from_header(
        headers.header, primary_header=headers.primary_header, reading=headers.reading
    )


def _keep_observation(
    key: tuple, observation: Observation, *, reusable: bool
) -> Observation:
    """Make an opened observation read-only and keep a large one for reuse."""
    observation.image.data.flags.writeable = False

    if reusable:
        if key in _OPENED:
            _OPENED.move_to_end(key)

            return _OPENED[key]

        _OPENED[key] = observation

        while len(_OPENED) > MAX_OPEN_OBSERVATIONS:
            _OPENED.popitem(last=False)

        path, hdu, _, *state = key
        _keep_metadata((path, hdu, *state), observation.metadata)

    return observation


def _keep_metadata(key: tuple, metadata: ObservationMetadata) -> None:
    """Keep the metadata of a large file, keyed by path, HDU and state."""
    _READ_METADATA[key] = metadata
    _READ_METADATA.move_to_end(key)

    while len(_READ_METADATA) > 4 * MAX_OPEN_OBSERVATIONS:
        _READ_METADATA.popitem(last=False)


def _region_center(target: Any) -> str | tuple[float, float] | None:
    """Return the position of a target setting, in the form of a request."""
    if target is None:
        return None

    if isinstance(target, str):
        if target.strip().lower() != "center":
            raise ValueError(
                "data.target_position must be 'center' or two coordinates; got "
                f"{target!r}."
            )

        return "center"

    if isinstance(target, Mapping):
        coordinates = (target.get("x"), target.get("y"))
    elif isinstance(target, Sequence) and len(target) == 2:
        coordinates = tuple(target)
    else:
        raise ValueError(
            "data.target_position must be 'center' or two coordinates; got "
            f"{target!r}."
        )

    values = []

    for value in coordinates:
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ValueError(
                "data.target_position must be 'center' or two finite "
                f"coordinates; got {target!r}."
            )

        values.append(float(value))

    return (values[0], values[1])


def _side_in_pixels(size: float, pixel_scale: float) -> int:
    """Return the number of pixels that spans a side in arcsec."""
    ratio = size / pixel_scale

    return max(1, math.ceil(ratio - 1.0e-9 * ratio))


__all__ = [
    "MAX_OPEN_OBSERVATIONS",
    "REUSE_MIN_BYTES",
    "RegionRequest",
    "WHOLE_IMAGE_MAX_SIDE",
    "describe_working_region",
    "file_state",
    "forget_opened_observations",
    "open_configured_observation",
    "open_observation",
    "read_observation_metadata",
    "region_request_message",
    "validate_region",
    "working_region",
]
