"""Image containers and FITS I/O."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from astropy.io import fits

from baymorph.core.header_metadata import (
    HeaderReading,
    ensure_linear_pixels,
    read_header,
)

SCIENCE_EXTENSION_NAMES = ("SCI", "IMAGE")
"""Extension names that hold the science image, in order of preference."""


@dataclass(frozen=True)
class WorkingRegion:
    """Part of an image that is read in place of the whole image (AD-053).

    The bounds are half-open pixel indices of the whole image, like the bounds
    of a cutout: the region holds the pixels ``[ymin:ymax, xmin:xmax]`` of an
    image of ``image_width`` x ``image_height`` pixels. Positions keep the
    coordinates of the whole image: its pixel ``(x, y)`` is the element
    ``[y - ymin, x - xmin]`` of the array of the region.

    Parameters
    ----------
    xmin, xmax : int
        Columns of the region, with ``0 <= xmin < xmax <= image_width``.
    ymin, ymax : int
        Rows of the region, with ``0 <= ymin < ymax <= image_height``.
    image_width, image_height : int
        Size of the whole image, in pixels.

    Raises
    ------
    ValueError
        If a bound or a size is not an integer, or the region does not lie
        inside the image.
    """

    xmin: int
    xmax: int
    ymin: int
    ymax: int
    image_width: int
    image_height: int

    def __post_init__(self) -> None:
        """Validate the bounds and store them as Python integers."""
        for name in ("xmin", "xmax", "ymin", "ymax", "image_width", "image_height"):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(
                    f"The {name} of a working region must be an integer; "
                    f"got {value!r}."
                )

            object.__setattr__(self, name, int(value))

        if not (
            0 <= self.xmin < self.xmax <= self.image_width
            and 0 <= self.ymin < self.ymax <= self.image_height
        ):
            raise ValueError(
                f"The working region x {self.xmin}-{self.xmax}, y "
                f"{self.ymin}-{self.ymax} does not lie inside an image of "
                f"{self.image_width} x {self.image_height} pixels."
            )

    @classmethod
    def about(
        cls,
        image_shape: tuple[int, int],
        *,
        center_x: float,
        center_y: float,
        side: int,
    ) -> "WorkingRegion":
        """Return the region of ``side`` pixels about a position.

        Each axis of the region holds ``side`` pixels, or the whole axis when
        the image is smaller. The region is centred on the pixel nearest the
        position, as a cutout is, shifted inside the image where it would
        leave it, and never padded.

        Parameters
        ----------
        image_shape : tuple of int
            Shape of the whole image as ``(ny, nx)``.
        center_x, center_y : float
            Position about which the region is taken, in pixels of the whole
            image.
        side : int
            Side of the region, in pixels.

        Returns
        -------
        WorkingRegion
            The region, inside the image.

        Raises
        ------
        ValueError
            If the shape or the side are not positive, or the position is not
            finite or lies outside the image.
        """
        height, width = (int(length) for length in image_shape)

        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive.")

        if isinstance(side, bool) or int(side) != side or side <= 0:
            raise ValueError(
                f"The side of a working region must be a positive integer; "
                f"got {side!r}."
            )

        if not (np.isfinite(center_x) and np.isfinite(center_y)):
            raise ValueError("The centre of a working region must be finite.")

        if not (-0.5 <= center_x <= width - 0.5 and -0.5 <= center_y <= height - 0.5):
            raise ValueError(
                f"The position ({center_x:g}, {center_y:g}) about which the "
                f"working region is taken lies outside the image of {width} x "
                f"{height} pixels."
            )

        xmin, xmax = _region_axis(center_x, int(side), width)
        ymin, ymax = _region_axis(center_y, int(side), height)

        return cls(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            image_width=width,
            image_height=height,
        )

    @property
    def shape(self) -> tuple[int, int]:
        """tuple of int : Shape of the region as ``(ny, nx)``."""
        return (self.ymax - self.ymin, self.xmax - self.xmin)

    @property
    def image_shape(self) -> tuple[int, int]:
        """tuple of int : Shape of the whole image as ``(ny, nx)``."""
        return (self.image_height, self.image_width)

    @property
    def is_whole(self) -> bool:
        """bool : Whether the region holds the whole image."""
        return self.shape == self.image_shape

    def contains(self, *, xmin: int, xmax: int, ymin: int, ymax: int) -> bool:
        """Return whether the region holds a box of the whole image.

        Parameters
        ----------
        xmin, xmax, ymin, ymax : int
            Half-open bounds of the box, in pixels of the whole image.

        Returns
        -------
        bool
            True when every pixel of the box lies in the region.
        """
        return (
            self.xmin <= xmin < xmax <= self.xmax
            and self.ymin <= ymin < ymax <= self.ymax
        )

    def local_slices(
        self, *, xmin: int, xmax: int, ymin: int, ymax: int
    ) -> tuple[slice, slice]:
        """Return the slices of a box of the whole image in the region array.

        Parameters
        ----------
        xmin, xmax, ymin, ymax : int
            Half-open bounds of the box, in pixels of the whole image.

        Returns
        -------
        tuple of slice
            Row and column slices, ``array[rows, columns]``.

        Raises
        ------
        ValueError
            If the box does not lie in the region.
        """
        if not self.contains(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax):
            raise ValueError(
                f"The box x {xmin}-{xmax}, y {ymin}-{ymax} does not lie in the "
                f"working region x {self.xmin}-{self.xmax}, y "
                f"{self.ymin}-{self.ymax}."
            )

        return (
            slice(ymin - self.ymin, ymax - self.ymin),
            slice(xmin - self.xmin, xmax - self.xmin),
        )

    def as_dict(self) -> dict[str, int]:
        """Return the bounds, with the keys of cutout bounds."""
        return {
            "xmin": self.xmin,
            "xmax": self.xmax,
            "ymin": self.ymin,
            "ymax": self.ymax,
        }


def _region_axis(center: float, side: int, length: int) -> tuple[int, int]:
    """Return the bounds of a region along one axis of the image."""
    size = min(side, length)
    start = int(round(center)) - size // 2
    start = min(max(start, 0), length - size)

    return start, start + size


@dataclass
class ImageData:
    """Container for astronomical image data.

    Parameters
    ----------
    data : ndarray
        Two-dimensional image array.
    header : astropy.io.fits.Header, optional
        FITS header of the HDU that holds the image.
    path : str, optional
        Path to the image on disk. This is informational and is not updated
        automatically when writing the image to a new file.
    hdu_index : int, default=0
        Index of the HDU the image was read from.
    hdu_name : str, optional
        ``EXTNAME`` of that HDU, when it has one.
    primary_header : astropy.io.fits.Header, optional
        Primary header of the file, kept only when the image is stored in an
        extension. Its keywords are inherited where the image header lacks
        them (AD-049).
    header_reading : HeaderReading, optional
        Reading of ``header`` and ``primary_header`` made when the image was
        read from a FITS file, so that the header is read once per opening.
        None for an image built in memory.
    region : WorkingRegion, optional
        Part of the image that ``data`` holds, when the image was read within
        a working region (AD-053). None when ``data`` is the whole image. The
        header, and every position, describe the whole image.

    Raises
    ------
    ValueError
        If ``data`` does not have the shape of ``region``.
    """

    data: np.ndarray
    header: fits.Header | None = None
    path: str | None = None
    hdu_index: int = 0
    hdu_name: str | None = None
    primary_header: fits.Header | None = None
    header_reading: HeaderReading | None = None
    region: WorkingRegion | None = None

    def __post_init__(self) -> None:
        """Require the array to be the region it is said to hold."""
        if self.region is not None and tuple(np.shape(self.data)) != self.region.shape:
            raise ValueError(
                f"The image array of shape {tuple(np.shape(self.data))} is not "
                f"the working region of shape {self.region.shape} it is said "
                "to hold."
            )

    @classmethod
    def from_fits(
        cls,
        filename: str | Path,
        hdu: int | str | None = None,
        *,
        region: (
            WorkingRegion | Callable[[ImageHeader], WorkingRegion | None] | None
        ) = None,
    ) -> "ImageData":
        """Load an image from a FITS file.

        Parameters
        ----------
        filename : str or pathlib.Path
            Path to the FITS file.
        hdu : int or str, optional
            Index or ``EXTNAME`` of the HDU that holds the image. By default
            the primary HDU is used when it holds an image; otherwise the
            single extension named ``SCI`` or ``IMAGE``, or the single
            two-dimensional image extension.
        region : WorkingRegion or callable, optional
            Part of the image to read (AD-053), or a function that returns it
            from the headers of the image, :class:`ImageHeader`, or None for
            the whole image, so that the region can depend on the size of the
            image without reading the header twice. Only the pixels of the
            region are read: the bytes that hold them in an uncompressed
            file, the tiles that cross it in a tile-compressed one. By
            default the whole image is read.

        Returns
        -------
        ImageData
            Image container initialized from the selected HDU.

        Raises
        ------
        ValueError
            If the selected HDU holds no two-dimensional image, if no HDU is
            selected and the file offers no image or several, if the header
            declares pixel values that are not linear in flux, such as a
            Pan-STARRS stack compressed through ``BSOFTEN`` and ``BOFFSET``
            (AD-049), or if ``region`` was taken in an image of another size.
        """
        filename = Path(filename)

        with fits.open(filename) as hdul:
            selected = _read_image_header(hdul, hdu, filename=filename)
            pixels = hdul[selected.index]

            if callable(region):
                region = region(selected)

            if region is None:
                data = pixels.data.astype(float)
            else:
                _require_region_of_image(
                    region, selected.header, image=f"HDU {selected.index} of {filename}"
                )
                data = pixels.section[
                    region.ymin : region.ymax, region.xmin : region.xmax
                ].astype(float)

        return cls(
            data=data,
            header=selected.header,
            path=str(filename),
            hdu_index=selected.index,
            hdu_name=selected.name,
            primary_header=selected.primary_header,
            header_reading=selected.reading,
            region=region,
        )

    def to_fits(self, filename: str | Path, overwrite: bool = True) -> None:
        """Write the image to a FITS file.

        Parameters
        ----------
        filename : str or pathlib.Path
            Output FITS filename.
        overwrite : bool, default=True
            If True, overwrite an existing file.

        Raises
        ------
        ValueError
            If the image was read within a working region, whose header
            describes the whole image.
        """
        if self.region is not None:
            raise ValueError(
                "The image holds only a working region of the file it was "
                "read from, and its header describes the whole image, so it "
                "cannot be written as a FITS image."
            )

        filename = Path(filename)

        hdu = fits.PrimaryHDU(data=self.data, header=self.header)

        hdu.writeto(filename, overwrite=overwrite)

    @property
    def shape(self) -> tuple[int, int]:
        """tuple of int : Shape of the image array as ``(ny, nx)``."""
        return self.data.shape

    @property
    def full_shape(self) -> tuple[int, int]:
        """tuple of int : Shape of the whole image as ``(ny, nx)``.

        It is the shape of the array, unless the array holds a working
        region.
        """
        if self.region is None:
            return self.data.shape

        return self.region.image_shape


def read_image_header(
    filename: str | Path,
    hdu: int | str | None = None,
    *,
    declarations: Mapping[str, Any] | None = None,
) -> HeaderReading:
    """Read the header of the image of a FITS file without reading its pixels.

    The HDU is chosen and its header read as :meth:`ImageData.from_fits` does,
    so the reading equals the one of the opened image, but no pixel is read: a
    mosaic of several gigabytes costs the same as a small stamp.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the FITS file.
    hdu : int or str, optional
        Index or ``EXTNAME`` of the HDU that holds the image. See
        :meth:`ImageData.from_fits` for the default.
    declarations : mapping, optional
        Values of ``data.calibration``, which take the place of the readings
        of the header (AD-053).

    Returns
    -------
    HeaderReading
        Reading of the header of the image, with the primary header inherited
        when the image is stored in an extension.

    Raises
    ------
    ValueError
        As :meth:`ImageData.from_fits`, when the HDU cannot be chosen or the
        pixel values are not linear in flux, or the declarations are not
        valid.
    """
    return read_image_headers(filename, hdu, declarations=declarations).reading


class ImageHeader(NamedTuple):
    """The HDU that holds the image of a FITS file, its headers and their reading.

    Attributes
    ----------
    index : int
        Index of the HDU.
    name : str or None
        Its ``EXTNAME``, for an extension that has one.
    header : astropy.io.fits.Header
        Copy of its header.
    primary_header : astropy.io.fits.Header or None
        Copy of the primary header, for an image stored in an extension.
    reading : HeaderReading
        Reading of the two headers.
    """

    index: int
    name: str | None
    header: fits.Header
    primary_header: fits.Header | None
    reading: HeaderReading


def read_image_headers(
    filename: str | Path,
    hdu: int | str | None = None,
    *,
    declarations: Mapping[str, Any] | None = None,
) -> ImageHeader:
    """Read the headers of the image of a FITS file without reading its pixels.

    As :func:`read_image_header`, but returning the HDU and the headers
    themselves beside their reading.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the FITS file.
    hdu : int or str, optional
        Index or ``EXTNAME`` of the HDU that holds the image. See
        :meth:`ImageData.from_fits` for the default.
    declarations : mapping, optional
        Values of ``data.calibration``, which take the place of the readings
        of the header (AD-053).

    Returns
    -------
    ImageHeader
        The HDU of the image, its headers and their reading.

    Raises
    ------
    ValueError
        As :func:`read_image_header`.
    """
    filename = Path(filename)

    with fits.open(filename) as hdul:
        return _read_image_header(
            hdul, hdu, filename=filename, declarations=declarations
        )


def _read_image_header(
    hdul: fits.HDUList,
    hdu: int | str | None,
    *,
    filename: Path,
    declarations: Mapping[str, Any] | None = None,
) -> ImageHeader:
    """Choose the image HDU and read its headers, refusing non-linear pixels."""
    index = select_image_hdu(hdul, hdu)
    selected = hdul[index]
    header = selected.header.copy()
    name = None
    primary_header = None

    if index > 0:
        name = str(selected.name).strip() or None
        primary_header = hdul[0].header.copy()

    reading = read_header(
        header, primary_header=primary_header, declarations=declarations
    )
    ensure_linear_pixels(reading, image=f"the image in HDU {index} of {filename}")

    return ImageHeader(
        index=index,
        name=name,
        header=header,
        primary_header=primary_header,
        reading=reading,
    )


def _require_region_of_image(
    region: WorkingRegion, header: fits.Header, *, image: str
) -> None:
    """Refuse a working region taken in an image of another size."""
    size = (header.get("NAXIS1"), header.get("NAXIS2"))

    if size != (region.image_width, region.image_height):
        raise ValueError(
            "The working region was taken in an image of "
            f"{region.image_width} x {region.image_height} pixels, but "
            f"{image} holds {size[0]} x {size[1]} pixels."
        )


def select_image_hdu(hdul: fits.HDUList, hdu: int | str | None = None) -> int:
    """Return the index of the HDU that holds the image.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        Opened FITS file.
    hdu : int or str, optional
        Index or ``EXTNAME`` of the HDU. Names are compared without regard to
        case.

    Returns
    -------
    int
        Index of the HDU.

    Raises
    ------
    ValueError
        If the requested HDU does not exist or holds no two-dimensional
        image, or if the choice is not unique when no HDU is requested.
    """
    if hdu is not None:
        index = _requested_index(hdul, hdu)

        if not _holds_image(hdul[index]):
            raise ValueError(
                f"HDU {hdu!r} of the FITS file holds no two-dimensional image."
            )

        return index

    if _holds_image(hdul[0]):
        return 0

    images = [index for index in range(1, len(hdul)) if _holds_image(hdul[index])]

    for name in SCIENCE_EXTENSION_NAMES:
        named = [index for index in images if str(hdul[index].name).upper() == name]

        if len(named) == 1:
            return named[0]

    if len(images) == 1:
        return images[0]

    if not images:
        raise ValueError("The FITS file holds no two-dimensional image.")

    listed = ", ".join(
        f"{index} ({str(hdul[index].name).strip() or 'unnamed'})" for index in images
    )

    raise ValueError(
        "The FITS file holds several two-dimensional images, in HDUs "
        f"{listed}; declare the one to analyse in data.hdu."
    )


def _requested_index(hdul: fits.HDUList, hdu: int | str) -> int:
    """Resolve an index or name to the index of an existing HDU."""
    if isinstance(hdu, bool):
        raise ValueError("data.hdu must be an index or an extension name.")

    if isinstance(hdu, int):
        if not 0 <= hdu < len(hdul):
            raise ValueError(
                f"HDU {hdu} does not exist: the FITS file has {len(hdul)} HDUs."
            )

        return hdu

    name = str(hdu).strip().upper()
    matches = [
        index
        for index in range(len(hdul))
        if str(hdul[index].name).strip().upper() == name
    ]

    if not matches:
        raise ValueError(f"The FITS file has no HDU named {hdu!r}.")

    if len(matches) > 1:
        raise ValueError(
            f"The FITS file has several HDUs named {hdu!r}; declare data.hdu "
            "by index."
        )

    return matches[0]


def _holds_image(hdu: object) -> bool:
    """Return whether an HDU holds a two-dimensional image."""
    if not isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)):
        return False

    header = hdu.header

    return header.get("NAXIS") == 2 and all(
        int(header.get(f"NAXIS{axis}", 0)) > 0 for axis in (1, 2)
    )
