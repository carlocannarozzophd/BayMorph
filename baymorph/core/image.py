"""Image containers and FITS I/O."""

from collections.abc import Mapping
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
    """

    data: np.ndarray
    header: fits.Header | None = None
    path: str | None = None
    hdu_index: int = 0
    hdu_name: str | None = None
    primary_header: fits.Header | None = None
    header_reading: HeaderReading | None = None

    @classmethod
    def from_fits(
        cls, filename: str | Path, hdu: int | str | None = None
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

        Returns
        -------
        ImageData
            Image container initialized from the selected HDU.

        Raises
        ------
        ValueError
            If the selected HDU holds no two-dimensional image, if no HDU is
            selected and the file offers no image or several, or if the
            header declares pixel values that are not linear in flux, such as
            a Pan-STARRS stack compressed through ``BSOFTEN`` and
            ``BOFFSET`` (AD-049).
        """
        filename = Path(filename)

        with fits.open(filename) as hdul:
            selected = _read_image_header(hdul, hdu, filename=filename)
            data = hdul[selected.index].data.astype(float)

        return cls(
            data=data,
            header=selected.header,
            path=str(filename),
            hdu_index=selected.index,
            hdu_name=selected.name,
            primary_header=selected.primary_header,
            header_reading=selected.reading,
        )

    def to_fits(self, filename: str | Path, overwrite: bool = True) -> None:
        """Write the image to a FITS file.

        Parameters
        ----------
        filename : str or pathlib.Path
            Output FITS filename.
        overwrite : bool, default=True
            If True, overwrite an existing file.
        """
        filename = Path(filename)

        hdu = fits.PrimaryHDU(data=self.data, header=self.header)

        hdu.writeto(filename, overwrite=overwrite)

    @property
    def shape(self) -> tuple[int, int]:
        """tuple of int : Shape of the image array as ``(ny, nx)``."""
        return self.data.shape


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
    filename = Path(filename)

    with fits.open(filename) as hdul:
        return _read_image_header(
            hdul, hdu, filename=filename, declarations=declarations
        ).reading


class _ImageHeader(NamedTuple):
    """The HDU that holds the image, its headers and their reading."""

    index: int
    name: str | None
    header: fits.Header
    primary_header: fits.Header | None
    reading: HeaderReading


def _read_image_header(
    hdul: fits.HDUList,
    hdu: int | str | None,
    *,
    filename: Path,
    declarations: Mapping[str, Any] | None = None,
) -> _ImageHeader:
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

    return _ImageHeader(
        index=index,
        name=name,
        header=header,
        primary_header=primary_header,
        reading=reading,
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
