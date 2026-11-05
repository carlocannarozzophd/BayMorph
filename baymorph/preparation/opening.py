"""The opening stage of the preparation: the image of an observation.

The opening is the first stage. It takes the ``data`` section of a
configuration, as :mod:`baymorph.preparation.schema` reads it, and returns an
:class:`OpenedObservation`:

- the file, with the identity of its content and its path relative to the
  project as information only;
- the HDU that holds the image;
- the headers and their reading through :mod:`baymorph.preparation.header`,
  with the declared calibration (AD-053);
- the working region (AD-053) and its pixels, read once and read-only;
- the pixels without data;
- the pixel scale, declared or from the celestial WCS, with a declared value
  that contradicts the WCS reported;
- two identities: the identity of the pixels, for the stages that use only
  the pixels, and the identity of the whole opening, which also covers the
  calibration.

Astropy decompresses a file compressed as a whole, such as a ``.gz``, from its
start whenever it opens it, even to read one header. The headers of such a
file are read here by decompressing only up to the header needed, and the
headers of every file are kept per process while the file is unchanged, so a
reading of the headers of a mosaic of several gigabytes costs little. Its
pixels are still read through Astropy.

Every position, of the target, of the region or of a pixel, is in the
zero-based pixels of the whole image, ``x`` along the columns; the pixel
``(x, y)`` of the region is the element ``[y - ymin, x - xmin]`` of its array.
"""

from __future__ import annotations

import bz2
import gzip
import math
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray
from scipy.ndimage import label

from baymorph.preparation.header import (
    HeaderReading,
    conventions_digest,
    ensure_linear_pixels,
    load_header_conventions,
    read_header,
    validate_declarations,
)
from baymorph.preparation.identity import (
    FileIdentity,
    file_identity,
    stage_identity,
)
from baymorph.preparation.store import StageProduct

OPENING_STAGE = "opening"
"""Name of the stage in the product store."""

OPENING_IMAGE_STAGE = "opening_image"
"""Name under which the identity of the pixels of an opening is computed."""

OPENING_VERSION = 1
"""Version of the rules of the opening; raising it changes every identity."""

WHOLE_IMAGE_MAX_SIDE = 4096
"""Largest side, in pixels, of an image read whole by default (AD-053)."""

NO_DATA_MIN_ZERO_PIXELS = 100
"""Smallest area of exact zeros, in pixels, that holds no data (AD-053)."""

SCIENCE_EXTENSION_NAMES = ("SCI", "IMAGE")
"""Extension names that hold the science image, in order of preference."""

PIXEL_SCALE_TOLERANCE = 1.0e-3
"""Relative difference above which a declared pixel scale and the scale of
the WCS of the header are reported as contradicting each other."""

MAX_KEPT_FILES = 8
"""Number of files whose headers are kept for reuse within one process."""

_BLOCK = 2880
_CARD = 80
_SKIP_CHUNK = 4 * 1024 * 1024
_IMAGE_KINDS = frozenset({"primary", "image", "compressed_image"})


# --- The working region -----------------------------------------------------


@dataclass(frozen=True)
class WorkingRegion:
    """Part of an image that is read in place of the whole image (AD-053).

    The bounds are half-open pixel indices of the whole image, like the bounds
    of a cutout: the region holds the pixels ``[ymin:ymax, xmin:xmax]`` of an
    image of ``image_width`` x ``image_height`` pixels.

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

            if isinstance(value, bool) or not isinstance(value, Integral):
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
    ) -> WorkingRegion:
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

        if not (math.isfinite(center_x) and math.isfinite(center_y)):
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

    def expanded(self, margin: int) -> WorkingRegion:
        """Return the region grown by ``margin`` pixels on every side.

        The grown region is clipped to the image.

        Parameters
        ----------
        margin : int
            Non-negative number of pixels.

        Returns
        -------
        WorkingRegion
            The grown region.
        """
        if isinstance(margin, bool) or not isinstance(margin, Integral) or margin < 0:
            raise ValueError("The margin of a region is a non-negative integer.")

        return WorkingRegion(
            xmin=max(self.xmin - margin, 0),
            xmax=min(self.xmax + margin, self.image_width),
            ymin=max(self.ymin - margin, 0),
            ymax=min(self.ymax + margin, self.image_height),
            image_width=self.image_width,
            image_height=self.image_height,
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


def _side_in_pixels(size: float, pixel_scale: float) -> int:
    """Return the number of pixels that spans a side in arcsec."""
    ratio = size / pixel_scale

    return max(1, math.ceil(ratio - 1.0e-9 * ratio))


# --- The headers of a file ---------------------------------------------------


@dataclass(frozen=True)
class HduHeader:
    """The header of one HDU of a FITS file.

    Parameters
    ----------
    index : int
        Index of the HDU in the file.
    name : str
        Name of the HDU as Astropy gives it: ``PRIMARY`` for the primary HDU,
        the ``EXTNAME`` of an extension, or an empty text.
    kind : {"primary", "image", "compressed_image", "table", "other"}
        What the HDU holds.
    header : astropy.io.fits.Header
        The header; for a tile-compressed image, the header of the image.
    """

    index: int
    name: str
    kind: str
    header: fits.Header = field(repr=False)

    @property
    def holds_image(self) -> bool:
        """bool : Whether the HDU holds a two-dimensional image."""
        if self.kind not in _IMAGE_KINDS:
            return False

        if self.header.get("NAXIS") != 2:
            return False

        return all(
            _positive_integer(self.header.get(f"NAXIS{axis}")) for axis in (1, 2)
        )

    @property
    def extension_name(self) -> str | None:
        """str or None : The ``EXTNAME`` of an extension, None otherwise."""
        if self.index == 0:
            return None

        return self.name.strip() or None


def _positive_integer(value: Any) -> bool:
    """Return whether a header value is a positive integer."""
    return isinstance(value, Integral) and not isinstance(value, bool) and value > 0


def _file_state(path: Path) -> tuple[int, int, int, int, int]:
    """Return the state that changes whenever the content of a file changes.

    It is the device, the inode, the size and the times of the last
    modification and of the last change of status, in nanoseconds. The
    second time is set by the system on every write and cannot be restored by
    copying tools.
    """
    status = path.stat()

    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def file_compression(path: str | Path) -> str | None:
    """Return how a FITS file is compressed as a whole, from its first bytes.

    Parameters
    ----------
    path : str or pathlib.Path
        The file.

    Returns
    -------
    {"gzip", "bzip2", "zip"} or None
        The compression, or None for a file that is not compressed as a
        whole, such as a plain or a tile-compressed FITS file.
    """
    with Path(path).open("rb") as stream:
        magic = stream.read(4)

    if magic[:2] == b"\x1f\x8b":
        return "gzip"

    if magic[:3] == b"BZh":
        return "bzip2"

    if magic == b"PK\x03\x04":
        return "zip"

    return None


class _HeaderScan:
    """The headers of one FITS file, read in order as they are needed."""

    def __init__(self, path: Path, compression: str | None) -> None:
        self.path = path
        self.compression = compression
        self.headers: list[HduHeader] = []
        self.complete = False

    def get(self, index: int) -> HduHeader | None:
        """Return the header of HDU ``index``, or None when there is none."""
        self._read(index + 1)

        return self.headers[index] if index < len(self.headers) else None

    def every(self) -> list[HduHeader]:
        """Return the headers of every HDU of the file."""
        self._read(None)

        return list(self.headers)

    def _read(self, count: int | None) -> None:
        """Read headers until ``count`` are known, or every header for None."""
        if self.complete or (count is not None and len(self.headers) >= count):
            return

        try:
            if self.compression in ("gzip", "bzip2"):
                headers, complete = _stream_headers(self.path, self.compression, count)
            else:
                headers, complete = _astropy_headers(self.path, count)
        except _NeedsAstropy:
            headers, complete = _astropy_headers(self.path, count)
        except (OSError, EOFError, ValueError) as error:
            raise ValueError(
                f"BayMorph cannot read the headers of {self.path}: "
                f"{type(error).__name__}: {error}"
            ) from error

        if len(headers) > len(self.headers):
            self.headers = headers

        self.complete = complete


class _NeedsAstropy(Exception):
    """Raised when a compressed file holds an HDU that Astropy must read."""


def _astropy_headers(path: Path, count: int | None) -> tuple[list[HduHeader], bool]:
    """Read headers through Astropy, which reads them lazily, in order."""
    headers: list[HduHeader] = []

    with fits.open(path, lazy_load_hdus=True) as hdul:
        index = 0

        while count is None or index < count:
            try:
                hdu = hdul[index]
            except IndexError:
                return headers, True

            headers.append(
                HduHeader(
                    index=index,
                    name=str(hdu.name),
                    kind=_astropy_kind(hdu),
                    header=hdu.header.copy(),
                )
            )
            index += 1

    return headers, False


def _astropy_kind(hdu: Any) -> str:
    """Return what an HDU opened by Astropy holds."""
    if isinstance(hdu, fits.PrimaryHDU):
        return "primary"

    if isinstance(hdu, fits.CompImageHDU):
        return "compressed_image"

    if isinstance(hdu, fits.ImageHDU):
        return "image"

    if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
        return "table"

    return "other"


def _stream_headers(
    path: Path, compression: str, count: int | None
) -> tuple[list[HduHeader], bool]:
    """Read the headers of a file compressed as a whole, decompressing in order.

    The data of every HDU before the last header needed is decompressed and
    discarded; nothing after it is decompressed.
    """
    opener = gzip.open if compression == "gzip" else bz2.open
    headers: list[HduHeader] = []

    with opener(path, "rb") as stream:
        index = 0

        while count is None or index < count:
            raw = _header_bytes(stream, first=index == 0)

            if raw is None:
                return headers, True

            text = raw.decode("ascii", errors="replace").replace("\ufffd", "?")
            header = fits.Header.fromstring(text)
            kind = _streamed_kind(header, index)

            if kind == "compressed_image":
                raise _NeedsAstropy

            name = "PRIMARY" if index == 0 else str(header.get("EXTNAME", ""))
            headers.append(HduHeader(index=index, name=name, kind=kind, header=header))

            if count is not None and len(headers) >= count:
                return headers, False

            _skip(stream, _data_bytes(header))
            index += 1

    return headers, False


def _header_bytes(stream: Any, *, first: bool) -> bytes | None:
    """Return the bytes of the next header, or None at the end of the file.

    After the last HDU, bytes that do not begin an extension end the file, as
    Astropy reads it.
    """
    blocks: list[bytes] = []

    while True:
        block = stream.read(_BLOCK)

        if not blocks:
            if not block:
                if first:
                    raise ValueError("the file is empty")

                return None

            if first and not block.startswith(b"SIMPLE"):
                raise ValueError("the file does not begin with a FITS header")

            if not first and not block.startswith(b"XTENSION"):
                return None

        if len(block) < _BLOCK:
            raise ValueError("the file ends inside a header")

        blocks.append(block)

        if any(
            block[start : start + 8] == b"END     " for start in range(0, _BLOCK, _CARD)
        ):
            return b"".join(blocks)


def _streamed_kind(header: fits.Header, index: int) -> str:
    """Return what an HDU holds, from its header alone."""
    if index == 0:
        return "primary"

    extension = str(header.get("XTENSION", "")).strip().upper()

    if extension == "IMAGE":
        return "image"

    if extension == "BINTABLE":
        return "compressed_image" if header.get("ZIMAGE") is True else "table"

    if extension == "TABLE":
        return "table"

    return "other"


def _data_bytes(header: fits.Header) -> int:
    """Return the number of bytes of the data of an HDU, padded to blocks."""
    bitpix = header.get("BITPIX")
    naxis = header.get("NAXIS", 0)

    if not isinstance(bitpix, Integral) or not isinstance(naxis, Integral):
        raise ValueError("a header has no integer BITPIX and NAXIS")

    if naxis == 0:
        return 0

    axes = [header.get(f"NAXIS{axis}", 0) for axis in range(1, naxis + 1)]

    if not all(isinstance(axis, Integral) and axis >= 0 for axis in axes):
        raise ValueError("a header has an axis length that is not an integer")

    if header.get("GROUPS") is True and axes[0] == 0:
        axes = axes[1:]

    size = (
        abs(int(bitpix))
        // 8
        * int(header.get("GCOUNT", 1))
        * (int(header.get("PCOUNT", 0)) + math.prod(int(axis) for axis in axes))
    )

    return -(-size // _BLOCK) * _BLOCK


def _skip(stream: Any, size: int) -> None:
    """Read and discard ``size`` bytes of a stream."""
    while size > 0:
        chunk = stream.read(min(size, _SKIP_CHUNK))

        if not chunk:
            raise ValueError("the file ends inside the data of an HDU")

        size -= len(chunk)


_SCANS: OrderedDict[tuple[Any, ...], _HeaderScan] = OrderedDict()


def _scan(path: Path) -> _HeaderScan:
    """Return the headers of a file, kept for reuse while it is unchanged."""
    key = (str(path), *_file_state(path))
    scan = _SCANS.get(key)

    if scan is None:
        scan = _HeaderScan(path, file_compression(path))
        _SCANS[key] = scan

        while len(_SCANS) > MAX_KEPT_FILES:
            _SCANS.popitem(last=False)
    else:
        _SCANS.move_to_end(key)

    return scan


def forget_kept_headers() -> None:
    """Release the headers kept for reuse; the next opening reads them again."""
    _SCANS.clear()


def _select_image_hdu(scan: _HeaderScan, requested: int | str | None) -> HduHeader:
    """Return the HDU that holds the image.

    Parameters
    ----------
    scan : _HeaderScan
        The headers of the file.
    requested : int, str or None
        ``data.hdu``: an index, an extension name compared without regard to
        case and surrounding spaces, or None. By default the primary HDU is
        used when it holds an image; otherwise the single extension named
        ``SCI`` or ``IMAGE``, or the single two-dimensional image extension.
        Only the headers needed are read: none after the primary one when it
        holds the image, none after the requested index.

    Returns
    -------
    HduHeader
        The HDU.

    Raises
    ------
    ValueError
        If the requested HDU does not exist or holds no two-dimensional
        image, or if the choice is not unique when no HDU is requested.
    """
    if isinstance(requested, bool):
        raise ValueError("data.hdu must be an index or an extension name.")

    if isinstance(requested, Integral):
        index = int(requested)
        chosen = scan.get(index) if index >= 0 else None

        if chosen is None:
            raise ValueError(
                f"HDU {index} does not exist: the FITS file has {len(scan.every())} "
                "HDUs."
            )

        if not chosen.holds_image:
            raise ValueError(
                f"HDU {requested!r} of the FITS file holds no two-dimensional image."
            )

        return chosen

    if requested is not None:
        name = str(requested).strip().upper()
        matches = [hdu for hdu in scan.every() if hdu.name.strip().upper() == name]

        if not matches:
            raise ValueError(f"The FITS file has no HDU named {requested!r}.")

        if len(matches) > 1:
            raise ValueError(
                f"The FITS file has several HDUs named {requested!r}; declare "
                "data.hdu by index."
            )

        if not matches[0].holds_image:
            raise ValueError(
                f"HDU {requested!r} of the FITS file holds no two-dimensional image."
            )

        return matches[0]

    primary = scan.get(0)

    if primary is not None and primary.holds_image:
        return primary

    images = [hdu for hdu in scan.every()[1:] if hdu.holds_image]

    for science in SCIENCE_EXTENSION_NAMES:
        named = [hdu for hdu in images if hdu.name.strip().upper() == science]

        if len(named) == 1:
            return named[0]

    if len(images) == 1:
        return images[0]

    if not images:
        raise ValueError("The FITS file holds no two-dimensional image.")

    listed = ", ".join(
        f"{hdu.index} ({hdu.name.strip() or 'unnamed'})" for hdu in images
    )

    raise ValueError(
        "The FITS file holds several two-dimensional images, in HDUs "
        f"{listed}; declare the one to analyse in data.hdu."
    )


@dataclass(frozen=True)
class ImageHeaders:
    """The HDU that holds the image of a file, its headers and their reading.

    Parameters
    ----------
    path : pathlib.Path
        The file, resolved.
    compression : str or None
        How the file is compressed as a whole, as :func:`file_compression`
        says.
    hdu : HduHeader
        The HDU of the image.
    primary_header : astropy.io.fits.Header or None
        The primary header, for an image stored in an extension; its keywords
        are inherited where the image header lacks them (AD-049).
    reading : HeaderReading
        The reading of the two headers, with the declarations.
    """

    path: Path
    compression: str | None
    hdu: HduHeader
    primary_header: fits.Header | None = field(repr=False)
    reading: HeaderReading = field(repr=False)

    @property
    def header(self) -> fits.Header:
        """astropy.io.fits.Header : The header of the image HDU."""
        return self.hdu.header

    @property
    def image_shape(self) -> tuple[int, int]:
        """tuple of int : Shape of the whole image as ``(ny, nx)``."""
        return (int(self.header["NAXIS2"]), int(self.header["NAXIS1"]))


def read_image_headers(
    path: str | Path,
    hdu: int | str | None = None,
    *,
    declarations: Mapping[str, Any] | None = None,
) -> ImageHeaders:
    """Read the headers of the image of a FITS file, without its pixels.

    The headers of a file are read once per process while the file is
    unchanged, and the declarations are applied to them without reading the
    file again.

    Parameters
    ----------
    path : str or pathlib.Path
        The FITS file.
    hdu : int, str or None, optional
        ``data.hdu``: an index, or an extension name compared without regard
        to case and surrounding spaces. By default the primary HDU is used
        when it holds an image; otherwise the single extension named ``SCI``
        or ``IMAGE``, or the single two-dimensional image extension. Only the
        headers needed are read: none after the primary one when it holds the
        image, none after the requested index.
    declarations : mapping, optional
        Values of ``data.calibration``, which take the place of the readings
        of the header (AD-053).

    Returns
    -------
    ImageHeaders
        The HDU, its headers and their reading.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the headers cannot be read, the HDU cannot be chosen, the pixel
        values are not linear in flux, or the declarations are not valid.
    """
    resolved = Path(path).expanduser().resolve()

    if not resolved.is_file():
        raise FileNotFoundError(f"FITS file not found: {resolved}")

    scan = _scan(resolved)
    chosen = _select_image_hdu(scan, hdu)
    primary = None

    if chosen.index > 0:
        primary_hdu = scan.get(0)
        primary = None if primary_hdu is None else primary_hdu.header.copy()

    image = HduHeader(
        index=chosen.index,
        name=chosen.name,
        kind=chosen.kind,
        header=chosen.header.copy(),
    )
    reading = read_header(
        image.header, primary_header=primary, declarations=declarations
    )
    ensure_linear_pixels(
        reading, image=f"the image in HDU {chosen.index} of {resolved}"
    )

    return ImageHeaders(
        path=resolved,
        compression=scan.compression,
        hdu=image,
        primary_header=primary,
        reading=reading,
    )


# --- The pixels ---------------------------------------------------------------


def _read_pixels(
    headers: ImageHeaders, window: WorkingRegion | None
) -> tuple[NDArray[np.float64], np.dtype]:
    """Read the pixels of a window of the image, or of the whole image.

    A file compressed as a whole can only be decompressed from its start, and
    Astropy goes back to its earlier place after each row of a section, so a
    section of such a file would be decompressed again for every row; its
    whole rows that cross the window are read at once instead.

    Returns the pixels in float64 and the type in which Astropy returned
    them, for a header whose BITPIX cannot say how they were stored.
    """
    index = headers.hdu.index

    try:
        with fits.open(headers.path) as hdul:
            hdu = hdul[index]

            if window is None:
                values = hdu.data

                if values is None:
                    raise ValueError("the HDU holds no data")
            else:
                rows = slice(window.ymin, window.ymax)
                columns = slice(window.xmin, window.xmax)
                section = hdu.section

                if hdul.fileinfo(index)["file"].compression is None:
                    values = section[rows, columns]
                else:
                    values = section[rows, :][:, columns]

            stored = np.asarray(values).dtype
            pixels = np.array(values, dtype=np.float64)
    except (OSError, EOFError, TypeError, ValueError) as error:
        raise ValueError(
            f"BayMorph cannot read the pixels of HDU {index} of {headers.path}: "
            f"{type(error).__name__}: {error}"
        ) from error

    expected = headers.image_shape if window is None else window.shape

    if pixels.shape != expected:
        raise ValueError(
            f"The pixels read from HDU {index} of {headers.path} have the shape "
            f"{pixels.shape}, not the {expected} of the header."
        )

    return pixels, stored


def no_data_mask(
    data: Any,
    *,
    floating_point: bool,
    min_zero_pixels: int = NO_DATA_MIN_ZERO_PIXELS,
) -> NDArray[np.bool_]:
    """Return the pixels of an image that hold no data (AD-053).

    A pixel holds no data when it is not finite or, in an image stored in
    floating point, when it belongs to an area of at least
    ``min_zero_pixels`` exact zeros connected along their edges. Mosaics and
    drizzled images fill the parts of their grid that no exposure covers with
    NaN or with zeros; a measured value in floating point is almost never an
    exact zero, while an integer image may hold real zeros, so the zero rule
    applies to floating-point images only.

    Parameters
    ----------
    data : array-like
        Two-dimensional image.
    floating_point : bool
        Whether the image is stored in floating point, ``BITPIX < 0`` in its
        FITS header.
    min_zero_pixels : int, default=NO_DATA_MIN_ZERO_PIXELS
        Smallest area of connected exact zeros that holds no data.

    Returns
    -------
    ndarray of bool
        Mask of the pixels without data, ``True`` = excluded.

    Raises
    ------
    ValueError
        If ``data`` is not a two-dimensional numeric array or
        ``min_zero_pixels`` is not a positive integer.
    """
    values = np.asarray(data)

    if values.ndim != 2 or not np.issubdtype(values.dtype, np.number):
        raise ValueError("'data' must be a two-dimensional numeric array.")

    if (
        isinstance(min_zero_pixels, bool)
        or not isinstance(min_zero_pixels, Integral)
        or min_zero_pixels < 1
    ):
        raise ValueError("'min_zero_pixels' must be a positive integer.")

    mask = ~np.isfinite(values)

    if not floating_point:
        return mask

    zeros = values == 0

    if np.count_nonzero(zeros) < min_zero_pixels:
        return mask

    areas, _ = label(zeros)
    large = np.bincount(areas.ravel()) >= min_zero_pixels
    large[0] = False

    return mask | large[areas]


def _floating_point(header: fits.Header, stored: np.dtype) -> bool:
    """Return whether the image is stored in floating point.

    It is ``BITPIX < 0``, or the type of the pixels as Astropy returned them
    for a header whose BITPIX is not an integer.
    """
    bitpix = header.get("BITPIX")

    if isinstance(bitpix, Integral) and not isinstance(bitpix, bool):
        return bool(bitpix < 0)

    return bool(np.issubdtype(stored, np.floating))


# --- The pixel scale ----------------------------------------------------------


@dataclass(frozen=True)
class PixelScale:
    """The pixel scale of an observation and where it comes from.

    Parameters
    ----------
    value : float or None
        Pixel scale in arcsec per pixel, or None when neither a declaration
        nor the WCS of the header gives one.
    source : {"declared", "wcs"} or None
        ``data.pixel_scale``, or the celestial WCS of the header.
    wcs_value : float or None
        The scale of the WCS of the header, when it is usable, whatever the
        source.
    note : str or None
        A contradiction between the declared value and the WCS, or why there
        is no scale.
    """

    value: float | None
    source: str | None
    wcs_value: float | None
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the scale for a record."""
        return {
            "value": self.value,
            "source": self.source,
            "wcs_value": self.wcs_value,
            "note": self.note,
            "unit": "arcsec/pixel",
        }


def resolve_pixel_scale(declared: Any, reading: HeaderReading) -> PixelScale:
    """Return the pixel scale from ``data.pixel_scale`` and the header.

    A declared scale takes the place of the scale of the WCS, which is kept
    beside it; when they differ by more than :data:`PIXEL_SCALE_TOLERANCE`
    the note reports it.

    Parameters
    ----------
    declared : float, "auto" or None
        ``data.pixel_scale``, as the schema returns it.
    reading : HeaderReading
        The reading of the header.

    Returns
    -------
    PixelScale
        The scale, its source and the scale of the WCS.
    """
    quantity = reading["pixel_scale"]
    wcs_value = float(quantity.value) if quantity.usable else None

    if declared is None or declared == "auto":
        if wcs_value is not None:
            return PixelScale(value=wcs_value, source="wcs", wcs_value=wcs_value)

        # Without a celestial WCS the reading has no note; with one that
        # cannot be used, the note says why.
        return PixelScale(value=None, source=None, wcs_value=None, note=quantity.note)

    value = float(declared)

    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(
            f"data.pixel_scale must be a finite positive number; got {declared!r}."
        )

    note = None

    if wcs_value is not None:
        difference = abs(value / wcs_value - 1.0)

        if difference > PIXEL_SCALE_TOLERANCE:
            note = (
                f"The declared data.pixel_scale, {value:.6g} arcsec per pixel, "
                f"differs from the {wcs_value:.6g} of the WCS of the header by "
                f"{100.0 * difference:.3g} %; the declared value is used."
            )

    return PixelScale(value=value, source="declared", wcs_value=wcs_value, note=note)


# --- The opened observation ---------------------------------------------------


@dataclass(frozen=True, eq=False)
class OpenedObservation:
    """The product of the opening: the image of an observation, read once.

    Parameters
    ----------
    headers : ImageHeaders
        The HDU of the image, its headers and their reading.
    file : FileIdentity
        Identity of the content of the file, with its path relative to the
        project as information.
    region : WorkingRegion or None
        The working region, or None when the whole image is read.
    pixels : ndarray of float64
        The pixels of the region, or of the whole image; read-only.
    no_data : ndarray of bool
        The pixels without data, of the same shape; read-only.
    floating_point : bool
        Whether the image is stored in floating point.
    pixel_scale : PixelScale
        The pixel scale and its source.
    image_identity : str
        Identity of the pixels: the content of the file, the HDU and the
        region. A stage that uses only the pixels depends on it.
    identity : str
        Identity of the whole opening, which adds the declared calibration,
        the declared pixel scale and the header conventions.
    record : Mapping
        What the opening read and decided, JSON-serializable.
    """

    headers: ImageHeaders = field(repr=False)
    file: FileIdentity
    region: WorkingRegion | None
    pixels: NDArray[np.float64] = field(repr=False)
    no_data: NDArray[np.bool_] = field(repr=False)
    floating_point: bool
    pixel_scale: PixelScale
    image_identity: str
    identity: str
    record: Mapping[str, Any] = field(repr=False)

    @property
    def reading(self) -> HeaderReading:
        """HeaderReading : The reading of the headers, with the declarations."""
        return self.headers.reading

    @property
    def image_shape(self) -> tuple[int, int]:
        """tuple of int : Shape of the whole image as ``(ny, nx)``."""
        return self.headers.image_shape

    @property
    def center(self) -> tuple[float, float]:
        """tuple of float : Centre of the whole image, ``(x, y)``."""
        height, width = self.image_shape

        return ((width - 1) / 2.0, (height - 1) / 2.0)

    def product(self) -> StageProduct:
        """Return the product of the stage, for the product store.

        The pixels and the mask of the pixels without data are not stored:
        they follow from the file, whose content the identity covers.
        """
        return StageProduct(
            stage=OPENING_STAGE,
            version=OPENING_VERSION,
            identity=self.identity,
            record=dict(self.record),
        )


def open_image(
    path: str | Path,
    *,
    project_root: str | Path,
    hdu: int | str | None = None,
    region: Any = None,
    target_position: Any = None,
    pixel_scale: Any = None,
    calibration: Mapping[str, Any] | None = None,
) -> OpenedObservation:
    """Open the image of an observation.

    Parameters
    ----------
    path : str or pathlib.Path
        The FITS file; a relative path is relative to ``project_root``.
    project_root : str or pathlib.Path
        The folder of the project, against which paths are recorded.
    hdu : int, str or None, optional
        ``data.hdu``; see :func:`read_image_headers`.
    region : None, str or mapping, optional
        ``data.region``; see :func:`validate_region`.
    target_position : None, "center" or mapping, optional
        ``data.target_position`` as the schema returns it, about which the
        working region of a large image is taken.
    pixel_scale : float, "auto" or None, optional
        ``data.pixel_scale``.
    calibration : mapping, optional
        ``data.calibration``, the declared calibration (AD-053).

    Returns
    -------
    OpenedObservation
        The opened image.

    Raises
    ------
    FileNotFoundError
        If the file or the project folder does not exist.
    ValueError
        If the HDU cannot be chosen, the pixel values are not linear, a
        setting is not valid, the working region cannot be resolved, or the
        pixels cannot be read.
    """
    root = Path(project_root).expanduser().resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"Project folder not found: {root}")

    location = Path(path).expanduser()

    if not location.is_absolute():
        location = root / location

    declarations = validate_declarations(calibration)
    setting = validate_region(region)
    headers = read_image_headers(location, hdu, declarations=declarations)
    scale = resolve_pixel_scale(pixel_scale, headers.reading)
    working, reason = resolve_working_region(
        setting,
        image_shape=headers.image_shape,
        target_position=target_position,
        pixel_scale=scale,
    )
    window = None if working is None else working.expanded(NO_DATA_MIN_ZERO_PIXELS)
    values, stored = _read_pixels(headers, window)
    floating_point = _floating_point(headers.header, stored)
    missing = no_data_mask(values, floating_point=floating_point)

    if working is not None:
        rows, columns = window.local_slices(**working.as_dict())
        values = values[rows, columns].copy()
        missing = missing[rows, columns].copy()

    values.flags.writeable = False
    missing.flags.writeable = False

    identity_of_file = file_identity(headers.path, project_root=root)
    region_record = None if working is None else working.as_dict()
    image_identity = stage_identity(
        stage=OPENING_IMAGE_STAGE,
        version=OPENING_VERSION,
        settings={
            "hdu_index": headers.hdu.index,
            "region": region_record,
            "no_data_min_zero_pixels": NO_DATA_MIN_ZERO_PIXELS,
        },
        inputs={"image": identity_of_file},
    )
    digest = conventions_digest(load_header_conventions())
    identity = stage_identity(
        stage=OPENING_STAGE,
        version=OPENING_VERSION,
        settings={
            "calibration": declarations,
            "pixel_scale": None if scale.source != "declared" else scale.value,
            "conventions": digest,
        },
        upstream={"image": image_identity},
    )
    count = int(np.count_nonzero(missing))
    record = {
        "stage": OPENING_STAGE,
        "version": OPENING_VERSION,
        "file": identity_of_file.as_record(),
        "compression": headers.compression or "none",
        "hdu": {
            "index": headers.hdu.index,
            "name": headers.hdu.extension_name,
            "requested": hdu,
        },
        "image": {
            "width": headers.image_shape[1],
            "height": headers.image_shape[0],
            "bitpix": _json_scalar(headers.header.get("BITPIX")),
            "floating_point": floating_point,
        },
        "region": {
            "setting": setting,
            "bounds": region_record,
            "reason": reason,
        },
        "pixel_scale": scale.as_dict(),
        "no_data": {
            "pixels": count,
            "fraction": count / missing.size,
            "rule": (
                "non-finite pixels, and areas of at least "
                f"{NO_DATA_MIN_ZERO_PIXELS} exact zeros connected along their "
                "edges in an image stored in floating point, counted on the "
                "whole image"
            ),
        },
        "reading": headers.reading.as_dict(),
        "conventions": {
            "matched": list(headers.reading.conventions),
            "digest": digest,
        },
        "image_identity": image_identity,
    }

    return OpenedObservation(
        headers=headers,
        file=identity_of_file,
        region=working,
        pixels=values,
        no_data=missing,
        floating_point=floating_point,
        pixel_scale=scale,
        image_identity=image_identity,
        identity=identity,
        record=MappingProxyType(record),
    )


def open_observation(settings: Any, *, project_root: str | Path) -> OpenedObservation:
    """Open the image of a configuration, as the first stage of the preparation.

    Parameters
    ----------
    settings : PreparationSettings
        The configuration as :func:`baymorph.preparation.schema.
        read_preparation_settings` reads it.
    project_root : str or pathlib.Path
        The folder of the project; ``data.image_path`` is relative to it.

    Returns
    -------
    OpenedObservation
        The opened image.

    Raises
    ------
    FileNotFoundError
        If the image or the project folder does not exist.
    ValueError
        As :func:`open_image`.
    """
    return open_image(
        settings.get("data.image_path"),
        project_root=project_root,
        hdu=settings.get("data.hdu"),
        region=settings.get("data.region"),
        target_position=settings.get("data.target_position"),
        pixel_scale=settings.get("data.pixel_scale"),
        calibration=settings.get("data.calibration"),
    )


def resolve_working_region(
    setting: str | dict[str, float] | None,
    *,
    image_shape: tuple[int, int],
    target_position: Any,
    pixel_scale: PixelScale,
) -> tuple[WorkingRegion | None, str]:
    """Return the working region of an image, or None, and why (AD-053).

    Parameters
    ----------
    setting : None, str or dict
        ``data.region`` in the canonical form of :func:`validate_region`.
    image_shape : tuple of int
        Shape of the whole image as ``(ny, nx)``.
    target_position : None, "center" or mapping
        ``data.target_position`` as the schema returns it.
    pixel_scale : PixelScale
        The pixel scale, which converts a side in arcsec.

    Returns
    -------
    tuple
        The region, or None when the whole image is read, and one sentence
        that says why.

    Raises
    ------
    ValueError
        If a side in arcsec meets an image without a pixel scale, or a region
        about the target is needed and the target is not set or lies outside
        the image.
    """
    height, width = image_shape

    if setting == "whole":
        return None, "data.region: whole."

    if setting is None:
        if max(width, height) <= WHOLE_IMAGE_MAX_SIDE:
            return None, (
                f"No side exceeds {WHOLE_IMAGE_MAX_SIDE} pixels, so the image is "
                "read whole."
            )

        side = WHOLE_IMAGE_MAX_SIDE
        reason = (
            f"A side exceeds {WHOLE_IMAGE_MAX_SIDE} pixels, so a square of "
            f"{WHOLE_IMAGE_MAX_SIDE} pixels about the target is read."
        )
    else:
        if pixel_scale.value is None:
            if pixel_scale.note is None:
                raise ValueError(
                    "data.region gives its side in arcsec, which needs the pixel "
                    "scale of the image, and its header holds no celestial WCS: "
                    "declare data.pixel_scale."
                )

            raise ValueError(
                "data.region gives its side in arcsec, which needs the pixel "
                f"scale of the image. {pixel_scale.note}"
            )

        side = _side_in_pixels(setting["size"], pixel_scale.value)
        reason = (
            f"data.region gives a side of {setting['size']:g} arcsec, "
            f"{side} pixels at {pixel_scale.value:.6g} arcsec per pixel "
            f"({pixel_scale.source})."
        )

        if side >= width and side >= height:
            return None, reason + " It holds the whole image, which is read whole."

    center = seed_position(target_position, image_shape)

    if center is None:
        raise ValueError(
            f"The image of {width} x {height} pixels is read within a working "
            f"region of {side} pixels about the target, and no target position "
            "is set: set data.target_position, or declare data.region: whole."
        )

    x, y = center

    if not (-0.5 <= x <= width - 0.5 and -0.5 <= y <= height - 0.5):
        raise ValueError(
            f"data.target_position ({x:g}, {y:g}) lies outside the image of "
            f"{width} x {height} pixels, so no working region can be taken "
            "about it."
        )

    working = WorkingRegion.about(image_shape, center_x=x, center_y=y, side=side)

    return working, reason


def seed_position(
    target_position: Any, image_shape: tuple[int, int]
) -> tuple[float, float] | None:
    """Return the target position in pixels of the whole image, or None.

    The opening takes the working region about this position, and the target
    stage checks it (:mod:`baymorph.preparation.target`).

    Parameters
    ----------
    target_position : None, "center" or mapping
        ``data.target_position`` as the schema returns it: ``"center"`` is the
        centre of the whole image, ``{x: ..., y: ...}`` a position in pixels.
    image_shape : tuple of int
        Shape of the whole image as ``(ny, nx)``.

    Returns
    -------
    tuple of float or None
        The position ``(x, y)``, or None when no position is set. It is not
        checked against the image.

    Raises
    ------
    ValueError
        If ``target_position`` has none of the forms above.
    """
    if target_position is None:
        return None

    height, width = image_shape

    if target_position == "center":
        return ((width - 1) / 2.0, (height - 1) / 2.0)

    if isinstance(target_position, Mapping) and set(target_position) == {"x", "y"}:
        x, y = target_position["x"], target_position["y"]

        if all(
            isinstance(value, Real)
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in (x, y)
        ):
            return (float(x), float(y))

    raise ValueError(
        "data.target_position must be 'center' or {x: ..., y: ...} with two finite "
        f"coordinates, as the configuration reads it; got {target_position!r}."
    )


def _json_scalar(value: Any) -> Any:
    """Return a header value as a JSON scalar."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    return str(value)


__all__ = [
    "MAX_KEPT_FILES",
    "NO_DATA_MIN_ZERO_PIXELS",
    "OPENING_IMAGE_STAGE",
    "OPENING_STAGE",
    "OPENING_VERSION",
    "PIXEL_SCALE_TOLERANCE",
    "SCIENCE_EXTENSION_NAMES",
    "WHOLE_IMAGE_MAX_SIDE",
    "HduHeader",
    "ImageHeaders",
    "OpenedObservation",
    "PixelScale",
    "WorkingRegion",
    "file_compression",
    "forget_kept_headers",
    "no_data_mask",
    "open_image",
    "open_observation",
    "read_image_headers",
    "resolve_pixel_scale",
    "resolve_working_region",
    "seed_position",
    "validate_region",
]
