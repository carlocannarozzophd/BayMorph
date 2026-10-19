"""Observation containers and robust FITS metadata extraction.

This module defines the core abstractions used to represent an astronomical
observation and the optional observational metadata stored in a FITS header.

FITS metadata are treated as optional. Missing, malformed, or unsupported
keywords are represented by ``None`` and must not prevent BayMorph from
loading or analysing an image.

At this development stage, these classes do not alter image preparation,
PSF construction, fitting, diagnostics, or configuration resolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from baymorph.core.header_metadata import (
    HeaderReading,
    read_header,
    read_pixel_scales,
)
from baymorph.core.image import ImageData


@dataclass(frozen=True)
class ObservationMetadata:
    """Normalized observational metadata associated with an image.

    Parameters
    ----------
    width, height : int
        Image dimensions in pixels.
    center_x, center_y : float
        Geometric centre of the image in zero-based pixel coordinates.
    pixel_scale : float, optional
        Mean projected celestial pixel scale in arcseconds per pixel.
    pixel_scale_x, pixel_scale_y : float, optional
        Projected celestial pixel scales along the two image axes, in
        arcseconds per pixel.
    gain : float, optional
        Detector or effective gain, normally in electrons per ADU.
    saturation : float, optional
        Saturation level in the image pixel-value units.
    seeing_fwhm : float, optional
        Raw seeing or PSF FWHM value reported in the FITS header.
    seeing_fwhm_unit : {"arcsec", "pixels"}, optional
        Explicitly declared unit of ``seeing_fwhm``. No unit is inferred
        from the numerical value alone.
    zeropoint : float, optional
        Photometric zeropoint.
    filter_name : str, optional
        Filter identifier.
    telescope : str, optional
        Telescope identifier.
    instrument : str, optional
        Instrument identifier.
    bunit : str, optional
        Unit of the image pixel values.
    exposure_time : float, optional
        Exposure time reported by ``EXPTIME``.
    total_exposure_time : float, optional
        Total combined exposure time reported by ``TEXPTIME``.
    background_header : float, optional
        Background estimate recorded in the FITS header.
    wcs : astropy.wcs.WCS, optional
        Valid two-dimensional celestial WCS.
    header_reading : HeaderReading, optional
        Every quantity read from the header through the conventions of
        AD-049, with its provenance and status. The fields above keep the
        header values that BayMorph has always read, whatever their status;
        only the reading says whether a value may be used without a
        declaration of the user.
    """

    width: int
    height: int

    center_x: float
    center_y: float

    pixel_scale: float | None = None
    pixel_scale_x: float | None = None
    pixel_scale_y: float | None = None

    gain: float | None = None
    saturation: float | None = None
    seeing_fwhm: float | None = None
    seeing_fwhm_unit: str | None = None
    zeropoint: float | None = None

    filter_name: str | None = None
    telescope: str | None = None
    instrument: str | None = None
    bunit: str | None = None

    exposure_time: float | None = None
    total_exposure_time: float | None = None
    background_header: float | None = None

    wcs: WCS | None = None
    header_reading: HeaderReading | None = None

    @classmethod
    def from_header(
        cls,
        header: fits.Header | None,
        *,
        image_shape: tuple[int, int] | None = None,
        primary_header: fits.Header | None = None,
        reading: HeaderReading | None = None,
    ) -> ObservationMetadata:
        """Construct normalized metadata from a FITS header.

        Image dimensions are read from ``NAXIS1`` and ``NAXIS2`` when valid.
        If either keyword is missing or malformed, ``image_shape`` is used as
        a fallback.

        All other metadata are optional. Missing, malformed, non-finite, or
        unsupported values are represented by ``None``.

        Parameters
        ----------
        header : astropy.io.fits.Header or None
            FITS header to inspect.
        image_shape : tuple of int, optional
            Image shape as ``(ny, nx)``. This provides a reliable fallback for
            dimensions when an image has no FITS header or when its dimension
            keywords are invalid.
        primary_header : astropy.io.fits.Header, optional
            Primary header of a file whose image is stored in an extension.
            Its keywords are inherited where the image header lacks them.
        reading : HeaderReading, optional
            Reading of ``header`` and ``primary_header`` already made by
            :func:`~baymorph.core.header_metadata.read_header`, with the
            celestial WCS it derived. The headers are read when it is not
            given.

        Returns
        -------
        ObservationMetadata
            Normalized observational metadata.

        Raises
        ------
        ValueError
            If valid image dimensions cannot be obtained from either the
            header or ``image_shape``.
        """
        width, height = _resolve_image_dimensions(header, image_shape=image_shape)

        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0

        if reading is None:
            reading = read_header(header, primary_header=primary_header)

        celestial_wcs = reading.celestial_wcs
        scale_x, scale_y, scale_mean = read_pixel_scales(celestial_wcs)

        psf_fwhm = reading["psf_fwhm"]
        seeing_fwhm_unit = (
            psf_fwhm.unit
            if psf_fwhm.usable and psf_fwhm.unit in {"arcsec", "pixels"}
            else None
        )

        return cls(
            width=width,
            height=height,
            center_x=center_x,
            center_y=center_y,
            pixel_scale=scale_mean,
            pixel_scale_x=scale_x,
            pixel_scale_y=scale_y,
            gain=_header_number(reading, "gain"),
            saturation=_header_number(reading, "saturation"),
            seeing_fwhm=_header_number(reading, "psf_fwhm"),
            seeing_fwhm_unit=seeing_fwhm_unit,
            zeropoint=_header_number(reading, "zeropoint"),
            filter_name=_header_text(reading, "filter"),
            telescope=_header_text(reading, "telescope"),
            instrument=_header_text(reading, "instrument"),
            bunit=_header_text(reading, "image_unit"),
            exposure_time=_header_number(reading, "exposure_time"),
            total_exposure_time=_header_number(reading, "total_exposure_time"),
            background_header=_header_number(reading, "header_background"),
            wcs=celestial_wcs,
            header_reading=reading,
        )

    @classmethod
    def from_image(cls, image: ImageData) -> ObservationMetadata:
        """Construct metadata from an :class:`~baymorph.core.image.ImageData`.

        The image-array shape is always supplied as a fallback, so this method
        also works for images without FITS headers.

        Parameters
        ----------
        image : ImageData
            BayMorph image container.

        Returns
        -------
        ObservationMetadata
            Metadata associated with the image.
        """
        return cls.from_header(
            image.header,
            image_shape=image.shape,
            primary_header=image.primary_header,
            reading=image.header_reading,
        )


def resolve_seeing_fwhm_pixels(
    metadata: ObservationMetadata, *, pixel_scale_arcsec: float | None
) -> float | None:
    """Resolve an explicitly unit-tagged seeing FWHM in pixels.

    Parameters
    ----------
    metadata : ObservationMetadata
        Metadata containing the raw FWHM value and its explicit unit.
    pixel_scale_arcsec : float, optional
        Pixel scale in arcseconds per pixel. This is required only when the
        FWHM unit is ``"arcsec"``.

    Returns
    -------
    float or None
        FWHM in pixels, or ``None`` when the FWHM unit is unresolved or an
        arcsecond value cannot be converted with a valid pixel scale.

    Raises
    ------
    ValueError
        If the FWHM unit is unsupported or an explicitly unit-tagged FWHM is
        not finite and positive.
    """
    unit = metadata.seeing_fwhm_unit

    if unit not in {None, "arcsec", "pixels"}:
        raise ValueError("seeing_fwhm_unit must be 'arcsec', 'pixels', or None.")

    if metadata.seeing_fwhm is None or unit is None:
        return None

    value = float(metadata.seeing_fwhm)

    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("seeing_fwhm must be finite and positive.")

    if unit == "pixels":
        return value

    if (
        pixel_scale_arcsec is None
        or not np.isfinite(pixel_scale_arcsec)
        or pixel_scale_arcsec <= 0.0
    ):
        return None

    return value / float(pixel_scale_arcsec)


@dataclass(frozen=True)
class Observation:
    """Astronomical observation used by BayMorph.

    Parameters
    ----------
    image : ImageData
        Astronomical image and its original FITS information.
    metadata : ObservationMetadata
        Normalized observational metadata.

    Notes
    -----
    This initial abstraction intentionally contains only the image and its
    metadata. Future additions such as variance maps, data-quality masks, PSF
    products, and external catalogues should be introduced only when their
    interfaces and scientific semantics have been defined.
    """

    image: ImageData
    metadata: ObservationMetadata

    def __post_init__(self) -> None:
        """Validate consistency between image and metadata dimensions."""
        ny, nx = self.image.shape

        if self.metadata.width != nx or self.metadata.height != ny:
            raise ValueError(
                "Observation metadata dimensions do not match the image: "
                f"metadata=(height={self.metadata.height}, "
                f"width={self.metadata.width}), "
                f"image_shape={self.image.shape}."
            )

    @classmethod
    def from_image(cls, image: ImageData) -> Observation:
        """Construct an observation from an existing image container.

        Parameters
        ----------
        image : ImageData
            BayMorph image container.

        Returns
        -------
        Observation
            Observation containing the image and metadata inferred from its
            FITS header and array shape.
        """
        return cls(image=image, metadata=ObservationMetadata.from_image(image))

    @classmethod
    def from_fits(
        cls, filename: str | Path, hdu: int | str | None = None
    ) -> Observation:
        """Load an astronomical observation from a FITS file.

        The image HDU is selected and loaded through
        :meth:`ImageData.from_fits`.
        Observational metadata are then extracted from the copied FITS header
        and validated against the loaded image-array shape.

        Missing optional metadata do not prevent the observation from being
        created. Required image dimensions can always be recovered from the
        loaded array.

        Parameters
        ----------
        filename : str or pathlib.Path
            Path to the FITS file.
        hdu : int or str, optional
            Index or ``EXTNAME`` of the HDU that holds the image, the
            ``data.hdu`` of a configuration. See :meth:`ImageData.from_fits`
            for the default.

        Returns
        -------
        Observation
            Observation containing the image and normalized metadata.
        """
        image = ImageData.from_fits(filename, hdu=hdu)

        return cls.from_image(image)


def _resolve_image_dimensions(
    header: fits.Header | None, *, image_shape: tuple[int, int] | None
) -> tuple[int, int]:
    """Resolve image width and height from a header or array shape."""
    width = _optional_positive_int(header, "NAXIS1")
    height = _optional_positive_int(header, "NAXIS2")

    fallback_width: int | None = None
    fallback_height: int | None = None

    if image_shape is not None:
        fallback_height, fallback_width = _validate_image_shape(image_shape)

    if width is None:
        width = fallback_width

    if height is None:
        height = fallback_height

    if width is None or height is None:
        raise ValueError(
            "Image dimensions could not be determined. Provide valid "
            "NAXIS1 and NAXIS2 FITS keywords or a valid image_shape."
        )

    if fallback_width is not None and width != fallback_width:
        raise ValueError(
            "FITS NAXIS1 is inconsistent with the image-array width: "
            f"NAXIS1={width}, array width={fallback_width}."
        )

    if fallback_height is not None and height != fallback_height:
        raise ValueError(
            "FITS NAXIS2 is inconsistent with the image-array height: "
            f"NAXIS2={height}, array height={fallback_height}."
        )

    return width, height


def _validate_image_shape(image_shape: tuple[int, int]) -> tuple[int, int]:
    """Validate and normalize an image shape as ``(height, width)``."""
    if len(image_shape) != 2:
        raise ValueError(
            "image_shape must contain exactly two dimensions as (ny, nx); "
            f"got {image_shape!r}."
        )

    try:
        height = int(image_shape[0])
        width = int(image_shape[1])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("image_shape values must be valid integers.") from exc

    if height <= 0 or width <= 0:
        raise ValueError(
            f"image_shape dimensions must be positive; got {image_shape!r}."
        )

    return height, width


def _optional_positive_int(header: fits.Header | None, keyword: str) -> int | None:
    """Read an optional positive integer FITS keyword."""
    if header is None or keyword not in header:
        return None

    try:
        value = int(header[keyword])
    except (TypeError, ValueError, OverflowError):
        return None

    if value <= 0:
        return None

    return value


def _header_number(reading: HeaderReading, name: str) -> float | None:
    """Return the finite number that the header holds for a quantity."""
    value = reading[name].raw_value

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None

    value = float(value)

    return value if math.isfinite(value) else None


def _header_text(reading: HeaderReading, name: str) -> str | None:
    """Return the non-empty string that the header holds for a quantity."""
    value = reading[name].raw_value

    if not isinstance(value, str) or not value.strip():
        return None

    return value.strip()
