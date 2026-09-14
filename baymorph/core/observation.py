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

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import FITSFixedWarning, WCS
from astropy.wcs.utils import proj_plane_pixel_scales

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

    @classmethod
    def from_header(
        cls,
        header: fits.Header | None,
        *,
        image_shape: tuple[int, int] | None = None,
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
        width, height = _resolve_image_dimensions(
            header,
            image_shape=image_shape,
        )

        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0

        celestial_wcs = _read_celestial_wcs(header)
        scale_x, scale_y, scale_mean = _read_pixel_scales(celestial_wcs)

        seeing_fwhm = _optional_finite_float(
            header,
            "FWHM",
        )
        seeing_fwhm_unit = (
            _read_seeing_fwhm_unit(
                header
            )
            if seeing_fwhm is not None
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
            gain=_optional_finite_float(header, "GAIN"),
            saturation=_first_finite_float(
                header,
                ("SATLEVEL", "SATURATE"),
            ),
            seeing_fwhm=seeing_fwhm,
            seeing_fwhm_unit=seeing_fwhm_unit,
            zeropoint=_optional_finite_float(header, "PHOTZP"),
            filter_name=_optional_clean_string(header, "FILTER"),
            telescope=_optional_clean_string(header, "TELESCOP"),
            instrument=_optional_clean_string(header, "INSTRUME"),
            bunit=_optional_clean_string(header, "BUNIT"),
            exposure_time=_optional_finite_float(header, "EXPTIME"),
            total_exposure_time=_optional_finite_float(
                header,
                "TEXPTIME",
            ),
            background_header=_optional_finite_float(
                header,
                "BKGMED",
            ),
            wcs=celestial_wcs,
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
        )


def resolve_seeing_fwhm_pixels(
    metadata: ObservationMetadata,
    *,
    pixel_scale_arcsec: float | None,
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

    if unit not in {
        None,
        "arcsec",
        "pixels",
    }:
        raise ValueError(
            "seeing_fwhm_unit must be 'arcsec', 'pixels', or None."
        )

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
        return cls(
            image=image,
            metadata=ObservationMetadata.from_image(image),
        )

    @classmethod
    def from_fits(
        cls,
        filename: str | Path,
    ) -> Observation:
        """Load an astronomical observation from a FITS file.

        The primary HDU is loaded through :meth:`ImageData.from_fits`.
        Observational metadata are then extracted from the copied FITS header
        and validated against the loaded image-array shape.

        Missing optional metadata do not prevent the observation from being
        created. Required image dimensions can always be recovered from the
        loaded array.

        Parameters
        ----------
        filename : str or pathlib.Path
            Path to the FITS file.

        Returns
        -------
        Observation
            Observation containing the image and normalized metadata.
        """
        image = ImageData.from_fits(filename)

        return cls.from_image(image)


def _resolve_image_dimensions(
    header: fits.Header | None,
    *,
    image_shape: tuple[int, int] | None,
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


def _validate_image_shape(
    image_shape: tuple[int, int],
) -> tuple[int, int]:
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


def _optional_positive_int(
    header: fits.Header | None,
    keyword: str,
) -> int | None:
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


def _normalize_seeing_fwhm_unit(
    value: Any,
) -> str | None:
    """Normalize one explicit FITS FWHM-unit declaration."""
    if value is None:
        return None

    normalized = "".join(
        character.lower()
        if character.isalpha()
        else " "
        for character in str(
            value
        )
    )

    tokens = set(
        normalized.split()
    )
    recognized = set()

    if tokens & {
        "arcsec",
        "arcsecond",
        "arcseconds",
        "asec",
    }:
        recognized.add(
            "arcsec"
        )

    if tokens & {
        "pix",
        "pixel",
        "pixels",
    }:
        recognized.add(
            "pixels"
        )

    if len(
        recognized
    ) != 1:
        return None

    return recognized.pop()


def _read_seeing_fwhm_unit(
    header: fits.Header | None,
) -> str | None:
    """Read an explicit FWHM unit without numerical inference."""
    if header is None:
        return None

    declarations = set()

    for keyword in (
        "FWHMUNIT",
        "SEEUNIT",
    ):
        unit = _normalize_seeing_fwhm_unit(
            header.get(
                keyword
            )
        )

        if unit is not None:
            declarations.add(
                unit
            )

    if "FWHM" in header:
        try:
            comment = header.comments[
                "FWHM"
            ]
        except (
            KeyError,
            TypeError,
        ):
            comment = None

        comment_unit = _normalize_seeing_fwhm_unit(
            comment
        )

        if comment_unit is not None:
            declarations.add(
                comment_unit
            )

    if len(
        declarations
    ) != 1:
        return None

    return declarations.pop()


def _optional_finite_float(
    header: fits.Header | None,
    keyword: str,
) -> float | None:
    """Read an optional finite floating-point FITS keyword."""
    if header is None or keyword not in header:
        return None

    value: Any = header[keyword]

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not np.isfinite(result):
        return None

    return result


def _first_finite_float(
    header: fits.Header | None,
    keywords: tuple[str, ...],
) -> float | None:
    """Return the first valid floating-point keyword in priority order."""
    for keyword in keywords:
        value = _optional_finite_float(header, keyword)

        if value is not None:
            return value

    return None


def _optional_clean_string(
    header: fits.Header | None,
    keyword: str,
) -> str | None:
    """Read an optional, non-empty string FITS keyword."""
    if header is None or keyword not in header:
        return None

    value = str(header[keyword]).strip()

    if not value:
        return None

    return value


def _read_celestial_wcs(
    header: fits.Header | None,
) -> WCS | None:
    """Construct a valid two-dimensional celestial WCS when possible."""
    if header is None:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                category=FITSFixedWarning,
            )
            wcs = WCS(header)
    except Exception:
        return None

    if not wcs.has_celestial:
        return None

    try:
        celestial = wcs.celestial
    except Exception:
        return None

    if celestial.pixel_n_dim != 2 or celestial.world_n_dim != 2:
        return None

    return celestial


def _read_pixel_scales(
    wcs: WCS | None,
) -> tuple[float | None, float | None, float | None]:
    """Return projected celestial scales in arcseconds per pixel."""
    if wcs is None:
        return None, None, None

    try:
        scales_deg = np.asarray(
            proj_plane_pixel_scales(wcs),
            dtype=float,
        )
    except Exception:
        return None, None, None

    if scales_deg.shape != (2,):
        return None, None, None

    scales_arcsec = np.abs(scales_deg) * 3600.0

    if not np.all(np.isfinite(scales_arcsec)):
        return None, None, None

    if np.any(scales_arcsec <= 0.0):
        return None, None, None

    scale_x = float(scales_arcsec[0])
    scale_y = float(scales_arcsec[1])
    scale_mean = float(np.mean(scales_arcsec))

    return scale_x, scale_y, scale_mean
