"""Tests for BayMorph observation containers and FITS metadata extraction."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from baymorph.core.image import ImageData
from baymorph.core.observation import (
    Observation,
    ObservationMetadata,
    resolve_seeing_fwhm_pixels,
)


def make_complete_header() -> fits.Header:
    """Return a representative VST/OmegaCAM-like FITS header."""
    header = fits.Header()

    header["NAXIS"] = 2
    header["NAXIS1"] = 1451
    header["NAXIS2"] = 1451

    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CRVAL1"] = 201.015
    header["CRVAL2"] = -31.60394444
    header["CRPIX1"] = 725.0
    header["CRPIX2"] = 725.0
    header["CD1_1"] = -5.833333151e-05
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 5.833333151e-05

    header["GAIN"] = 9.155573901
    header["SATLEVEL"] = 54652.6192
    header["SATURATE"] = 50000.0
    header["FWHM"] = 0.78812181
    header["PHOTZP"] = 30.0

    header["FILTER"] = "i_SDSS"
    header["TELESCOP"] = "ESO-VST"
    header["INSTRUME"] = "OMEGACAM"
    header["BUNIT"] = "ADU/s"

    header["EXPTIME"] = 1.0
    header["TEXPTIME"] = 1000.0
    header["BKGMED"] = 533.62216185

    return header


def make_image(
    *,
    shape: tuple[int, int] = (1451, 1451),
    header: fits.Header | None = None,
) -> ImageData:
    """Return an ImageData instance for observation tests."""
    data = np.zeros(shape, dtype=float)

    return ImageData(
        data=data,
        header=header,
    )


def test_metadata_reads_complete_header() -> None:
    """All supported metadata should be read from a complete header."""
    metadata = ObservationMetadata.from_header(make_complete_header())

    assert metadata.width == 1451
    assert metadata.height == 1451

    assert metadata.center_x == pytest.approx(725.0)
    assert metadata.center_y == pytest.approx(725.0)

    assert metadata.pixel_scale_x == pytest.approx(0.21)
    assert metadata.pixel_scale_y == pytest.approx(0.21)
    assert metadata.pixel_scale == pytest.approx(0.21)

    assert metadata.gain == pytest.approx(9.155573901)
    assert metadata.saturation == pytest.approx(54652.6192)
    assert metadata.seeing_fwhm == pytest.approx(0.78812181)
    assert metadata.zeropoint == pytest.approx(30.0)

    assert metadata.filter_name == "i_SDSS"
    assert metadata.telescope == "ESO-VST"
    assert metadata.instrument == "OMEGACAM"
    assert metadata.bunit == "ADU/s"

    assert metadata.exposure_time == pytest.approx(1.0)
    assert metadata.total_exposure_time == pytest.approx(1000.0)
    assert metadata.background_header == pytest.approx(533.62216185)

    assert metadata.wcs is not None


def test_metadata_uses_image_shape_without_header() -> None:
    """Array dimensions should be used when no FITS header is available."""
    metadata = ObservationMetadata.from_header(
        None,
        image_shape=(81, 101),
    )

    assert metadata.width == 101
    assert metadata.height == 81

    assert metadata.center_x == pytest.approx(50.0)
    assert metadata.center_y == pytest.approx(40.0)

    assert metadata.pixel_scale is None
    assert metadata.pixel_scale_x is None
    assert metadata.pixel_scale_y is None

    assert metadata.gain is None
    assert metadata.saturation is None
    assert metadata.seeing_fwhm is None
    assert metadata.zeropoint is None

    assert metadata.filter_name is None
    assert metadata.telescope is None
    assert metadata.instrument is None
    assert metadata.bunit is None

    assert metadata.exposure_time is None
    assert metadata.total_exposure_time is None
    assert metadata.background_header is None

    assert metadata.wcs is None


def test_image_shape_fills_missing_header_dimensions() -> None:
    """Missing dimension keywords should fall back to the image shape."""
    header = fits.Header()

    metadata = ObservationMetadata.from_header(
        header,
        image_shape=(60, 80),
    )

    assert metadata.width == 80
    assert metadata.height == 60
    assert metadata.center_x == pytest.approx(39.5)
    assert metadata.center_y == pytest.approx(29.5)


def test_header_and_image_shape_must_be_consistent() -> None:
    """Conflicting header and array dimensions should raise an error."""
    header = fits.Header()
    header["NAXIS1"] = 100
    header["NAXIS2"] = 80

    with pytest.raises(
        ValueError,
        match="NAXIS1 is inconsistent",
    ):
        ObservationMetadata.from_header(
            header,
            image_shape=(80, 101),
        )


def test_dimensions_are_required_without_image_shape() -> None:
    """Dimensions must exist in either the header or image shape."""
    with pytest.raises(
        ValueError,
        match="Image dimensions could not be determined",
    ):
        ObservationMetadata.from_header(fits.Header())


def test_satlevel_has_priority_over_saturate() -> None:
    """SATLEVEL should take precedence over SATURATE."""
    header = make_complete_header()

    metadata = ObservationMetadata.from_header(header)

    assert metadata.saturation == pytest.approx(header["SATLEVEL"])


def test_saturate_is_fallback_for_missing_satlevel() -> None:
    """SATURATE should be used when SATLEVEL is unavailable."""
    header = make_complete_header()
    del header["SATLEVEL"]

    metadata = ObservationMetadata.from_header(header)

    assert metadata.saturation == pytest.approx(header["SATURATE"])


@pytest.mark.parametrize(
    ("keyword", "attribute"),
    [
        ("GAIN", "gain"),
        ("FWHM", "seeing_fwhm"),
        ("PHOTZP", "zeropoint"),
        ("EXPTIME", "exposure_time"),
        ("TEXPTIME", "total_exposure_time"),
        ("BKGMED", "background_header"),
    ],
)
def test_missing_optional_numeric_keyword_returns_none(
    keyword: str,
    attribute: str,
) -> None:
    """Missing optional numeric metadata should be represented by None."""
    header = make_complete_header()
    del header[keyword]

    metadata = ObservationMetadata.from_header(header)

    assert getattr(metadata, attribute) is None


@pytest.mark.parametrize(
    ("keyword", "attribute"),
    [
        ("FILTER", "filter_name"),
        ("TELESCOP", "telescope"),
        ("INSTRUME", "instrument"),
        ("BUNIT", "bunit"),
    ],
)
def test_missing_optional_string_keyword_returns_none(
    keyword: str,
    attribute: str,
) -> None:
    """Missing optional string metadata should be represented by None."""
    header = make_complete_header()
    del header[keyword]

    metadata = ObservationMetadata.from_header(header)

    assert getattr(metadata, attribute) is None


def test_malformed_optional_values_return_none() -> None:
    """Malformed optional metadata must not prevent image ingestion."""
    header = make_complete_header()

    header["GAIN"] = "not-a-number"
    header["FWHM"] = "invalid"
    header["PHOTZP"] = "invalid"
    header["FILTER"] = ""

    metadata = ObservationMetadata.from_header(header)

    assert metadata.gain is None
    assert metadata.seeing_fwhm is None
    assert metadata.zeropoint is None
    assert metadata.filter_name is None


def test_metadata_from_image_uses_header_and_array_shape() -> None:
    """Metadata should be constructible directly from ImageData."""
    image = make_image(header=make_complete_header())

    metadata = ObservationMetadata.from_image(image)

    assert metadata.width == image.shape[1]
    assert metadata.height == image.shape[0]
    assert metadata.pixel_scale == pytest.approx(0.21)
    assert metadata.filter_name == "i_SDSS"


def test_observation_from_image_builds_metadata() -> None:
    """Observation.from_image should retain the image and infer metadata."""
    image = make_image(header=make_complete_header())

    observation = Observation.from_image(image)

    assert observation.image is image
    assert observation.metadata.width == image.shape[1]
    assert observation.metadata.height == image.shape[0]
    assert observation.metadata.filter_name == "i_SDSS"


def test_observation_rejects_inconsistent_metadata() -> None:
    """Observation must reject metadata with incompatible dimensions."""
    image = make_image(
        shape=(80, 100),
        header=None,
    )

    metadata = ObservationMetadata(
        width=101,
        height=80,
        center_x=50.0,
        center_y=39.5,
    )

    with pytest.raises(
        ValueError,
        match="metadata dimensions do not match",
    ):
        Observation(
            image=image,
            metadata=metadata,
        )


def test_observation_from_fits_loads_image_and_metadata(
    tmp_path,
) -> None:
    """Observation.from_fits should load data, header, and metadata."""
    filename = tmp_path / "observation.fits"

    data = np.arange(
        60,
        dtype=np.float32,
    ).reshape(6, 10)

    header = fits.Header()
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CRVAL1"] = 201.015
    header["CRVAL2"] = -31.60394444
    header["CRPIX1"] = 5.0
    header["CRPIX2"] = 3.0
    header["CD1_1"] = -5.833333151e-05
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 5.833333151e-05
    header["FILTER"] = "i_SDSS"
    header["GAIN"] = 9.155573901

    fits.PrimaryHDU(
        data=data,
        header=header,
    ).writeto(filename)

    observation = Observation.from_fits(filename)

    assert observation.image.shape == (6, 10)
    assert np.array_equal(observation.image.data, data)

    assert observation.metadata.width == 10
    assert observation.metadata.height == 6
    assert observation.metadata.center_x == pytest.approx(4.5)
    assert observation.metadata.center_y == pytest.approx(2.5)

    assert observation.metadata.pixel_scale == pytest.approx(0.21)
    assert observation.metadata.filter_name == "i_SDSS"
    assert observation.metadata.gain == pytest.approx(9.155573901)


def test_observation_from_fits_accepts_string_path(
    tmp_path,
) -> None:
    """Observation.from_fits should accept ordinary string paths."""
    filename = tmp_path / "string_path.fits"

    data = np.ones(
        (5, 7),
        dtype=np.float32,
    )

    fits.PrimaryHDU(data=data).writeto(filename)

    observation = Observation.from_fits(str(filename))

    assert observation.image.shape == (5, 7)
    assert observation.metadata.width == 7
    assert observation.metadata.height == 5


def test_observation_from_fits_missing_file_raises(
    tmp_path,
) -> None:
    """Missing FITS files should preserve the loader's file error."""
    filename = tmp_path / "missing.fits"

    with pytest.raises(
        FileNotFoundError,
    ):
        Observation.from_fits(filename)


def test_fwhm_unit_requires_explicit_metadata() -> None:
    """A bare numerical FWHM must not acquire an inferred unit."""
    header = make_complete_header()
    header[
        "FWHM"
    ] = (
        0.78812181,
        "FWHM median",
    )

    metadata = ObservationMetadata.from_header(
        header
    )

    assert metadata.seeing_fwhm == pytest.approx(
        0.78812181
    )
    assert metadata.seeing_fwhm_unit is None


def test_fwhm_unit_reads_explicit_header_keyword() -> None:
    """Recognized explicit FITS unit keywords should be normalized."""
    for declared, expected in (
        (
            "arcseconds",
            "arcsec",
        ),
        (
            "pix",
            "pixels",
        ),
    ):
        header = make_complete_header()
        header[
            "FWHMUNIT"
        ] = declared

        metadata = ObservationMetadata.from_header(
            header
        )

        assert metadata.seeing_fwhm_unit == expected


def test_fwhm_unit_reads_explicit_card_comment() -> None:
    """An explicit unit in the FWHM card comment should be retained."""
    header = make_complete_header()
    header[
        "FWHM"
    ] = (
        3.75,
        "Median FWHM [pixels]",
    )

    metadata = ObservationMetadata.from_header(
        header
    )

    assert metadata.seeing_fwhm == pytest.approx(
        3.75
    )
    assert metadata.seeing_fwhm_unit == "pixels"


def test_conflicting_fwhm_unit_declarations_are_unresolved() -> None:
    """Conflicting FITS declarations must never trigger conversion."""
    header = make_complete_header()
    header[
        "FWHM"
    ] = (
        0.8,
        "Median FWHM [pixels]",
    )
    header[
        "FWHMUNIT"
    ] = "arcsec"

    metadata = ObservationMetadata.from_header(
        header
    )

    assert metadata.seeing_fwhm == pytest.approx(
        0.8
    )
    assert metadata.seeing_fwhm_unit is None

@pytest.mark.parametrize(
    (
        "seeing_fwhm",
        "unit",
        "pixel_scale_arcsec",
        "expected",
    ),
    [
        (0.8, "arcsec", 0.2, 4.0),
        (4.0, "pixels", None, 4.0),
    ],
)
def test_resolve_seeing_fwhm_pixels_uses_explicit_units(
    seeing_fwhm: float,
    unit: str,
    pixel_scale_arcsec: float | None,
    expected: float,
) -> None:
    """Explicit FWHM units should determine the pixel conversion."""
    metadata = ObservationMetadata(
        width=1,
        height=1,
        center_x=0.0,
        center_y=0.0,
        seeing_fwhm=seeing_fwhm,
        seeing_fwhm_unit=unit,
    )

    resolved = resolve_seeing_fwhm_pixels(
        metadata,
        pixel_scale_arcsec=pixel_scale_arcsec,
    )

    assert resolved == pytest.approx(expected)


def test_resolve_seeing_fwhm_pixels_rejects_implicit_conversion() -> None:
    """An unresolved FWHM unit must not trigger numerical conversion."""
    metadata = ObservationMetadata(
        width=1,
        height=1,
        center_x=0.0,
        center_y=0.0,
        seeing_fwhm=0.8,
        seeing_fwhm_unit=None,
    )

    assert resolve_seeing_fwhm_pixels(
        metadata,
        pixel_scale_arcsec=0.2,
    ) is None


@pytest.mark.parametrize(
    "pixel_scale_arcsec",
    [None, 0.0, -0.2, np.nan],
)
def test_resolve_arcsec_fwhm_requires_valid_pixel_scale(
    pixel_scale_arcsec: float | None,
) -> None:
    """Arcsecond FWHM requires a finite positive pixel scale."""
    metadata = ObservationMetadata(
        width=1,
        height=1,
        center_x=0.0,
        center_y=0.0,
        seeing_fwhm=0.8,
        seeing_fwhm_unit="arcsec",
    )

    assert resolve_seeing_fwhm_pixels(
        metadata,
        pixel_scale_arcsec=pixel_scale_arcsec,
    ) is None


def test_resolve_seeing_fwhm_pixels_rejects_unknown_unit() -> None:
    """Unsupported explicit FWHM units must fail clearly."""
    metadata = ObservationMetadata(
        width=1,
        height=1,
        center_x=0.0,
        center_y=0.0,
        seeing_fwhm=800.0,
        seeing_fwhm_unit="mas",
    )

    with pytest.raises(
        ValueError,
        match="seeing_fwhm_unit",
    ):
        resolve_seeing_fwhm_pixels(
            metadata,
            pixel_scale_arcsec=0.2,
        )
