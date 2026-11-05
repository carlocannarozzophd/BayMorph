"""The opening stage of the preparation (AD-053).

The opening chooses the HDU of the image, reads its headers and their
calibration, resolves the working region, reads its pixels and the pixels
without data, resolves the pixel scale and gives the identities of the
product. These tests check each step against images whose truth they build:
the pixel ``(x, y)`` of a truth image holds ``1000 y + x``.
"""

from __future__ import annotations

import bz2
import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import baymorph.preparation.opening as opening
from baymorph.preparation.header import read_header
from baymorph.preparation.opening import (
    NO_DATA_MIN_ZERO_PIXELS,
    OPENING_STAGE,
    WHOLE_IMAGE_MAX_SIDE,
    PixelScale,
    WorkingRegion,
    file_compression,
    no_data_mask,
    open_image,
    open_observation,
    read_image_headers,
    resolve_pixel_scale,
    resolve_working_region,
    validate_region,
)
from baymorph.preparation.schema import read_preparation_settings
from baymorph.preparation.store import ProductStore


@pytest.fixture(autouse=True)
def _fresh_headers():
    """Start and end every test without headers kept for reuse."""
    opening.forget_kept_headers()
    yield
    opening.forget_kept_headers()


def _truth(ny: int, nx: int, dtype=np.float32) -> np.ndarray:
    """Return an image whose pixel (x, y) holds exactly 1000 y + x."""
    y, x = np.indices((ny, nx))

    return (1000 * y + x).astype(dtype)


def _wcs_cards(nx: int, ny: int, scale: float = 0.2) -> dict:
    """Return the cards of a TAN celestial WCS of ``scale`` arcsec per pixel."""
    return {
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRVAL1": 195.15,
        "CRVAL2": 27.94,
        "CRPIX1": (nx + 1) / 2.0,
        "CRPIX2": (ny + 1) / 2.0,
        "CD1_1": -scale / 3600.0,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": scale / 3600.0,
    }


def _header(cards: dict | None = None) -> fits.Header:
    header = fits.Header()

    for key, value in (cards or {}).items():
        header[key] = value

    return header


def _compress(plain: Path, kind: str) -> Path:
    """Return a copy of a FITS file compressed as a whole."""
    target = plain.with_name(plain.name + (".gz" if kind == "gzip" else ".bz2"))
    opener = gzip.open if kind == "gzip" else bz2.open

    with plain.open("rb") as source, opener(target, "wb") as sink:
        shutil.copyfileobj(source, sink)

    return target


def _write(path: Path, hdus: list, compression: str | None = None) -> Path:
    """Write HDUs to ``path``, compressed as a whole when asked."""
    fits.HDUList(hdus).writeto(path, overwrite=True)

    return path if compression is None else _compress(path, compression)


def _primary(tmp_path: Path, data, cards: dict | None = None, name: str = "a.fits"):
    return _write(tmp_path / name, [fits.PrimaryHDU(data, header=_header(cards))])


def _science_file(tmp_path: Path, compression: str | None = None) -> Path:
    """Return a file with an empty primary and SCI, WHT and CTX extensions."""
    primary = fits.PrimaryHDU(header=_header({"EXPTIME": 1400.0, "TELESCOP": "X"}))

    return _write(
        tmp_path / "science.fits",
        [
            primary,
            fits.ImageHDU(_truth(20, 30), name="SCI"),
            fits.ImageHDU(np.ones((20, 30), np.float32), name="WHT"),
            fits.ImageHDU(np.ones((20, 30), np.int32), name="CTX"),
        ],
        compression,
    )


# --- The HDU of the image ---------------------------------------------------


@pytest.mark.parametrize("compression", [None, "gzip", "bzip2"])
def test_a_primary_image_is_the_image(tmp_path, compression) -> None:
    path = _write(tmp_path / "a.fits", [fits.PrimaryHDU(_truth(20, 30))], compression)

    headers = read_image_headers(path)

    assert headers.hdu.index == 0
    assert headers.hdu.extension_name is None
    assert headers.primary_header is None
    assert headers.image_shape == (20, 30)
    assert headers.compression == compression


@pytest.mark.parametrize("compression", [None, "gzip", "bzip2"])
@pytest.mark.parametrize(
    ("requested", "index"),
    [(None, 1), ("sci", 1), (" SCI ", 1), ("WHT", 2), (1, 1), (3, 3)],
)
def test_the_hdu_of_the_image_is_chosen_as_before(
    tmp_path, compression, requested, index
) -> None:
    headers = read_image_headers(_science_file(tmp_path, compression), requested)

    assert headers.hdu.index == index
    assert headers.reading.value("exposure_time") == 1400.0
    assert headers.reading["exposure_time"].source == "primary"


@pytest.mark.parametrize("compression", [None, "gzip"])
@pytest.mark.parametrize(
    ("requested", "message"),
    [
        (0, "HDU 0 of the FITS file holds no two-dimensional image."),
        (9, "HDU 9 does not exist: the FITS file has 4 HDUs."),
        ("ERR", "The FITS file has no HDU named 'ERR'."),
        ("1", "The FITS file has no HDU named '1'."),
    ],
)
def test_a_requested_hdu_that_holds_no_image_is_refused(
    tmp_path, compression, requested, message
) -> None:
    path = _science_file(tmp_path, compression)

    with pytest.raises(ValueError) as error:
        read_image_headers(path, requested)

    assert message in str(error.value)


def test_the_science_extension_is_preferred(tmp_path) -> None:
    empty = fits.PrimaryHDU()
    image = _truth(10, 10)
    path = _write(
        tmp_path / "a.fits",
        [
            empty,
            fits.ImageHDU(image, name="SCI_BKSUB"),
            fits.ImageHDU(image, name="SCI"),
            fits.ImageHDU(image, name="ERR"),
        ],
    )

    assert read_image_headers(path).hdu.index == 2

    path = _write(
        tmp_path / "b.fits",
        [empty, fits.ImageHDU(image, name="IMAGE"), fits.ImageHDU(image, name="SCI")],
    )

    assert read_image_headers(path).hdu.index == 2


def test_several_images_without_a_rule_must_be_declared(tmp_path) -> None:
    image = _truth(10, 10)
    unnamed = _write(
        tmp_path / "a.fits",
        [fits.PrimaryHDU(), fits.ImageHDU(image), fits.ImageHDU(image)],
    )

    with pytest.raises(ValueError, match=r"HDUs 1 \(unnamed\), 2 \(unnamed\)"):
        read_image_headers(unnamed)

    twice = _write(
        tmp_path / "b.fits",
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(image, name="SCI"),
            fits.ImageHDU(image, name="SCI"),
        ],
    )

    with pytest.raises(ValueError, match="several two-dimensional images"):
        read_image_headers(twice)

    with pytest.raises(ValueError, match="several HDUs named 'SCI'"):
        read_image_headers(twice, "SCI")

    assert read_image_headers(twice, 2).hdu.index == 2


def test_a_table_is_never_the_image(tmp_path) -> None:
    table = fits.BinTableHDU.from_columns(
        [fits.Column(name="A", format="E", array=np.arange(3.0))], name="SCI"
    )
    path = _write(
        tmp_path / "a.fits",
        [fits.PrimaryHDU(), table, fits.ImageHDU(_truth(10, 10), name="DATA")],
    )

    assert read_image_headers(path).hdu.index == 2

    with pytest.raises(ValueError, match="HDU 'SCI' of the FITS file holds no"):
        read_image_headers(path, "SCI")


def test_a_tile_compressed_image_is_an_image(tmp_path) -> None:
    truth = _truth(40, 50, np.int32)
    path = _write(tmp_path / "a.fits.fz", [fits.PrimaryHDU(), fits.CompImageHDU(truth)])

    opened = open_image(path, project_root=tmp_path)

    assert opened.headers.hdu.index == 1
    assert opened.headers.hdu.kind == "compressed_image"
    assert np.array_equal(opened.pixels, truth)
    assert opened.floating_point is False


def test_a_cube_and_a_file_without_images_are_refused(tmp_path) -> None:
    cube = _write(tmp_path / "a.fits", [fits.PrimaryHDU(np.zeros((1, 5, 5)))])

    with pytest.raises(ValueError, match="holds no two-dimensional image"):
        read_image_headers(cube)

    table = fits.BinTableHDU.from_columns(
        [fits.Column(name="A", format="E", array=np.arange(3.0))]
    )
    empty = _write(tmp_path / "b.fits", [fits.PrimaryHDU(), table])

    with pytest.raises(ValueError, match="holds no two-dimensional image"):
        read_image_headers(empty)


def test_pixels_that_are_not_linear_are_refused_at_the_opening(tmp_path) -> None:
    cards = {"BSOFTEN": 87.3959, "BOFFSET": 2.65496}
    path = _primary(tmp_path, _truth(10, 10), cards)

    with pytest.raises(ValueError, match="asinh encoding"):
        read_image_headers(path)

    linear = _primary(
        tmp_path,
        _truth(10, 10),
        {
            **cards,
            "HISTORY": "Pixel values converted from asinh-scaled flux to linear "
            "flux using the BSOFTEN/BOFFSET keywords.",
        },
        name="b.fits",
    )

    assert read_image_headers(linear).reading.value("pixel_encoding") == "linear"


def test_the_hdu_choice_reads_no_header_beyond_the_one_it_needs(tmp_path) -> None:
    """A gzip file cut after the SCI extension still gives its headers."""
    path = _science_file(tmp_path, None)
    plain = path.read_bytes()

    with fits.open(path) as hdul:
        start_of_wht = hdul.fileinfo(2)["hdrLoc"]

    # The copy keeps the header of WHT and none of its data.
    truncated = tmp_path / "cut.fits.gz"

    with gzip.open(truncated, "wb") as sink:
        sink.write(plain[: start_of_wht + 2880])

    assert read_image_headers(truncated, 1).hdu.index == 1

    with pytest.raises(ValueError, match="cannot read the headers"):
        read_image_headers(truncated)


@pytest.mark.parametrize("compression", ["gzip", "bzip2"])
def test_the_streamed_headers_equal_those_of_astropy(tmp_path, compression) -> None:
    path = _science_file(tmp_path, compression)
    scan = opening._scan(path)

    with fits.open(path) as hdul:
        expected = [hdu.header for hdu in hdul]
        names = [hdu.name for hdu in hdul]

    streamed = scan.every()

    assert [hdu.name for hdu in streamed] == names
    assert all(
        hdu.header == header for hdu, header in zip(streamed, expected, strict=True)
    )
    assert [hdu.kind for hdu in streamed] == ["primary", "image", "image", "image"]


def test_bytes_after_the_last_hdu_end_the_file(tmp_path) -> None:
    path = _primary(tmp_path, _truth(10, 10))
    padded = tmp_path / "padded.fits"
    padded.write_bytes(path.read_bytes() + b"\0" * 100)
    compressed = _compress(padded, "gzip")

    scan = opening._scan(compressed)

    assert len(scan.every()) == 1
    assert scan.complete


def test_a_non_ascii_header_byte_reads_as_astropy_reads_it(tmp_path) -> None:
    path = _primary(tmp_path, _truth(10, 10), {"OBSERVER": "Canas"})
    raw = path.read_bytes().replace(b"'Canas", b"'Ca\xf1as")
    path.write_bytes(raw)
    compressed = _compress(path, "gzip")

    observer = read_image_headers(compressed).header["OBSERVER"]

    assert observer == "Ca?as"


def test_file_compression_is_read_from_the_first_bytes(tmp_path) -> None:
    plain = _primary(tmp_path, _truth(5, 5))

    assert file_compression(plain) is None
    assert file_compression(_compress(plain, "gzip")) == "gzip"
    assert file_compression(_compress(plain, "bzip2")) == "bzip2"


# --- The pixels ---------------------------------------------------------------


@pytest.mark.parametrize("compression", [None, "gzip", "bzip2", "tile"])
def test_the_pixels_of_the_whole_image_and_of_a_region_are_the_truth(
    tmp_path, compression
) -> None:
    truth = _truth(200, 300, np.int32 if compression == "tile" else np.float32)
    cards = _wcs_cards(300, 200, scale=0.5)

    if compression == "tile":
        path = _write(
            tmp_path / "a.fits.fz",
            [fits.PrimaryHDU(), fits.CompImageHDU(truth, header=_header(cards))],
        )
    else:
        path = _write(
            tmp_path / "a.fits",
            [fits.PrimaryHDU(truth, header=_header(cards))],
            compression,
        )

    whole = open_image(path, project_root=tmp_path)

    assert whole.region is None
    assert np.array_equal(whole.pixels, truth)

    # A side of 25 arcsec at 0.5 arcsec per pixel is 50 pixels.
    region = open_image(
        path,
        project_root=tmp_path,
        region={"size": 25.0},
        target_position={"x": 120.0, "y": 80.0},
    )

    assert region.region.as_dict() == {"xmin": 95, "xmax": 145, "ymin": 55, "ymax": 105}
    assert np.array_equal(region.pixels, truth[55:105, 95:145])
    assert region.image_shape == (200, 300)
    assert region.center == (149.5, 99.5)


@pytest.mark.parametrize("compression", [None, "gzip"])
@pytest.mark.parametrize(
    "window",
    [
        (0, 300, 0, 200),
        (0, 64, 30, 130),
        (299, 300, 0, 200),
        (0, 300, 199, 200),
        (17, 18, 3, 4),
    ],
)
def test_a_window_at_the_edges_reads_the_truth(tmp_path, compression, window) -> None:
    truth = _truth(200, 300)
    path = _write(tmp_path / "a.fits", [fits.PrimaryHDU(truth)], compression)
    headers = read_image_headers(path)
    xmin, xmax, ymin, ymax = window
    region = WorkingRegion(
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, image_width=300, image_height=200
    )

    pixels, _ = opening._read_pixels(headers, region)

    assert np.array_equal(pixels, truth[ymin:ymax, xmin:xmax])


@pytest.mark.parametrize("compression", [None, "gzip"])
def test_scaled_integers_read_as_their_values(tmp_path, compression) -> None:
    truth = _truth(30, 40, np.uint16)
    unsigned = _write(tmp_path / "u.fits", [fits.PrimaryHDU(truth)], compression)

    with fits.open(unsigned) as hdul:
        assert hdul[0].header["BZERO"] == 32768

    opened = open_image(unsigned, project_root=tmp_path)

    assert np.array_equal(opened.pixels, truth)
    assert opened.floating_point is False

    values = 100.0 + 0.5 * (np.arange(30 * 40).reshape(30, 40) % 2000)
    scaled = fits.PrimaryHDU(values)
    scaled.scale("int16", bscale=0.5, bzero=100.0)
    path = _write(tmp_path / "s.fits", [scaled], compression)

    with fits.open(path) as hdul:
        assert hdul[0].header["BITPIX"] == 16
        assert hdul[0].header["BSCALE"] == 0.5

    with fits.open(path) as hdul:
        expected = np.asarray(hdul[0].data, dtype=float)

    assert np.array_equal(open_image(path, project_root=tmp_path).pixels, expected)
    assert open_image(path, project_root=tmp_path).floating_point is False


def test_blank_pixels_of_an_integer_image_hold_no_data(tmp_path) -> None:
    stored = (np.arange(20 * 30).reshape(20, 30) % 1000).astype(np.int16)
    stored[3, 4] = -999
    path = _primary(tmp_path, stored, {"BLANK": -999})

    opened = open_image(path, project_root=tmp_path)

    assert np.isnan(opened.pixels[3, 4])
    assert opened.no_data[3, 4]
    assert np.count_nonzero(opened.no_data) == 1
    assert opened.record["no_data"]["pixels"] == 1


@pytest.mark.filterwarnings("ignore:File may have been truncated")
@pytest.mark.parametrize("compression", [None, "gzip"])
def test_a_truncated_file_is_refused(tmp_path, compression) -> None:
    path = _primary(tmp_path, _truth(200, 300))
    raw = path.read_bytes()
    cut = tmp_path / "cut.fits"
    cut.write_bytes(raw[: len(raw) // 2])

    if compression is not None:
        cut = _compress(cut, compression)

    with pytest.raises(ValueError, match="cannot read the pixels"):
        open_image(cut, project_root=tmp_path)


def test_a_gzip_stream_cut_in_half_is_refused(tmp_path) -> None:
    path = _compress(_primary(tmp_path, _truth(200, 300)), "gzip")
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) // 2])

    with pytest.raises(ValueError, match="cannot read the"):
        open_image(path, project_root=tmp_path)


def test_the_opened_arrays_are_read_only(tmp_path) -> None:
    opened = open_image(_primary(tmp_path, _truth(10, 10)), project_root=tmp_path)

    for array in (opened.pixels, opened.no_data):
        with pytest.raises(ValueError):
            array[0, 0] = 1


def test_a_one_pixel_image_and_a_float64_image_open(tmp_path) -> None:
    one = open_image(
        _primary(tmp_path, np.ones((1, 1), np.float32)), project_root=tmp_path
    )

    assert one.pixels.shape == (1, 1)

    truth = _truth(20, 30, np.float64)
    path = _primary(tmp_path, truth, name="b.fits")

    assert np.array_equal(open_image(path, project_root=tmp_path).pixels, truth)


# --- The pixels without data -------------------------------------------------


def test_the_no_data_rule_counts_areas_of_zeros() -> None:
    image = np.ones((60, 60), np.float32)
    image[0:10, 0:9] = 0.0
    image[20:30, 20:30] = 0.0
    image[40:50, 40:50] = 0.0
    image[50:60, 50:60] = 0.0
    image[5, 50] = np.nan

    mask = no_data_mask(image, floating_point=True)

    assert NO_DATA_MIN_ZERO_PIXELS == 100
    assert not mask[0:10, 0:9].any()
    assert mask[20:30, 20:30].all()
    assert mask[40:50, 40:50].all() and mask[50:60, 50:60].all()
    assert mask[5, 50]
    assert np.count_nonzero(mask) == 301

    assert np.count_nonzero(no_data_mask(image, floating_point=False)) == 1


def test_zeros_that_touch_at_a_corner_are_separate_areas() -> None:
    image = np.ones((20, 20), np.float32)
    image[0:7, 0:7] = 0.0
    image[7:14, 7:14] = 0.0

    assert not no_data_mask(image, floating_point=True).any()

    # One more zero joins the two areas into 99 zeros, still data; one more
    # makes 100.
    image[6, 7] = 0.0

    assert not no_data_mask(image, floating_point=True).any()

    image[7, 6] = 0.0

    assert np.count_nonzero(no_data_mask(image, floating_point=True)) == 100


def test_negative_zero_is_a_zero_and_a_read_only_array_is_accepted() -> None:
    image = np.full((10, 10), -0.0, np.float32)
    image.flags.writeable = False

    assert no_data_mask(image, floating_point=True).all()


@pytest.mark.parametrize(
    ("data", "minimum", "message"),
    [
        (np.zeros((2, 2, 2)), 100, "two-dimensional numeric"),
        (np.zeros((2, 2)), 0, "positive integer"),
        (np.zeros((2, 2)), True, "positive integer"),
    ],
)
def test_the_no_data_rule_refuses_invalid_input(data, minimum, message) -> None:
    with pytest.raises(ValueError, match=message):
        no_data_mask(data, floating_point=True, min_zero_pixels=minimum)


def test_zeros_cut_by_the_region_hold_no_data_as_in_the_whole_image(tmp_path) -> None:
    """A region that holds 80 of 400 connected zeros still flags those 80."""
    image = _truth(300, 300)
    image[100:120, 180:200] = 0.0
    cards = _wcs_cards(300, 300, scale=1.0)
    path = _primary(tmp_path, image, cards)

    whole = open_image(path, project_root=tmp_path)
    region = open_image(
        path,
        project_root=tmp_path,
        region={"size": 100.0},
        target_position={"x": 134.0, "y": 110.0},
    )

    assert region.region.as_dict() == {"xmin": 84, "xmax": 184, "ymin": 60, "ymax": 160}
    assert np.count_nonzero(region.no_data) == 80
    rows, columns = region.region.local_slices(xmin=84, xmax=184, ymin=60, ymax=160)
    assert np.array_equal(region.no_data, whole.no_data[60:160, 84:184])
    assert rows == slice(0, 100) and columns == slice(0, 100)


@pytest.mark.parametrize(
    ("dtype", "tile", "floating"),
    [
        (np.float32, False, True),
        (np.int16, False, False),
        (np.float32, True, True),
        (np.int32, True, False),
    ],
)
def test_the_storage_says_whether_the_image_is_in_floating_point(
    tmp_path, dtype, tile, floating
) -> None:
    data = _truth(20, 20, dtype)

    if tile:
        path = _write(
            tmp_path / "a.fits.fz", [fits.PrimaryHDU(), fits.CompImageHDU(data)]
        )
    else:
        path = _primary(tmp_path, data)

    assert open_image(path, project_root=tmp_path).floating_point is floating


# --- The working region -----------------------------------------------------


def _scale(value: float | None = 0.21) -> PixelScale:
    return PixelScale(value=value, source="wcs", wcs_value=value)


@pytest.mark.parametrize(
    ("shape", "setting", "target", "bounds"),
    [
        ((4096, 4096), None, None, None),
        ((100, 4097), None, "center", (0, 4096, 0, 100)),
        ((5000, 5000), None, {"x": 100.0, "y": 100.0}, (0, 4096, 0, 4096)),
        ((5000, 5000), None, {"x": 4950.0, "y": 4950.0}, (904, 5000, 904, 5000)),
        ((5000, 5000), None, {"x": 2500.4, "y": 2500.6}, (452, 4548, 453, 4549)),
        ((20000, 3000), None, "center", (0, 3000, 7952, 12048)),
        ((5000, 5000), "whole", None, None),
        (
            (5000, 5000),
            {"size": 300.0},
            {"x": 2500.0, "y": 2500.0},
            (1786, 3215, 1786, 3215),
        ),
        ((5000, 5000), {"size": 100000.0}, None, None),
        ((1451, 1451), {"size": 300.0}, "center", (11, 1440, 11, 1440)),
    ],
)
def test_the_working_region_follows_its_setting(shape, setting, target, bounds) -> None:
    region, reason = resolve_working_region(
        setting, image_shape=shape, target_position=target, pixel_scale=_scale()
    )

    assert reason

    if bounds is None:
        assert region is None
    else:
        assert (region.xmin, region.xmax, region.ymin, region.ymax) == bounds
        assert region.image_shape == shape


def test_a_region_about_the_target_needs_a_target_inside_the_image() -> None:
    with pytest.raises(ValueError, match="set data.target_position, or declare"):
        resolve_working_region(
            None, image_shape=(5000, 5000), target_position=None, pixel_scale=_scale()
        )

    with pytest.raises(ValueError, match=r"data.target_position \(6000, 10\) lies"):
        resolve_working_region(
            None,
            image_shape=(5000, 5000),
            target_position={"x": 6000.0, "y": 10.0},
            pixel_scale=_scale(),
        )

    region, _ = resolve_working_region(
        None, image_shape=(100, 100), target_position=None, pixel_scale=_scale()
    )

    assert region is None


def test_a_side_in_arcsec_needs_a_pixel_scale() -> None:
    with pytest.raises(ValueError, match="holds no celestial WCS: declare"):
        resolve_working_region(
            {"size": 30.0},
            image_shape=(500, 500),
            target_position="center",
            pixel_scale=PixelScale(value=None, source=None, wcs_value=None),
        )

    reason = "The header declares a celestial WCS without its scale."

    with pytest.raises(ValueError, match=reason):
        resolve_working_region(
            {"size": 30.0},
            image_shape=(500, 500),
            target_position="center",
            pixel_scale=PixelScale(
                value=None, source=None, wcs_value=None, note=reason
            ),
        )


def test_a_wcs_without_its_scale_never_gives_a_one_pixel_region(tmp_path) -> None:
    cards = _wcs_cards(300, 300)

    for keyword in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"):
        del cards[keyword]

    path = _primary(tmp_path, _truth(300, 300), cards)

    with pytest.raises(ValueError, match="neither CDi_j nor CDELT1 and CDELT2"):
        open_image(
            path,
            project_root=tmp_path,
            region={"size": 300.0},
            target_position="center",
        )

    declared = open_image(
        path,
        project_root=tmp_path,
        region={"size": 30.0},
        target_position="center",
        pixel_scale=0.2,
    )

    assert declared.region.shape == (150, 150)


@pytest.mark.parametrize(
    ("size", "scale", "side"),
    [
        (300.0, 0.21, 1429),
        (100.0, 0.1, 1000),
        (300.09, 0.21, 1429),
        (1000.0000001, 1.0, 1000),
        (0.05, 0.21, 1),
        (2.0, 0.21, 10),
    ],
)
def test_a_side_in_arcsec_spans_the_pixels_it_needs(size, scale, side) -> None:
    assert opening._side_in_pixels(size, scale) == side


@pytest.mark.parametrize(
    "value",
    [
        {"size": 0},
        {"size": -1},
        {"size": "300"},
        {"size": True},
        {"size": float("nan")},
        {"side": 300},
        {"size": 300, "x": 1},
        "half",
        300,
        [300],
    ],
)
def test_an_invalid_region_setting_is_refused(value) -> None:
    with pytest.raises(ValueError, match="data.region must be null"):
        validate_region(value)


def test_a_region_setting_is_returned_in_canonical_form() -> None:
    assert validate_region(None) is None
    assert validate_region(" Whole ") == "whole"
    assert validate_region({"size": np.float64(30)}) == {"size": 30.0}


def test_a_working_region_validates_its_bounds() -> None:
    with pytest.raises(ValueError, match="lies outside the image"):
        WorkingRegion.about((100, 100), center_x=100.0, center_y=10.0, side=10)

    edge = WorkingRegion.about((100, 100), center_x=-0.5, center_y=10.0, side=10)

    assert edge.xmin == 0

    covering = WorkingRegion.about((100, 100), center_x=50.0, center_y=50.0, side=500)

    assert covering.is_whole

    with pytest.raises(ValueError, match="positive integer"):
        WorkingRegion.about((100, 100), center_x=50.0, center_y=50.0, side=0)

    with pytest.raises(ValueError, match="must be an integer"):
        WorkingRegion(
            xmin=0.0, xmax=10, ymin=0, ymax=10, image_width=10, image_height=10
        )

    with pytest.raises(ValueError, match="does not lie inside"):
        WorkingRegion(xmin=0, xmax=11, ymin=0, ymax=10, image_width=10, image_height=10)

    region = WorkingRegion(
        xmin=100, xmax=200, ymin=50, ymax=150, image_width=300, image_height=300
    )

    assert region.local_slices(xmin=110, xmax=120, ymin=60, ymax=70) == (
        slice(10, 20),
        slice(10, 20),
    )

    with pytest.raises(ValueError, match="does not lie in the working region"):
        region.local_slices(xmin=0, xmax=10, ymin=60, ymax=70)

    assert region.expanded(100).as_dict() == {
        "xmin": 0,
        "xmax": 300,
        "ymin": 0,
        "ymax": 250,
    }


# --- The pixel scale ----------------------------------------------------------


def test_the_pixel_scale_comes_from_the_wcs_or_from_the_declaration() -> None:
    header = _header(_wcs_cards(100, 100, scale=0.21))
    reading = read_header(header)

    assert resolve_pixel_scale(None, reading).value == pytest.approx(0.21)
    assert resolve_pixel_scale("auto", reading).source == "wcs"

    agreeing = resolve_pixel_scale(0.21, reading)

    assert agreeing.source == "declared"
    assert agreeing.note is None

    contradicting = resolve_pixel_scale(0.1, reading)

    assert contradicting.value == 0.1
    assert contradicting.wcs_value == pytest.approx(0.21)
    assert "differs from the 0.21 of the WCS of the header by 52.4 %" in (
        contradicting.note
    )

    without = resolve_pixel_scale(None, read_header(_header({"BUNIT": "ADU"})))

    assert without.value is None
    assert without.note is None


def test_a_declared_pixel_scale_sets_the_region_and_is_reported(tmp_path) -> None:
    path = _primary(tmp_path, _truth(300, 300), _wcs_cards(300, 300, scale=0.21))

    opened = open_image(
        path,
        project_root=tmp_path,
        region={"size": 21.0},
        target_position="center",
        pixel_scale=0.1,
    )

    assert opened.region.shape == (210, 210)
    assert opened.pixel_scale.value == 0.1
    assert "the declared value is used" in opened.record["pixel_scale"]["note"]


# --- The identities, the record and the reuse of headers ---------------------


def test_the_headers_of_a_file_are_read_once_until_it_changes(
    tmp_path, monkeypatch
) -> None:
    calls = {"stream": 0}
    stream = opening._stream_headers

    def counted(*args, **kwargs):
        calls["stream"] += 1
        return stream(*args, **kwargs)

    monkeypatch.setattr(opening, "_stream_headers", counted)
    path = _write(tmp_path / "a.fits", [fits.PrimaryHDU(_truth(20, 30))], "gzip")

    first = read_image_headers(path)
    read_image_headers(path)
    read_image_headers(path, declarations={"gain": 2.0})

    assert calls["stream"] == 1

    # A declaration changes the reading, not what is read from the file.
    declared = read_image_headers(path, declarations={"gain": 2.0})

    assert declared.reading.value("gain") == 2.0
    assert first.reading.value("gain") is None

    _write(tmp_path / "a.fits", [fits.PrimaryHDU(_truth(20, 30) + 1)], "gzip")
    read_image_headers(path)

    assert calls["stream"] == 2


def test_a_kept_header_is_never_changed_by_its_reader(tmp_path) -> None:
    path = _primary(tmp_path, _truth(10, 10), {"EXPTIME": 10.0})

    read_image_headers(path).header["EXPTIME"] = 99.0

    assert read_image_headers(path).header["EXPTIME"] == 10.0


def _project(root: Path) -> Path:
    """Write a small project with a science image in data/."""
    (root / "data").mkdir(parents=True)
    primary = fits.PrimaryHDU(header=_header({"EXPTIME": 100.0}))
    fits.HDUList(
        [
            primary,
            fits.ImageHDU(
                _truth(300, 300), header=_header(_wcs_cards(300, 300)), name="SCI"
            ),
        ]
    ).writeto(root / "data" / "image.fits")

    return root


def test_a_moved_project_keeps_its_identities(tmp_path) -> None:
    first = _project(tmp_path / "first")
    second = tmp_path / "elsewhere" / "second"
    shutil.copytree(first, second)

    here = open_image("data/image.fits", project_root=first)
    there = open_image("data/image.fits", project_root=second)

    assert here.image_identity == there.image_identity
    assert here.identity == there.identity
    assert here.record["file"]["path"] == "data/image.fits"
    assert here.record["file"]["inside_project"] is True


def test_the_spelling_of_the_hdu_does_not_change_the_identity(tmp_path) -> None:
    root = _project(tmp_path / "project")
    identities = {
        open_image("data/image.fits", project_root=root, hdu=hdu).identity
        for hdu in (None, 1, "SCI", " sci ")
    }

    assert len(identities) == 1

    plain = _primary(tmp_path, _truth(10, 10))

    assert (
        open_image(plain, project_root=tmp_path).identity
        == open_image(plain, project_root=tmp_path, hdu=0).identity
    )


def test_the_calibration_changes_the_identity_of_the_opening_not_of_its_pixels(
    tmp_path,
) -> None:
    root = _project(tmp_path / "project")
    plain = open_image("data/image.fits", project_root=root)
    declared = open_image(
        "data/image.fits", project_root=root, calibration={"gain": 2.5}
    )
    scaled = open_image("data/image.fits", project_root=root, pixel_scale=0.2)

    assert declared.image_identity == plain.image_identity
    assert declared.identity != plain.identity
    assert declared.reading.value("gain") == 2.5
    assert scaled.image_identity == plain.image_identity
    assert scaled.identity != plain.identity


def test_the_region_enters_the_identity_of_the_pixels(tmp_path) -> None:
    root = _project(tmp_path / "project")
    whole = open_image("data/image.fits", project_root=root)
    also_whole = open_image("data/image.fits", project_root=root, region="whole")
    region = open_image(
        "data/image.fits",
        project_root=root,
        region={"size": 20.0},
        target_position="center",
    )

    assert whole.image_identity == also_whole.image_identity
    assert region.image_identity != whole.image_identity
    assert region.record["region"]["bounds"] == {
        "xmin": 100,
        "xmax": 200,
        "ymin": 100,
        "ymax": 200,
    }


def test_a_file_outside_the_project_is_recorded_by_its_absolute_path(tmp_path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    path = _primary(tmp_path, _truth(10, 10))

    opened = open_image(path, project_root=root)

    assert opened.record["file"]["inside_project"] is False
    assert opened.record["file"]["path"] == path.resolve().as_posix()


def test_the_record_is_json_and_the_product_round_trips(tmp_path) -> None:
    root = _project(tmp_path / "project")
    opened = open_image(
        "data/image.fits", project_root=root, calibration={"saturation": 5.0e4}
    )

    record = json.loads(json.dumps(dict(opened.record)))

    assert record["hdu"] == {"index": 1, "name": "SCI", "requested": None}
    assert record["image"]["width"] == 300
    assert record["image"]["floating_point"] is True
    assert record["reading"]["quantities"]["saturation"]["status"] == "declared"
    assert record["reading"]["quantities"]["exposure_time"]["source"] == "primary"
    assert record["image_identity"] == opened.image_identity
    assert record["conventions"]["matched"] == ["generic"]

    store = ProductStore(tmp_path / "products")
    store.save(opened.product())
    loaded = store.load(OPENING_STAGE, identity=opened.identity)

    assert loaded.record["image_identity"] == opened.image_identity


def test_the_opening_reads_its_settings_from_the_configuration(tmp_path) -> None:
    root = _project(tmp_path / "project")
    settings = read_preparation_settings(
        {
            "object": {"name": "galaxy"},
            "data": {
                "image_path": "data/image.fits",
                "hdu": "SCI",
                "region": {"size": 20.0},
                "target_position": [120, 130],
                "calibration": {"zeropoint": 25.0, "photometric_system": "AB"},
                "cutout": {"size": 51},
            },
            "outputs": {"root": "runs"},
        }
    )

    opened = open_observation(settings, project_root=root)

    assert opened.headers.hdu.index == 1
    assert opened.region.as_dict() == {
        "xmin": 70,
        "xmax": 170,
        "ymin": 80,
        "ymax": 180,
    }
    assert opened.reading.value("zeropoint") == 25.0
    assert opened.reading.value("photometric_system") == "AB"
    assert np.array_equal(opened.pixels, _truth(300, 300)[80:180, 70:170])


def test_a_missing_file_or_project_is_refused(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="FITS file not found"):
        open_image("data/none.fits", project_root=tmp_path)

    with pytest.raises(FileNotFoundError, match="Project folder not found"):
        open_image("a.fits", project_root=tmp_path / "none")

    assert WHOLE_IMAGE_MAX_SIDE == 4096
