"""Tests for the selection of the image HDU and data.hdu (AD-049)."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from baymorph.core.image import ImageData, read_image_header, select_image_hdu
from baymorph.core.observation import Observation
from baymorph.pipeline.config import resolve_config
from baymorph.pipeline.config_maker import make_config_from_fits
from baymorph.pipeline.prepared_identity import build_prepared_input_identity


def _image(value: float = 0.0, shape: tuple[int, int] = (12, 10)) -> np.ndarray:
    """Return deterministic pixels."""
    return np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + value


def _wcs_header() -> fits.Header:
    """Return a header with a celestial WCS of 0.25 arcsec per pixel."""
    header = fits.Header()
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CRPIX1"] = 5.0
    header["CRPIX2"] = 6.0
    header["CRVAL1"] = 150.0
    header["CRVAL2"] = 2.0
    header["CD1_1"] = -0.25 / 3600.0
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 0.25 / 3600.0
    return header


def _write_extensions(path, extensions) -> None:
    """Write an empty primary HDU followed by the given extensions."""
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "Example telescope"
    primary.header["EXPTIME"] = 600.0
    fits.HDUList([primary, *extensions]).writeto(path)


@pytest.mark.filterwarnings("ignore:File may have been truncated")
def test_the_header_is_read_without_the_pixels(tmp_path) -> None:
    """The reading of the header needs no pixel, even in a file without them."""
    path = tmp_path / "header_only.fits"
    header = _wcs_header()
    header["BUNIT"] = "ADU/s"
    fits.PrimaryHDU(data=_image(shape=(64, 64)), header=header).writeto(path)
    header_size = len(fits.getheader(path).tostring())
    path.write_bytes(path.read_bytes()[:header_size])

    reading = read_image_header(path)

    assert path.stat().st_size < 64 * 64 * 4
    assert reading.value("image_unit") == "ADU/s"
    assert reading.value("pixel_scale") == pytest.approx(0.25)


def test_the_header_reading_equals_the_reading_of_the_opened_image(
    tmp_path,
) -> None:
    """Same HDU, same headers and the same inherited primary keywords."""
    path = tmp_path / "extensions.fits"
    _write_extensions(
        path,
        [
            fits.ImageHDU(_image(), header=_wcs_header(), name="SCI"),
            fits.ImageHDU(_image(1.0), name="WHT"),
        ],
    )
    opened = Observation.from_fits(path).metadata.header_reading

    reading = read_image_header(path)

    assert reading.as_dict() == opened.as_dict()
    assert reading["exposure_time"].source == "primary"
    assert read_image_header(path, hdu="WHT").as_dict() == (
        Observation.from_fits(path, hdu="WHT").metadata.header_reading.as_dict()
    )


def test_a_primary_image_loads_as_before(tmp_path) -> None:
    path = tmp_path / "primary.fits"
    data = _image()
    fits.PrimaryHDU(data=data, header=_wcs_header()).writeto(path)

    image = ImageData.from_fits(path)

    assert image.hdu_index == 0
    assert image.hdu_name is None
    assert image.primary_header is None
    assert image.data.dtype == np.float64
    assert np.array_equal(image.data, fits.getdata(path).astype(float))


def test_an_empty_primary_selects_the_science_extension(tmp_path) -> None:
    path = tmp_path / "drizzled.fits"
    _write_extensions(
        path,
        [
            fits.ImageHDU(data=_image(1.0), header=_wcs_header(), name="SCI"),
            fits.ImageHDU(data=_image(2.0), name="WHT"),
            fits.ImageHDU(data=_image(3.0), name="ERR"),
        ],
    )

    observation = Observation.from_fits(path)

    assert observation.image.hdu_index == 1
    assert observation.image.hdu_name == "SCI"
    assert np.array_equal(observation.image.data, _image(1.0))
    assert observation.metadata.pixel_scale == pytest.approx(0.25)

    reading = observation.metadata.header_reading
    assert reading["telescope"].value == "Example telescope"
    assert reading["telescope"].source == "primary"
    assert observation.metadata.telescope == "Example telescope"
    assert observation.metadata.exposure_time == pytest.approx(600.0)


def test_an_image_extension_is_selected_when_named_image(tmp_path) -> None:
    path = tmp_path / "calexp.fits"
    _write_extensions(
        path,
        [
            fits.ImageHDU(data=_image(1.0), name="IMAGE"),
            fits.ImageHDU(data=_image(2.0), name="MASK"),
            fits.ImageHDU(data=_image(3.0), name="VARIANCE"),
        ],
    )

    with fits.open(path) as hdul:
        assert select_image_hdu(hdul) == 1


def test_the_science_extension_is_preferred_to_a_background_subtracted_one(
    tmp_path,
) -> None:
    """A CEERS mosaic holds SCI_BKSUB before SCI; SCI is the default."""
    path = tmp_path / "ceers_i2d.fits"
    _write_extensions(
        path,
        [
            fits.ImageHDU(data=_image(1.0), name="SCI_BKSUB"),
            fits.ImageHDU(data=_image(2.0), header=_wcs_header(), name="SCI"),
            fits.ImageHDU(data=_image(3.0), name="ERR"),
        ],
    )

    image = ImageData.from_fits(path)

    assert image.hdu_index == 2
    assert image.hdu_name == "SCI"
    assert ImageData.from_fits(path, hdu="SCI_BKSUB").hdu_index == 1


def test_a_single_image_extension_is_selected(tmp_path) -> None:
    path = tmp_path / "compressed.fits"
    _write_extensions(path, [fits.CompImageHDU(data=_image(4.0))])

    image = ImageData.from_fits(path)

    assert image.hdu_index == 1
    assert np.allclose(image.data, _image(4.0))


def test_several_unnamed_images_require_a_declaration(tmp_path) -> None:
    path = tmp_path / "mosaic.fits"
    _write_extensions(
        path,
        [
            fits.ImageHDU(data=_image(1.0), name="N4"),
            fits.ImageHDU(data=_image(2.0), name="S4"),
        ],
    )

    with pytest.raises(ValueError, match="data.hdu"):
        ImageData.from_fits(path)

    by_name = ImageData.from_fits(path, hdu="s4")
    by_index = ImageData.from_fits(path, hdu=1)

    assert by_name.hdu_index == 2
    assert np.array_equal(by_name.data, _image(2.0))
    assert by_index.hdu_name == "N4"


@pytest.mark.parametrize(
    ("hdu", "message"),
    [
        (5, "does not exist"),
        ("SCI", "no HDU named"),
        (2, "no two-dimensional image"),
        (True, "index or an extension name"),
    ],
)
def test_a_declared_hdu_must_hold_an_image(tmp_path, hdu, message) -> None:
    path = tmp_path / "with_table.fits"
    table = fits.BinTableHDU.from_columns(
        [fits.Column(name="flux", format="E", array=np.ones(3))],
        name="CATALOG",
    )
    _write_extensions(path, [fits.ImageHDU(data=_image(), name="IMG"), table])

    with pytest.raises(ValueError, match=message):
        ImageData.from_fits(path, hdu=hdu)


def test_a_file_without_images_is_rejected(tmp_path) -> None:
    path = tmp_path / "table_only.fits"
    table = fits.BinTableHDU.from_columns(
        [fits.Column(name="flux", format="E", array=np.ones(3))],
    )
    _write_extensions(path, [table])

    with pytest.raises(ValueError, match="no two-dimensional image"):
        ImageData.from_fits(path)


def _config(tmp_path, image_path, **data) -> dict:
    """Return a resolved single-observation configuration."""
    return resolve_config(
        {
            "object": {"name": "hdu_target"},
            "data": {
                "image_path": str(image_path),
                "target_position": {"x": 5, "y": 5},
                "cutout": {"mode": "fixed", "size": 10},
                **data,
            },
            "outputs": {"root": str(tmp_path / "outputs")},
        }
    )


@pytest.mark.parametrize("hdu", [0, 3, "SCI", "sci"])
def test_data_hdu_accepts_an_index_or_a_name(tmp_path, hdu) -> None:
    config = _config(tmp_path, tmp_path / "image.fits", hdu=hdu)

    assert config["data"]["hdu"] == hdu


@pytest.mark.parametrize("hdu", [-1, "", "  ", True, 1.5])
def test_data_hdu_rejects_other_values(tmp_path, hdu) -> None:
    with pytest.raises(ValueError, match="data.hdu"):
        _config(tmp_path, tmp_path / "image.fits", hdu=hdu)


def test_the_default_hdu_leaves_the_preparation_identity_unchanged(
    tmp_path,
) -> None:
    path = tmp_path / "primary.fits"
    fits.PrimaryHDU(data=_image(), header=_wcs_header()).writeto(path)

    identity = build_prepared_input_identity(_config(tmp_path, path))

    assert "hdu" not in identity["preparation_config"]["data"]
    assert "hdu" not in identity["fits"]


def test_a_declared_hdu_enters_the_preparation_identity(tmp_path) -> None:
    path = tmp_path / "mosaic.fits"
    _write_extensions(
        path,
        [
            fits.ImageHDU(data=_image(1.0), header=_wcs_header(), name="N4"),
            fits.ImageHDU(data=_image(2.0), header=_wcs_header(), name="S4"),
        ],
    )

    first = build_prepared_input_identity(_config(tmp_path, path, hdu="N4"))
    second = build_prepared_input_identity(_config(tmp_path, path, hdu="S4"))

    assert first["preparation_config"]["data"]["hdu"] == "N4"
    assert first["fits"]["hdu"] == {"index": 1, "name": "N4"}
    assert second["fits"]["hdu"] == {"index": 2, "name": "S4"}
    assert first["preparation_config_sha256"] != second["preparation_config_sha256"]
    assert first["fits"]["loaded_data_sha256"] != second["fits"]["loaded_data_sha256"]


def test_the_config_maker_records_a_declared_hdu(tmp_path) -> None:
    path = tmp_path / "mosaic.fits"
    _write_extensions(
        path,
        [
            fits.ImageHDU(data=_image(1.0), header=_wcs_header(), name="N4"),
            fits.ImageHDU(data=_image(2.0), header=_wcs_header(), name="S4"),
        ],
    )

    config = make_config_from_fits(
        path,
        target_position="center",
        hdu="S4",
        model_priors={"flux": {"type": "log_uniform", "lower": 1.0, "upper": 1e6}},
    )

    assert config["data"]["hdu"] == "S4"
    assert config["data"]["pixel_scale"] == pytest.approx(0.25)


def test_the_config_maker_omits_the_default_hdu(tmp_path) -> None:
    path = tmp_path / "primary.fits"
    fits.PrimaryHDU(data=_image(), header=_wcs_header()).writeto(path)

    config = make_config_from_fits(
        path,
        target_position="center",
        model_priors={"flux": {"type": "log_uniform", "lower": 1.0, "upper": 1e6}},
    )

    assert "hdu" not in config["data"]


def _compressed_stack_header() -> fits.Header:
    """Return a header with the asinh compression keywords of Pan-STARRS."""
    header = _wcs_header()
    header["BSOFTEN"] = 87.3959
    header["BOFFSET"] = 2.65496
    return header


def test_an_asinh_compressed_stack_is_refused_at_loading(tmp_path) -> None:
    path = tmp_path / "stack.fits"
    fits.PrimaryHDU(data=_image(), header=_compressed_stack_header()).writeto(path)

    with pytest.raises(ValueError, match="asinh encoding of the pixel values"):
        ImageData.from_fits(path)

    with pytest.raises(ValueError, match="cutout service are already linear"):
        Observation.from_fits(path)


def test_a_compressed_stack_in_a_tile_compressed_extension_is_refused(
    tmp_path,
) -> None:
    path = tmp_path / "stack.fits.fz"
    _write_extensions(
        path,
        [fits.CompImageHDU(data=_image(), header=_compressed_stack_header())],
    )

    with pytest.raises(ValueError, match="HDU 1 of"):
        ImageData.from_fits(path)


def test_a_cutout_linearized_by_astrocut_is_loaded(tmp_path) -> None:
    path = tmp_path / "cutout.fits"
    header = _compressed_stack_header()
    header["HISTORY"] = (
        "Pixel values converted from asinh-scaled flux to linear flux using the "
        "BSOFTEN/BOFFSET keywords."
    )
    fits.PrimaryHDU(data=_image(), header=header).writeto(path)

    image = ImageData.from_fits(path)

    assert np.array_equal(image.data, fits.getdata(path).astype(float))
