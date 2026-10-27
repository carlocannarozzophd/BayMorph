"""Tests for a target given on the sky, and for the note on SIP terms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy import log as astropy_log
from astropy.coordinates import FK5, SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

import baymorph.pipeline.opening as opening
from baymorph import ObservationAdvisor
from baymorph.cli import main
from baymorph.core.header_metadata import read_header
from baymorph.pipeline.config_maker import make_config_from_fits
from baymorph.pipeline.opening import (
    RegionRequest,
    SkyPosition,
    read_observation_metadata,
    working_region,
)


@pytest.fixture(autouse=True)
def _fresh_openings():
    """Start and end every test without observations kept for reuse."""
    opening.forget_opened_observations()
    yield
    opening.forget_opened_observations()


def _header(scale: float = 0.1, *, radesys: str | None = None) -> fits.Header:
    """Return a tangent-plane header of ``scale`` arcsec per pixel."""
    header = fits.Header()

    for key, value in (
        ("CTYPE1", "RA---TAN"),
        ("CTYPE2", "DEC--TAN"),
        ("CRPIX1", 50.0),
        ("CRPIX2", 40.0),
        ("CRVAL1", 150.0),
        ("CRVAL2", 2.0),
        ("CD1_1", -scale / 3600.0),
        ("CD1_2", 0.0),
        ("CD2_1", 0.0),
        ("CD2_2", scale / 3600.0),
    ):
        header[key] = value

    if radesys is not None:
        header["RADESYS"] = radesys
        header["EQUINOX"] = 2000.0

    return header


def _mosaic(path: Path, header: fits.Header | None = None) -> None:
    """Write a 300 x 400 image with a galaxy at (330, 70)."""
    rng = np.random.default_rng(11)
    yy, xx = np.mgrid[0:300, 0:400]
    image = 10.0 + rng.normal(0.0, 1.0, size=(300, 400))
    image += 200.0 * np.exp(-0.5 * ((yy - 70.0) ** 2 + (xx - 330.0) ** 2) / 4.0**2)

    fits.writeto(path, image, header=_header() if header is None else header)


def _sky_of(path: Path, x: float, y: float) -> dict[str, float]:
    """Return the ICRS position of a pixel of a file, through its WCS."""
    position = WCS(fits.getheader(path)).pixel_to_world(x, y).icrs

    return {"ra": float(position.ra.deg), "dec": float(position.dec.deg)}


# --- reading a position on the sky ------------------------------------------


def test_a_sky_target_reads_degrees_and_texts_with_units():
    assert SkyPosition.of({"ra": 3.5237, "dec": -30.37241}) == SkyPosition(
        3.5237, -30.37241
    )

    texts = SkyPosition.of({"ra": "00h14m05.688s", "dec": "-30d22m20.676s"})

    assert texts.ra == pytest.approx(3.5237, abs=1e-9)
    assert texts.dec == pytest.approx(-30.37241, abs=1e-9)
    assert texts.as_dict() == {"ra": texts.ra, "dec": texts.dec, "frame": "icrs"}


@pytest.mark.parametrize(
    "target, message",
    [
        ({"ra": "00:14:05.69", "dec": -30.0}, "without a unit"),
        ({"ra": "3.5237", "dec": -30.0}, "without a unit"),
        ({"ra": True, "dec": -30.0}, "number of degrees"),
        ({"ra": 360.0, "dec": 0.0}, r"\[0, 360\)"),
        ({"ra": float("nan"), "dec": 0.0}, r"\[0, 360\)"),
        ({"ra": 1.0, "dec": 91.0}, r"\[-90, 90\]"),
        ({"ra": 1.0}, "exactly 'ra' and 'dec'"),
        ({"ra": 1.0, "dec": 2.0, "x": 3.0}, "exactly 'ra' and 'dec'"),
    ],
)
def test_a_sky_target_must_be_a_readable_position_on_the_sphere(target, message):
    with pytest.raises(ValueError, match=message):
        SkyPosition.of(target)


def test_a_sky_target_falls_on_the_pixel_of_its_wcs(tmp_path):
    path = tmp_path / "mosaic.fits"
    _mosaic(path)
    metadata = read_observation_metadata(path)

    x, y = SkyPosition.of(_sky_of(path, 330.0, 70.0)).pixel(metadata)

    assert (x, y) == pytest.approx((330.0, 70.0), abs=1e-6)


def test_a_sky_target_is_converted_from_icrs_to_the_frame_of_the_header(tmp_path):
    # At 0.01 arcsec per pixel the offset between ICRS and FK5 is a few pixels.
    header = _header(0.01, radesys="FK5")
    path = tmp_path / "fk5.fits"
    fits.writeto(path, np.zeros((300, 400)), header=header)
    target = SkyPosition(149.9995, 2.0003)
    fk5 = SkyCoord(target.ra, target.dec, unit="deg", frame="icrs").transform_to(
        FK5(equinox="J2000")
    )
    wcs = WCS(header)

    expected = wcs.all_world2pix(fk5.ra.deg, fk5.dec.deg, 0)
    naive = wcs.all_world2pix(target.ra, target.dec, 0)
    found = target.pixel(read_observation_metadata(path))

    assert found == pytest.approx((float(expected[0]), float(expected[1])), abs=1e-6)
    assert np.hypot(found[0] - naive[0], found[1] - naive[1]) > 0.5


def test_a_sky_target_outside_the_image_or_without_a_wcs_stops(tmp_path):
    path = tmp_path / "mosaic.fits"
    _mosaic(path)
    outside = SkyPosition.of(_sky_of(path, -50.0, 70.0))

    with pytest.raises(ValueError, match="outside the image of 400 x 300 pixels"):
        outside.pixel(read_observation_metadata(path))

    plain = tmp_path / "plain.fits"
    fits.writeto(plain, np.zeros((30, 40)))

    with pytest.raises(ValueError, match="no celestial WCS"):
        SkyPosition(150.0, 2.0).pixel(read_observation_metadata(plain))


# --- where a position on the sky is accepted ---------------------------------


def test_the_region_is_taken_about_a_sky_target(tmp_path):
    path = tmp_path / "mosaic.fits"
    _mosaic(path)
    metadata = read_observation_metadata(path)
    sky = SkyPosition.of(_sky_of(path, 330.0, 70.0))

    about_sky = RegionRequest.of({"size": 10}, target=sky).resolve(metadata)
    about_pixel = RegionRequest.of({"size": 10}, target=(330.0, 70.0)).resolve(metadata)

    assert about_sky == about_pixel


def test_a_configuration_gives_its_target_in_pixels(tmp_path):
    path = tmp_path / "mosaic.fits"
    _mosaic(path)
    data = {
        "image_path": str(path),
        "region": {"size": 10},
        "target_position": _sky_of(path, 330.0, 70.0),
    }

    with pytest.raises(ValueError, match="gives the target in pixels"):
        working_region(data)


def test_the_advisor_takes_a_target_on_the_sky(tmp_path):
    path = tmp_path / "mosaic.fits"
    _mosaic(path)
    sky = _sky_of(path, 330.0, 70.0)

    advisor = ObservationAdvisor(
        path, target_position=sky, region={"size": 10}, online=False
    )
    in_pixels = ObservationAdvisor(
        path, target_position=(330.0, 70.0), region={"size": 10}, online=False
    )
    inspection = advisor.inspect()

    assert isinstance(advisor.target_position, SkyPosition)
    assert (inspection.target_x, inspection.target_y) == pytest.approx(
        (330.0, 70.0), abs=1e-6
    )
    assert inspection.working_region == in_pixels.inspect().working_region
    assert inspection.target_area_pixels == in_pixels.inspect().target_area_pixels
    assert f"from RA {sky['ra']:.10g} and Dec {sky['dec']:.10g} (ICRS)" in (
        advisor.summary()
    )

    with pytest.raises(ValueError, match=r"\[-90, 90\]"):
        ObservationAdvisor(path, target_position={"ra": 150.0, "dec": 95.0})


def test_the_config_maker_writes_a_sky_target_in_pixels(tmp_path):
    path = tmp_path / "mosaic.fits"
    _mosaic(path)
    options = {
        "psf_type": "gaussian",
        "gaussian_fwhm_pixels": 2.0,
        "cutout_mode": "fixed",
        "fixed_size": 40,
        "region": {"size": 6},
        "_require_complete_priors": False,
    }

    on_sky = make_config_from_fits(
        path, target_position=_sky_of(path, 330.004, 69.996), **options
    )
    in_pixels = make_config_from_fits(path, target_position=(330.0, 70.0), **options)

    assert on_sky["data"]["target_position"] == {"x": 330.0, "y": 70.0}
    assert working_region(on_sky["data"]) == working_region(in_pixels["data"])


def test_the_config_command_takes_a_sky_target(tmp_path, monkeypatch, capsys):
    path = tmp_path / "mosaic.fits"
    _mosaic(path)
    sky = _sky_of(path, 330.0, 70.0)
    base = [
        "baymorph",
        "config",
        str(path),
        "--psf",
        "gaussian",
        "--gaussian-fwhm-pixels",
        "2",
        "--uniform-prior",
        "single_sersic.flux",
        "1",
        "1000",
        "--region-size",
        "10",
    ]
    ra = SkyCoord(sky["ra"], sky["dec"], unit="deg").ra.to_string(unit="hourangle")

    monkeypatch.setattr(
        "sys.argv",
        [
            *base,
            "--target-ra",
            ra,
            "--target-dec",
            f"{sky['dec']}",
            "--output",
            str(tmp_path / "sky.yaml"),
        ],
    )

    assert main() == 0

    written = yaml.safe_load((tmp_path / "sky.yaml").read_text())
    output = capsys.readouterr().out

    assert written["data"]["target_position"] == pytest.approx(
        {"x": 330.0, "y": 70.0}, abs=0.011
    )
    assert "(ICRS), at pixel (330.00, 70.00) of the whole image" in output

    for extra, message in (
        (["--target-ra", "150.0"], "must be supplied together"),
        (
            ["--target-ra", "150.0", "--target-dec", "2.0", "--target-x", "3"],
            "either in pixels",
        ),
        (
            ["--target-center", "--target-ra", "150.0", "--target-dec", "2.0"],
            "--target-center cannot be combined",
        ),
        (["--target-ra", "10:00:00", "--target-dec", "2.0"], "without a unit"),
    ):
        monkeypatch.setattr("sys.argv", [*base, *extra])

        with pytest.raises(SystemExit) as error:
            main()

        assert error.value.code == 2
        assert message in capsys.readouterr().err


# --- SIP terms without -SIP -------------------------------------------------


def _sip_header(ctype_suffix: str) -> fits.Header:
    """Return a header with SIP terms of order 2 and the given CTYPE suffix."""
    header = _header()
    header["CTYPE1"] = "RA---TAN" + ctype_suffix
    header["CTYPE2"] = "DEC--TAN" + ctype_suffix
    header["A_ORDER"] = 2
    header["B_ORDER"] = 2
    header["A_2_0"] = 1.0e-6
    header["B_0_2"] = 1.0e-6
    header["NAXIS"] = 2
    header["NAXIS1"] = 400
    header["NAXIS2"] = 300

    return header


def test_sip_terms_without_their_suffix_get_a_note_instead_of_astropy_info():
    with astropy_log.log_to_list() as messages:
        reading = read_header(_sip_header(""))

    assert not any("SIP" in str(message.msg) for message in messages)
    assert "SIP distortion terms (A_ORDER = 2)" in reading["pixel_scale"].note
    assert "BayMorph leaves them out" in reading["pixel_scale"].note
    assert "Notes" in reading.report()


def test_a_sky_target_ignores_sip_terms_that_the_ctype_does_not_declare(tmp_path):
    # At 300 pixels from the reference pixel, A_2_0 = 1e-4 moves x by 9 pixels.
    tangent = tmp_path / "tangent.fits"
    _mosaic(tangent)
    target = SkyPosition.of(_sky_of(tangent, 349.0, 250.0))

    found = {}

    for suffix in ("", "-SIP"):
        header = _sip_header(suffix)
        header["A_2_0"] = 1.0e-4
        path = tmp_path / f"sip{suffix or '_undeclared'}.fits"
        _mosaic(path, header)

        with astropy_log.log_to_list() as messages:
            found[suffix] = target.pixel(read_observation_metadata(path))

        assert not any("SIP" in str(message.msg) for message in messages)

    assert found[""] == pytest.approx((349.0, 250.0), abs=1e-6)
    assert abs(found["-SIP"][0] - 349.0) > 5.0


def test_sip_terms_with_their_suffix_or_no_sip_get_no_note():
    for header in (_sip_header("-SIP"), _header()):
        with astropy_log.log_to_list() as messages:
            reading = read_header(header)

        assert not any("SIP" in str(message.msg) for message in messages)
        assert "SIP" not in reading["pixel_scale"].note
