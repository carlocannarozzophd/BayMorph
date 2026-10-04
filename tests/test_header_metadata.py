"""Tests for the reading of FITS headers through conventions (AD-049)."""

from __future__ import annotations

import json

import pytest
from astropy.io import fits

from baymorph.core.header_metadata import (
    DERIVED_QUANTITIES,
    QUANTITIES,
    STATUSES,
    USABLE_STATUSES,
    load_header_conventions,
    normalize_declared_unit,
    read_header,
)
from baymorph.core.observation import ObservationMetadata


def _wcs_cards(header: fits.Header, scale_arcsec: float = 0.2) -> None:
    """Add a celestial WCS of the given pixel scale."""
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CRPIX1"] = 50.0
    header["CRPIX2"] = 50.0
    header["CRVAL1"] = 150.0
    header["CRVAL2"] = 2.0
    header["CD1_1"] = -scale_arcsec / 3600.0
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = scale_arcsec / 3600.0


def _generic_header() -> fits.Header:
    """Return a header that only the generic convention recognizes."""
    header = fits.Header()
    header["TELESCOP"] = "Example telescope"
    header["INSTRUME"] = "Example camera"
    header["FILTER"] = "r"
    header["BUNIT"] = "ADU"
    header["EXPTIME"] = 300.0
    header["GAIN"] = 1.5
    _wcs_cards(header)
    return header


def _coadd_header() -> fits.Header:
    """Return a synthetic header of a VST-Tube coadd combined by SWarp."""
    header = _generic_header()
    header["TELESCOP"] = "ESO-VST"
    header["INSTRUME"] = "OMEGACAM"
    header["BUNIT"] = "ADU/s"
    header["EXPTIME"] = (1.0, "Exposure time [s]")
    header["PROCSOFT"] = ("VST-tube vers. 9.9", "Pipeline")
    header["SOFTNAME"] = "SWarp"
    header["GAIN"] = (30.0, "Maximum equivalent gain (e-/ADU)")
    header["SATURATE"] = (20000.0, "Saturation Level (ADU)")
    header["SATLEVEL"] = (20000.0, "Saturation Level (ADU)")
    header["FWHM"] = (0.6, "FWHM median")
    header["PHOTZP"] = (30.0, "Phot. zeropoint. mag = -2.5log(ADU)+PHOTZP")
    header["PHOTZPER"] = 0.02
    header["PHOTSYS"] = "AB"
    header["NCOMBINE"] = 7
    header["TEXPTIME"] = (2100.0, "Total exposure time [s]")
    header["BKGMED"] = (250.0, "BACKMEAN median")
    header["ASTRERR"] = (0.25, 'RMS error of astrometric solution ["]')
    header["RESAMPT1"] = "LANCZOS3"
    header["COMBINET"] = "WEIGHTED"
    return header


def test_every_distributed_convention_loads_and_cites_its_sources() -> None:
    conventions = load_header_conventions()
    names = [convention.name for convention in conventions]

    assert names == sorted(names)
    assert len(names) == len(set(names))
    assert {"generic", "eso_phase3", "eso_raw", "swarp", "vst_tube"} <= set(names)

    for convention in conventions:
        assert convention.references
        assert set(convention.quantities) <= set(QUANTITIES)

    generic = next(item for item in conventions if item.name == "generic")
    assert generic.rules is None
    assert generic.priority == min(item.priority for item in conventions)


def test_reading_lists_every_quantity_with_a_known_status() -> None:
    reading = read_header(_generic_header())

    assert set(reading.quantities) == set(QUANTITIES) | set(DERIVED_QUANTITIES)
    assert all(item.status in STATUSES for item in reading.quantities.values())
    assert reading.conventions == ("generic",)


def test_statuses_of_a_generic_header() -> None:
    reading = read_header(_generic_header())

    telescope = reading["telescope"]
    assert telescope.status == "explicit"
    assert telescope.value == "Example telescope"
    assert telescope.keyword == "TELESCOP"
    assert telescope.source == "image"

    exposure = reading["exposure_time"]
    assert exposure.status == "convention"
    assert exposure.value == pytest.approx(300.0)
    assert exposure.unit == "s"

    gain = reading["gain"]
    assert gain.status == "convention"
    assert gain.unit == "electron/adu"

    pixel_scale = reading["pixel_scale"]
    assert pixel_scale.status == "derived"
    assert pixel_scale.value == pytest.approx(0.2)
    assert pixel_scale.unit == "arcsec/pixel"

    assert reading["zeropoint"].status == "missing"
    assert reading.value("zeropoint") is None


def test_a_unit_is_never_inferred_from_a_value() -> None:
    header = _generic_header()
    # 0.4 pixels would be an impossible FWHM, but no value implies a unit.
    header["FWHM"] = (0.4, "FWHM median")

    reading = read_header(header)
    fwhm = reading["psf_fwhm"]

    assert fwhm.status == "ambiguous"
    assert fwhm.unit is None
    assert fwhm.value is None
    assert fwhm.raw_value == pytest.approx(0.4)
    assert reading.value("psf_fwhm") is None


@pytest.mark.parametrize(
    ("card", "unit_keyword", "expected"),
    [
        ((3.5, "Median FWHM [pixels]"), None, "pixels"),
        ((0.8, "Seeing"), ("FWHMUNIT", "arcseconds"), "arcsec"),
        ((0.8, "Seeing"), ("SEEUNIT", "pix"), "pixels"),
    ],
)
def test_a_unit_declared_by_the_header_is_explicit(
    card,
    unit_keyword,
    expected,
) -> None:
    header = _generic_header()
    header["FWHM"] = card

    if unit_keyword is not None:
        header[unit_keyword[0]] = unit_keyword[1]

    fwhm = read_header(header)["psf_fwhm"]

    assert fwhm.status == "explicit"
    assert fwhm.unit == expected
    assert fwhm.value == pytest.approx(card[0])


def test_conflicting_unit_declarations_leave_the_unit_ambiguous() -> None:
    header = _generic_header()
    header["FWHM"] = (0.8, "Median FWHM [pixels]")
    header["FWHMUNIT"] = "arcsec"

    fwhm = read_header(header)["psf_fwhm"]

    assert fwhm.status == "ambiguous"
    assert fwhm.unit is None


def test_keywords_that_disagree_are_conflicting() -> None:
    header = _generic_header()
    header["SATLEVEL"] = 50000.0
    header["SATURATE"] = 45000.0

    saturation = read_header(header)["saturation"]

    assert saturation.status == "conflicting"
    assert saturation.value is None
    assert saturation.raw_value == pytest.approx(50000.0)
    assert [keyword for keyword, _ in saturation.candidates] == [
        "SATLEVEL",
        "SATURATE",
    ]


def test_keywords_that_agree_are_not_conflicting() -> None:
    header = _generic_header()
    header["SATLEVEL"] = 50000.0
    header["SATURATE"] = 50000.0

    saturation = read_header(header)["saturation"]

    assert saturation.status == "convention"
    assert saturation.value == pytest.approx(50000.0)
    assert saturation.unit == "image"


@pytest.mark.parametrize("value", ["not-a-number", "NaN"])
def test_a_malformed_number_is_ambiguous(value) -> None:
    header = _generic_header()
    header["GAIN"] = value

    gain = read_header(header)["gain"]

    assert gain.status == "ambiguous"
    assert gain.value is None


def test_a_blank_string_is_missing() -> None:
    header = _generic_header()
    header["FILTER"] = "   "

    assert read_header(header)["filter"].status == "missing"


def test_a_vst_tube_coadd_reads_through_its_conventions() -> None:
    reading = read_header(_coadd_header())

    assert reading.conventions == ("vst_tube", "swarp", "generic")

    assert reading["gain"].status == "convention"
    assert reading["gain"].convention == "swarp"
    assert reading["gain"].value == pytest.approx(30.0)

    sky = reading["sky_level"]
    assert sky.status == "convention"
    assert sky.convention == "vst_tube"
    assert sky.unit == "image"
    assert sky.value == pytest.approx(250.0)

    # VST-Tube documents no unit for its FWHM nor a rescaling of its
    # saturation level: both stay ambiguous.
    assert reading["psf_fwhm"].status == "ambiguous"
    assert reading["saturation"].status == "ambiguous"

    assert reading["zeropoint"].status == "convention"
    assert reading["zeropoint"].unit == "mag"
    assert reading["photometric_system"].value == "AB"
    assert reading["n_combined"].status == "explicit"
    assert reading["n_combined"].value == 7
    assert reading["total_exposure_time"].unit == "s"
    assert reading["astrometric_error"].unit == "arcsec"
    assert reading["resampling_kernel"].value == "LANCZOS3"
    assert reading["combine_type"].value == "WEIGHTED"


def test_a_non_integer_count_is_ambiguous() -> None:
    header = _coadd_header()
    header["NCOMBINE"] = 7.5

    assert read_header(header)["n_combined"].status == "ambiguous"


def test_an_eso_science_data_product_gives_its_resolution_in_arcsec() -> None:
    header = _generic_header()
    header["PRODCATG"] = "SCIENCE.IMAGE"
    header["PSF_FWHM"] = 0.85
    header["PHOTZP"] = 27.3
    header["PHOTSYS"] = "AB"

    reading = read_header(header)

    assert reading.conventions[0] == "eso_phase3"
    assert reading["psf_fwhm"].status == "convention"
    assert reading["psf_fwhm"].unit == "arcsec"
    assert reading["zeropoint"].status == "convention"

    metadata = ObservationMetadata.from_header(header, image_shape=(100, 100))
    assert metadata.seeing_fwhm == pytest.approx(0.85)
    assert metadata.seeing_fwhm_unit == "arcsec"


def test_the_gain_of_an_eso_raw_frame_is_never_trusted() -> None:
    header = _generic_header()
    del header["GAIN"]
    header["HIERARCH ESO DET OUT1 GAIN"] = 2.5
    header["HIERARCH ESO DET OUT1 RON"] = 4.2

    reading = read_header(header)

    assert "eso_raw" in reading.conventions
    assert reading["gain"].status == "ambiguous"
    assert reading["read_noise"].status == "convention"
    assert reading["read_noise"].unit == "electron"

    header["HIERARCH ESO DET OUT1 CONAD"] = 0.4

    assert read_header(header)["gain"].status == "conflicting"


def test_an_extension_inherits_the_primary_header() -> None:
    primary = fits.Header()
    primary["TELESCOP"] = "Example telescope"
    primary["FILTER"] = "i"

    image = fits.Header()
    image["BUNIT"] = "electron/s"
    image["FILTER"] = "z"

    reading = read_header(image, primary_header=primary)

    assert reading["telescope"].value == "Example telescope"
    assert reading["telescope"].source == "primary"
    assert reading["filter"].value == "z"
    assert reading["filter"].source == "image"


def test_a_user_convention_is_read_and_preferred(tmp_path) -> None:
    path = tmp_path / "camera.yaml"
    path.write_text(
        "\n".join(
            [
                "name: example_camera",
                "title: Example camera",
                "priority: 50",
                "references:",
                "  - Example camera manual, section 3.",
                "match:",
                "  - keyword: INSTRUME",
                "    equals: example camera",
                "quantities:",
                "  zeropoint:",
                "    keywords: [MAGZP]",
                "    unit: mag",
                "  gain:",
                "    keywords: [EGAIN]",
                "    unit: electron/adu",
            ]
        ),
        encoding="utf-8",
    )
    header = _generic_header()
    header["MAGZP"] = 28.4
    header["EGAIN"] = 3.1

    conventions = load_header_conventions([path])
    reading = read_header(header, conventions=conventions)

    assert reading.conventions[0] == "example_camera"
    assert reading["zeropoint"].value == pytest.approx(28.4)
    assert reading["zeropoint"].status == "convention"
    assert reading["gain"].keyword == "EGAIN"
    assert reading["gain"].value == pytest.approx(3.1)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("name: generic\ntitle: x\npriority: 1\nreferences: [a]\n"
         "match: always\nquantities:\n  filter:\n    keywords: [F]\n",
         "unique names"),
        ("name: a\ntitle: x\npriority: 1\nreferences: [a]\nmatch: always\n"
         "quantities:\n  colour:\n    keywords: [C]\n",
         "unknown quantity"),
        ("name: a\ntitle: x\npriority: 1\nreferences: []\nmatch: always\n"
         "quantities:\n  filter:\n    keywords: [F]\n",
         "reference"),
        ("name: a\ntitle: x\npriority: 1\nreferences: [a]\nmatch: always\n"
         "quantities:\n  psf_fwhm:\n    keywords: [F]\n    unit: arcsec\n"
         "    unit_from_comment: true\n    units: [arcsec]\n",
         "not both"),
        ("name: a\ntitle: x\npriority: 1\nreferences: [a]\nmatch: always\n"
         "quantities:\n  filter:\n    keywords: [F]\n    colour: red\n",
         "unknown fields"),
        ("name: a\ntitle: x\npriority: 1\nreferences: [a]\n"
         "match:\n  - keyword: X\nquantities:\n  filter:\n    keywords: [F]\n",
         "exactly one"),
    ],
)
def test_malformed_conventions_are_rejected(tmp_path, text, message) -> None:
    path = tmp_path / "convention.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_header_conventions([path])


def test_the_reading_serializes_and_reports() -> None:
    reading = read_header(_coadd_header())

    serialized = json.loads(json.dumps(reading.as_dict()))
    assert serialized["conventions"] == ["vst_tube", "swarp", "generic"]
    assert serialized["quantities"]["gain"]["status"] == "convention"

    report = reading.report()
    assert report.startswith("Conventions: vst_tube, swarp, generic")
    assert "sky_level" in report
    assert "ambiguous" in report


def test_usable_statuses_are_the_certain_ones() -> None:
    assert USABLE_STATUSES == {"explicit", "convention", "derived"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("FWHM [arcsec]", "arcsec"),
        ("Median FWHM in pixels", "pixels"),
        ("Total exposure time [s]", "s"),
        ("arcsec per pixel", None),
        ("FWHM median", None),
        (None, None),
    ],
)
def test_declared_units_are_recognized_only_when_unique(text, expected) -> None:
    assert (
        normalize_declared_unit(text, allowed=("arcsec", "pixels", "s")) == expected
    )


def test_no_header_reads_as_missing() -> None:
    reading = read_header(None)

    assert reading.conventions == ("generic",)
    assert all(item.status == "missing" for item in reading.quantities.values())
