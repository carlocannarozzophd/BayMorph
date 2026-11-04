"""The header engine of the preparation (AD-049, AD-053).

These are the tests of the 0.6 engine, ported with it, and the tests of what
the port fixed: an unusable reading no longer hides the values other
conventions read, a WCS that cannot be used is reported, a repeated keyword
is conflicting, BKGMED and the generic SATURATE are read only where their
unit is established, names agree across spellings, and an image unit that is
a flux density gives the AB zeropoint.
"""

from __future__ import annotations

import json
import math

import pytest
import yaml
from astropy import log as astropy_log
from astropy.io import fits

from baymorph.preparation.header import (
    DERIVED_QUANTITIES,
    QUANTITIES,
    STATUSES,
    UNIT_ZEROPOINT_RULE,
    USABLE_STATUSES,
    Formula,
    canonical_name,
    canonical_unit,
    conventions_digest,
    ensure_linear_pixels,
    load_header_conventions,
    normalize_declared_unit,
    read_celestial_wcs,
    read_header,
    unit_flux_density_jy,
    validate_declarations,
)


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
    assert {
        "astrocut",
        "decam",
        "eso_phase3",
        "eso_raw",
        "coma_treasury",
        "generic",
        "grizli",
        "hscpipe",
        "hst",
        "hst_drizzle",
        "jwst",
        "legacy_survey",
        "pan_starrs",
        "sdss_frame",
        "swarp",
        "vst_tube",
        "xdf",
    } <= set(names)

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

    # The unit of BKGMED is not established, so it is read only as a witness.
    sky = reading["sky_level"]
    assert sky.status == "ambiguous"
    assert sky.convention == "vst_tube"
    assert sky.unit is None
    assert sky.value is None
    assert sky.raw_value == pytest.approx(250.0)
    assert "not established" in sky.note

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
    assert reading.value("psf_fwhm") == pytest.approx(0.85)


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

    # The GAIN keyword of the generic convention says 1.5: the two readings
    # disagree until the convention declares that it overrides the generic one.
    assert reading["gain"].status == "conflicting"
    assert "generic gives 1.5 electron/adu (GAIN)" in reading["gain"].note

    path.write_text(
        path.read_text(encoding="utf-8") + "\n    overrides: [generic]\n",
        encoding="utf-8",
    )
    reading = read_header(header, conventions=load_header_conventions([path]))

    assert reading["gain"].keyword == "EGAIN"
    assert reading["gain"].value == pytest.approx(3.1)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            "name: generic\ntitle: x\npriority: 1\nreferences: [a]\n"
            "match: always\nquantities:\n  filter:\n    keywords: [F]\n",
            "unique names",
        ),
        (
            "name: a\ntitle: x\npriority: 1\nreferences: [a]\nmatch: always\n"
            "quantities:\n  colour:\n    keywords: [C]\n",
            "unknown quantity",
        ),
        (
            "name: a\ntitle: x\npriority: 1\nreferences: []\nmatch: always\n"
            "quantities:\n  filter:\n    keywords: [F]\n",
            "reference",
        ),
        (
            "name: a\ntitle: x\npriority: 1\nreferences: [a]\nmatch: always\n"
            "quantities:\n  psf_fwhm:\n    keywords: [F]\n    unit: arcsec\n"
            "    unit_from_comment: true\n    units: [arcsec]\n",
            "not both",
        ),
        (
            "name: a\ntitle: x\npriority: 1\nreferences: [a]\nmatch: always\n"
            "quantities:\n  filter:\n    keywords: [F]\n    colour: red\n",
            "unknown fields",
        ),
        (
            "name: a\ntitle: x\npriority: 1\nreferences: [a]\n"
            "match:\n  - keyword: X\nquantities:\n  filter:\n    keywords: [F]\n",
            "exactly one",
        ),
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
    assert USABLE_STATUSES == {"declared", "explicit", "convention", "derived"}


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
    assert normalize_declared_unit(text, allowed=("arcsec", "pixels", "s")) == expected


def test_no_header_reads_as_missing() -> None:
    reading = read_header(None)

    assert reading.conventions == ("generic",)
    assert all(item.status == "missing" for item in reading.quantities.values())


# The conventions of the first wave that state values or derive them through
# formulas (AD-049, milestone 2).


def _hst_headers(
    bunit: str = "ELECTRONS/S",
    filename: str = "j8xi01010_drz.fits",
) -> tuple[fits.Header, fits.Header]:
    """Return the primary and SCI headers of a synthetic HST product."""
    primary = fits.Header()
    primary["TELESCOP"] = "HST"
    primary["INSTRUME"] = "ACS"
    primary["FILENAME"] = filename
    primary["EXPTIME"] = 2000.0
    image = fits.Header()
    image["BUNIT"] = bunit
    image["PHOTFLAM"] = 7.0e-20
    image["PHOTPLAM"] = 8045.0
    _wcs_cards(image, 0.05)
    return primary, image


def _ps1_header() -> fits.Header:
    """Return a synthetic header of an asinh-compressed Pan-STARRS stack."""
    header = fits.Header()
    header["BSOFTEN"] = 87.3959
    header["BOFFSET"] = 2.65496
    header["EXPTIME"] = 1260.0
    _wcs_cards(header, 0.25)
    return header


def test_the_hst_zeropoint_is_derived_for_count_rates_of_drizzled_products() -> None:
    primary, image = _hst_headers()

    reading = read_header(image, primary_header=primary)
    zeropoint = reading["zeropoint"]

    assert reading.conventions[0] == "hst"
    assert zeropoint.status == "derived"
    assert zeropoint.value == pytest.approx(
        -2.5 * math.log10(7.0e-20) - 5.0 * math.log10(8045.0) - 2.408,
        rel=1e-12,
    )
    assert zeropoint.unit == "mag"
    assert zeropoint.keyword == "PHOTFLAM, PHOTPLAM"
    assert zeropoint.source == "image"
    assert zeropoint.convention == "hst"
    assert zeropoint.raw_value is None
    assert reading.value("photometric_system") == "AB"


@pytest.mark.parametrize(
    ("bunit", "filename", "observed"),
    [
        ("ELECTRONS", "j8xi01abq_flt.fits", "the image unit is 'ELECTRONS'"),
        ("ELECTRONS/S", "ibxl01abq_flt.fits", "FILENAME = 'ibxl01abq_flt.fits'"),
    ],
)
def test_the_hst_zeropoint_is_ambiguous_outside_the_documented_products(
    bunit, filename, observed
) -> None:
    primary, image = _hst_headers(bunit=bunit, filename=filename)

    reading = read_header(image, primary_header=primary)
    zeropoint = reading["zeropoint"]

    assert zeropoint.status == "ambiguous"
    assert zeropoint.value is None
    assert observed in zeropoint.note
    assert reading["photometric_system"].status == "ambiguous"


def test_the_jwst_zeropoint_agrees_with_the_documented_example() -> None:
    primary = fits.Header()
    primary["TELESCOP"] = "JWST"
    primary["INSTRUME"] = "NIRCAM"
    primary["FILENAME"] = "jw01234-o001_t001_nircam_clear-f200w_i2d.fits"
    image = fits.Header()
    image["BUNIT"] = "MJy/sr"
    image["PIXAR_SR"] = 2.29e-14

    reading = read_header(image, primary_header=primary)
    zeropoint = reading["zeropoint"]

    # The documentation gives ZP_AB = 28.0 for PIXAR_SR = 2.29e-14, through
    # the constant -6.10 that the formula carries unrounded.
    assert zeropoint.status == "derived"
    assert zeropoint.value == pytest.approx(28.0, abs=1e-3)
    assert zeropoint.value == pytest.approx(
        -6.10 - 2.5 * math.log10(2.29e-14), abs=1e-4
    )
    assert reading.value("photometric_system") == "AB"


def test_an_sdss_frame_states_its_zeropoint_and_reads_no_gain() -> None:
    header = fits.Header()
    header["NMGY"] = 0.0048
    header["NMGYIVAR"] = 1.0e6
    header["FILTER"] = "r"
    _wcs_cards(header, 0.396)

    reading = read_header(header)

    assert reading.conventions == ("sdss_frame", "generic")
    assert reading["zeropoint"].status == "convention"
    assert reading["zeropoint"].value == pytest.approx(22.5)
    assert reading["zeropoint"].unit == "mag"
    assert reading["zeropoint"].keyword is None
    assert reading.value("photometric_system") == "AB"
    assert reading["gain"].status == "missing"


def test_a_legacy_survey_coadd_states_its_zeropoint_but_no_system() -> None:
    header = fits.Header()
    header["LEGPIPEV"] = "DR10.1.3"
    header["BUNIT"] = "nanomaggy"
    _wcs_cards(header, 0.262)

    reading = read_header(header)

    assert reading.value("zeropoint") == pytest.approx(22.5)
    assert reading["photometric_system"].status == "missing"
    assert reading.value("image_unit") == "nanomaggy"


@pytest.mark.parametrize(
    ("bunit", "status"),
    [("adu", "convention"), (None, "ambiguous"), ("electron/s", "ambiguous")],
)
def test_the_decam_zeropoint_holds_only_for_an_image_in_adu(bunit, status) -> None:
    primary = fits.Header()
    primary["INSTRUME"] = "DECam"
    primary["MAGZERO"] = 31.6
    image = fits.Header()
    image["SATURATA"] = 44000.0
    image["SATURATB"] = 42000.0

    if bunit is not None:
        image["BUNIT"] = bunit

    reading = read_header(image, primary_header=primary)
    zeropoint = reading["zeropoint"]

    assert zeropoint.status == status
    assert zeropoint.raw_value == pytest.approx(31.6)
    assert zeropoint.source == "primary"
    assert (zeropoint.value is None) == (status == "ambiguous")
    assert reading["saturation"].status == "conflicting"


def test_the_hscpipe_zeropoint_is_derived_from_fluxmag0() -> None:
    primary = fits.Header()
    primary["FLUXMAG0"] = 10.0 ** (0.4 * 27.0)

    reading = read_header(fits.Header(), primary_header=primary)
    zeropoint = reading["zeropoint"]

    assert zeropoint.status == "derived"
    assert zeropoint.value == pytest.approx(27.0, abs=1e-12)
    assert zeropoint.source == "primary"


@pytest.mark.parametrize(
    ("fluxmag0", "note"),
    [(-5.0, "not defined"), ("bright", "not a finite number")],
)
def test_a_formula_undefined_for_the_header_is_ambiguous(fluxmag0, note) -> None:
    header = fits.Header()
    header["FLUXMAG0"] = fluxmag0

    zeropoint = read_header(header)["zeropoint"]

    assert zeropoint.status == "ambiguous"
    assert zeropoint.value is None
    assert note in zeropoint.note


def test_a_pan_starrs_stack_declares_asinh_pixels_and_is_refused() -> None:
    reading = read_header(_ps1_header())
    encoding = reading["pixel_encoding"]

    assert encoding.status == "convention"
    assert encoding.value == "asinh"
    assert encoding.keyword == "BSOFTEN"

    with pytest.raises(ValueError, match="asinh encoding"):
        ensure_linear_pixels(reading, image="the stack")


def test_a_compression_without_boffset_is_not_established_and_refused() -> None:
    header = _ps1_header()
    del header["BOFFSET"]

    reading = read_header(header)

    assert reading["pixel_encoding"].status == "ambiguous"
    assert "the header has no BOFFSET" in reading["pixel_encoding"].note

    with pytest.raises(ValueError, match="does not establish the encoding"):
        ensure_linear_pixels(reading, image="the stack")


def test_an_astrocut_cutout_is_linear_despite_the_compression_keywords() -> None:
    header = _ps1_header()
    header["HISTORY"] = (
        "Pixel values converted from asinh-scaled flux to linear flux using the "
        "BSOFTEN/BOFFSET keywords."
    )

    reading = read_header(header)

    assert reading.conventions[:2] == ("astrocut", "pan_starrs")
    assert reading["pixel_encoding"].status == "convention"
    assert reading.value("pixel_encoding") == "linear"
    ensure_linear_pixels(reading, image="the cutout")


def test_an_image_without_an_encoding_is_taken_as_linear() -> None:
    reading = read_header(_generic_header())

    assert reading["pixel_encoding"].status == "missing"
    ensure_linear_pixels(reading, image="the image")


def test_a_formula_evaluates_only_where_it_is_defined() -> None:
    formula = Formula.parse("2.5 * log10(F) + sqrt(G) - X ** 0.5")

    assert formula.keywords == ("F", "G", "X")
    assert formula.evaluate({"F": 100.0, "G": 4.0, "X": 9.0}) == pytest.approx(4.0)
    assert formula.evaluate({"F": -1.0, "G": 4.0, "X": 9.0}) is None
    assert formula.evaluate({"F": 100.0, "G": 4.0, "X": -9.0}) is None
    assert Formula.parse("1 / X").evaluate({"X": 0.0}) is None
    assert Formula.parse("exp(X)").evaluate({"X": 1.0e6}) is None
    assert Formula.parse("-ln(X)").evaluate({"X": math.e}) == pytest.approx(-1.0)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("__import__('os').system('true')", "may use only"),
        ("PHOTFLAM.real", "may use only"),
        ("log10(PHOTFLAM, 2)", "may use only"),
        ("sin(PHOTFLAM)", "may use only"),
        ("PHOTFLAM if PHOTFLAM else 1", "may use only"),
        ("'a' + PHOTFLAM", "only numbers"),
        ("True + PHOTFLAM", "only numbers"),
        ("log10 * 2", "as a value"),
        ("2.5 * 10", "reads no keyword"),
        ("2.5 *", "not an expression"),
        ("", "non-empty"),
    ],
)
def test_malformed_formulas_are_rejected(text, message) -> None:
    with pytest.raises(ValueError, match=message):
        Formula.parse(text)


def test_a_formula_nested_too_deeply_is_rejected() -> None:
    with pytest.raises(ValueError, match="nested too deeply"):
        Formula.parse("+".join(["1"] * 20000) + "+X")


@pytest.mark.parametrize(
    ("quantity", "spec", "message"),
    [
        (
            "zeropoint",
            {"formula": "2.5 * log10(F)", "keywords": ["F"], "unit": "mag"},
            "neither keywords nor",
        ),
        ("zeropoint", {"formula": "2.5 * log10(F)"}, "needs its unit"),
        ("filter", {"formula": "F"}, "only a number"),
        ("zeropoint", {"formula": "import os", "unit": "mag"}, "not an expression"),
        ("zeropoint", {"value": 22.5}, "needs its unit"),
        ("zeropoint", {"value": "22.5", "unit": "mag"}, "finite number"),
        ("n_combined", {"value": 2.5}, "finite integer"),
        ("photometric_system", {"value": ""}, "non-empty string"),
        ("photometric_system", {"value": "AB", "unit": "mag"}, "takes no unit"),
        (
            "psf_fwhm",
            {"value": 1.0, "unit_from_comment": True, "units": ["arcsec"]},
            "not from the header",
        ),
        (
            "zeropoint",
            {"value": 22.5, "unit": "mag", "requires": {"keyword": "B"}},
            "list of rules",
        ),
        (
            "zeropoint",
            {"value": 22.5, "unit": "mag", "requires": [{"keyword": "B"}]},
            "exactly one",
        ),
    ],
)
def test_malformed_values_formulas_and_requirements_are_rejected(
    tmp_path, quantity, spec, message
) -> None:
    path = tmp_path / "convention.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "a",
                "title": "x",
                "priority": 1,
                "references": ["a"],
                "match": "always",
                "quantities": {quantity: spec},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_header_conventions([path])


def test_a_user_convention_derives_a_quantity_under_its_requirements(
    tmp_path,
) -> None:
    path = tmp_path / "camera.yaml"
    path.write_text(
        "\n".join(
            [
                "name: example_camera",
                "title: Example camera",
                "priority: 50",
                "references:",
                "  - Example camera manual, section 4.",
                "match:",
                "  - keyword: INSTRUME",
                "    equals: example camera",
                "quantities:",
                "  zeropoint:",
                '    formula: "ZPRATE + 2.5 * log10(EXPTIME)"',
                "    unit: mag",
                "    requires:",
                "      - keyword: BUNIT",
                "        equals: ADU",
            ]
        ),
        encoding="utf-8",
    )
    header = _generic_header()
    header["ZPRATE"] = 25.0
    conventions = load_header_conventions([path])

    zeropoint = read_header(header, conventions=conventions)["zeropoint"]

    assert zeropoint.status == "derived"
    assert zeropoint.value == pytest.approx(25.0 + 2.5 * math.log10(300.0))
    assert zeropoint.keyword == "ZPRATE, EXPTIME"

    header["BUNIT"] = "ADU/s"
    zeropoint = read_header(header, conventions=conventions)["zeropoint"]

    assert zeropoint.status == "ambiguous"
    assert "BUNIT is 'ADU'" in zeropoint.note


def _convention_lines(*quantity_lines: str, match: tuple[str, ...] = ()) -> str:
    """Return a user convention that recognizes the example camera."""
    return "\n".join(
        [
            "name: example_camera",
            "title: Example camera",
            "priority: 50",
            "references:",
            "  - Example camera manual, section 4.",
            "match:",
            *(match or ("  - keyword: INSTRUME", "    equals: example camera")),
            "quantities:",
            *quantity_lines,
        ]
    )


def test_a_declared_value_takes_the_place_of_the_header_reading() -> None:
    """The declaration is used, and the reading of the header stays beside it."""
    reading = read_header(
        _coadd_header(),
        declarations={
            "zeropoint": 29.8,
            "photometric_system": "ab",
            "exposure_time": 2100,
            "saturation": 75.0,
        },
    )
    zeropoint = reading["zeropoint"]

    assert zeropoint.status == "declared"
    assert zeropoint.usable
    assert zeropoint.value == 29.8
    assert zeropoint.unit == "mag"
    assert zeropoint.source == "configuration"
    assert zeropoint.header.status == "convention"
    assert zeropoint.header.value == 30.0
    assert reading.value("photometric_system") == "AB"
    assert reading.value("exposure_time") == 2100.0
    assert reading["saturation"].unit == "image"
    assert reading["saturation"].header.status == "ambiguous"
    assert reading.as_dict()["quantities"]["zeropoint"]["header"]["value"] == 30.0
    assert "header" not in reading.as_dict()["quantities"]["gain"]

    report = reading.report()

    assert "declared" in report
    assert "data.calibration" in report


def test_a_declared_zeropoint_sets_aside_the_system_and_error_of_the_header() -> None:
    """The system and the error of the header belong to its own zeropoint."""
    reading = read_header(_coadd_header(), declarations={"zeropoint": 29.8})

    assert reading.value("zeropoint") == 29.8
    assert reading["photometric_system"].status == "missing"
    assert reading["photometric_system"].header.value == "AB"
    assert reading["zeropoint_error"].status == "missing"


def test_a_declared_unit_unlocks_the_hst_zeropoint() -> None:
    """A rate declared for a drizzled image in ELECTRONS meets the convention."""
    primary, image = _hst_headers(bunit="ELECTRONS")

    assert read_header(image, primary_header=primary)["zeropoint"].status == (
        "ambiguous"
    )

    reading = read_header(
        image, primary_header=primary, declarations={"image_unit": "electrons/s"}
    )

    assert reading["image_unit"].status == "declared"
    assert reading["image_unit"].header.value == "ELECTRONS"
    assert reading["zeropoint"].status == "derived"
    assert reading["zeropoint"].value == pytest.approx(
        -2.5 * math.log10(7.0e-20) - 5.0 * math.log10(8045.0) - 2.408,
        rel=1e-12,
    )
    assert reading.value("photometric_system") == "AB"


def test_a_declared_unit_unlike_the_header_sets_aside_values_in_image_units() -> None:
    """Header values in the units of the image no longer hold under another unit."""
    header = _generic_header()
    header["BUNIT"] = "ADU/s"
    header["SOFTNAME"] = "SWarp"
    header["SATURATE"] = 250.0

    assert read_header(header)["saturation"].status == "convention"

    same = read_header(header, declarations={"image_unit": "adu / sec"})
    other = read_header(header, declarations={"image_unit": "ADU"})

    assert same["saturation"].status == "convention"
    assert other["saturation"].status == "ambiguous"
    assert other["saturation"].value is None
    assert other["saturation"].raw_value == 250.0
    assert "declared as 'ADU'" in other["saturation"].note


@pytest.mark.parametrize(
    ("text", "other"),
    [
        ("ELECTRONS/S", "electron/s"),
        ("electrons / sec", "e-/s"),
        ("ADU/s", "adu/second"),
        ("MJy/sr", "mjy / sr"),
    ],
)
def test_units_are_compared_in_one_form(text, other) -> None:
    assert canonical_unit(text) == canonical_unit(other)


@pytest.mark.parametrize(
    ("declarations", "message"),
    [
        ({"seeing": 1.0}, "cannot declare"),
        ({"photometric_system": "AB"}, "declare data.calibration.zeropoint"),
        ({"zeropoint": 25.0, "photometric_system": "ST"}, "one of AB, Vega"),
        ({"zeropoint": True}, "must be a number"),
        ({"zeropoint": float("nan")}, "must be finite"),
        ({"gain": -1.0}, "must be positive"),
        ({"exposure_time": 0}, "must be positive"),
        ({"saturation": "high"}, "must be a number"),
        ({"image_unit": "  "}, "non-empty unit"),
        ("ADU", "must map quantities"),
    ],
)
def test_declarations_are_checked(declarations, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_declarations(declarations)


def test_a_requirement_tests_only_a_quantity_read_before(tmp_path) -> None:
    """A requirement on a later quantity, or a match on a quantity, is refused."""
    path = tmp_path / "camera.yaml"
    path.write_text(
        _convention_lines(
            "  exposure_time:",
            "    keywords: [EXPTIME]",
            "    unit: s",
            "    requires:",
            "      - quantity: zeropoint",
            "        present: true",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="read before"):
        load_header_conventions([path])

    path.write_text(
        _convention_lines(
            "  exposure_time:",
            "    keywords: [EXPTIME]",
            "    unit: s",
            match=("  - quantity: image_unit", "    equals: ADU"),
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="only a requirement"):
        load_header_conventions([path])


# Opening an observation (AD-053, milestone 4): the readings of every matched
# convention are compared, and the drizzled HST images and the grizli mosaics
# have their own conventions.


def _write_convention(path, name: str, priority: int, quantities: list[str]):
    """Write a user convention that matches every header."""
    path.write_text(
        "\n".join(
            [
                f"name: {name}",
                f"title: {name}",
                f"priority: {priority}",
                "references:",
                "  - Example manual.",
                "match: always",
                "quantities:",
                *quantities,
            ]
        ),
        encoding="utf-8",
    )

    return path


def test_conventions_that_disagree_make_a_quantity_conflicting(tmp_path) -> None:
    """Two usable readings that disagree must name both conventions."""
    first = _write_convention(
        tmp_path / "first.yaml",
        "first",
        50,
        ["  zeropoint:", "    keywords: [MAGZP]", "    unit: mag"],
    )
    second = _write_convention(
        tmp_path / "second.yaml",
        "second",
        40,
        ["  zeropoint:", "    keywords: [ZPAB]", "    unit: mag"],
    )
    header = _generic_header()
    header["MAGZP"] = 28.4
    header["ZPAB"] = 28.1

    reading = read_header(header, conventions=load_header_conventions([first, second]))
    zeropoint = reading["zeropoint"]

    assert zeropoint.status == "conflicting"
    assert zeropoint.value is None
    assert zeropoint.raw_value == pytest.approx(28.4)
    assert zeropoint.convention == "first"
    assert "Conventions disagree: first gives 28.4 mag (MAGZP)" in zeropoint.note
    assert "second gives 28.1 mag (ZPAB)" in zeropoint.note
    assert ("ZPAB", 28.1) in zeropoint.candidates

    header["ZPAB"] = 28.4
    agreed = read_header(header, conventions=load_header_conventions([first, second]))

    assert agreed["zeropoint"].status == "convention"
    assert agreed["zeropoint"].convention == "first"
    assert agreed["zeropoint"].note is None


def test_two_zeropoints_within_a_thousandth_of_a_magnitude_agree(tmp_path) -> None:
    """Zeropoints 0.0005 mag apart agree; 0.0015 mag apart they conflict."""
    first = _write_convention(
        tmp_path / "first.yaml",
        "first",
        50,
        ["  zeropoint:", "    keywords: [MAGZP]", "    unit: mag"],
    )
    second = _write_convention(
        tmp_path / "second.yaml",
        "second",
        40,
        ["  zeropoint:", "    keywords: [ZPAB]", "    unit: mag"],
    )
    conventions = load_header_conventions([first, second])
    header = _generic_header()
    header["MAGZP"] = 25.937
    header["ZPAB"] = 25.9365

    zeropoint = read_header(header, conventions=conventions)["zeropoint"]

    assert zeropoint.status == "convention"
    assert zeropoint.value == pytest.approx(25.937)
    assert zeropoint.note == (
        "Another reading agrees within 0.001 mag: second gives 25.9365 mag (ZPAB)."
    )

    # Written 0.001 mag apart, two zeropoints agree whichever way the binary
    # difference rounds.
    for other in (25.936, 25.938):
        header["ZPAB"] = other

        assert read_header(header, conventions=conventions)["zeropoint"].status == (
            "convention"
        )

    header["ZPAB"] = 25.9355

    assert read_header(header, conventions=conventions)["zeropoint"].status == (
        "conflicting"
    )

    # Any other number must still agree exactly.
    exposure = [
        "  exposure_time:",
        "    keywords: [EXPTIME]",
        "    unit: s",
    ]
    _write_convention(tmp_path / "first.yaml", "first", 50, exposure)
    _write_convention(
        tmp_path / "second.yaml",
        "second",
        40,
        ["  exposure_time:", "    keywords: [EXPOSURE]", "    unit: s"],
    )
    header["EXPOSURE"] = 300.0005

    assert (
        read_header(header, conventions=load_header_conventions([first, second]))[
            "exposure_time"
        ].status
        == "conflicting"
    )


def test_hst_headers_give_the_detector_and_the_dates_of_the_observation() -> None:
    """DETECTOR, EXPSTART and EXPEND are read for the calibration services."""
    primary, image = _hst_headers()
    primary["DETECTOR"] = ("WFC", "detector in use: WFC, HRC, or SBC")
    primary["EXPSTART"] = 54122.6577005
    primary["EXPEND"] = 54122.74369596

    reading = read_header(image, primary_header=primary)

    assert reading["detector"].status == "explicit"
    assert reading.value("detector") == "WFC"
    assert reading["detector"].convention == "hst"

    for name, keyword, value in (
        ("observation_start", "EXPSTART", 54122.6577005),
        ("observation_end", "EXPEND", 54122.74369596),
    ):
        quantity = reading[name]

        assert quantity.status == "convention"
        assert quantity.value == pytest.approx(value)
        assert quantity.unit == "MJD"
        assert quantity.keyword == keyword

    generic = read_header(_generic_header())

    for name in ("detector", "observation_start", "observation_end"):
        assert generic[name].status == "missing"


def test_a_convention_may_override_another_and_an_ambiguous_one_decides(
    tmp_path,
) -> None:
    """An override silences a conflict; an ambiguous decision is not bypassed."""
    first = _write_convention(
        tmp_path / "first.yaml",
        "first",
        50,
        [
            "  zeropoint:",
            "    keywords: [MAGZP]",
            "    unit: mag",
            "    overrides: [second]",
        ],
    )
    second = _write_convention(
        tmp_path / "second.yaml",
        "second",
        40,
        ["  zeropoint:", "    keywords: [ZPAB]", "    unit: mag"],
    )
    header = _generic_header()
    header["MAGZP"] = 28.4
    header["ZPAB"] = 28.1

    reading = read_header(header, conventions=load_header_conventions([first, second]))

    assert reading["zeropoint"].status == "convention"
    assert reading.value("zeropoint") == pytest.approx(28.4)
    assert "It overrides another reading: second gives 28.1 mag (ZPAB)." in (
        reading["zeropoint"].note
    )

    # The generic convention reads SATURATE in the units of the image, but a
    # VST-Tube coadd leaves it ambiguous, and that decision stands.
    coadd = read_header(_coadd_header())

    assert coadd["saturation"].status == "ambiguous"
    assert coadd["saturation"].convention == "vst_tube"
    assert "swarp gives 20000 in the units of the image (SATURATE)" in (
        coadd["saturation"].note
    )


def test_a_value_with_an_ignored_prefix_reads_as_blank(tmp_path) -> None:
    """A CLEAR element of a filter wheel holds no filter."""
    primary, image = _hst_headers()
    primary["FILTER1"] = "CLEAR1L"
    primary["FILTER2"] = "F814W"

    reading = read_header(image, primary_header=primary)

    assert reading["filter"].status == "explicit"
    assert reading.value("filter") == "F814W"
    assert reading["filter"].keyword == "FILTER2"
    assert reading["filter"].convention == "hst"

    primary["FILTER1"] = "F606W"
    primary["FILTER2"] = "POL0V"

    assert read_header(image, primary_header=primary)["filter"].status == (
        "conflicting"
    )

    primary["FILTER1"] = "CLEAR1L"
    primary["FILTER2"] = "CLEAR2L"

    assert read_header(image, primary_header=primary)["filter"].status == "missing"

    malformed = _write_convention(
        tmp_path / "malformed.yaml",
        "malformed",
        50,
        ["  filter:", "    value: r", "    ignore_prefixes: [CLEAR]"],
    )

    with pytest.raises(ValueError, match="ignore_prefixes applies only"):
        load_header_conventions([malformed])

    malformed = _write_convention(
        tmp_path / "malformed.yaml",
        "malformed",
        50,
        ["  filter:", "    keywords: [F]", "    overrides: generic"],
    )

    with pytest.raises(ValueError, match="overrides must be a list"):
        load_header_conventions([malformed])


def _drizzled_hst_headers(
    *,
    bunit: str = "ELECTRONS",
    output_units: str = "cps",
    photflam: float = 7.07236e-20,
    filename: str = "j8pu32ucq_flt.fits",
) -> tuple[fits.Header, fits.Header]:
    """Return the headers of an ACS image drizzled by the STScI software."""
    primary, image = _hst_headers(bunit=bunit, filename=filename)
    primary["FILTER1"] = "CLEAR1L"
    primary["FILTER2"] = "F814W"
    primary["D001DATA"] = "j8pu32ucq_flt.fits[sci,1]"
    primary["D001OUUN"] = (
        output_units,
        "Drizzle, units of output image - counts or cps",
    )
    image["PHOTFLAM"] = photflam
    image["PHOTPLAM"] = 8059.7598

    return primary, image


def test_an_hst_image_drizzled_to_count_rates_is_in_electrons_per_second() -> None:
    """D001OUUN = cps over inputs in electrons gives electrons per second."""
    primary, image = _drizzled_hst_headers()

    reading = read_header(image, primary_header=primary)

    assert reading.conventions[:2] == ("hst_drizzle", "hst")
    assert reading["image_unit"].status == "convention"
    assert reading.value("image_unit") == "electron/s"
    assert reading["image_unit"].convention == "hst_drizzle"
    assert reading["zeropoint"].status == "derived"
    assert reading["zeropoint"].convention == "hst_drizzle"
    assert reading.value("zeropoint") == pytest.approx(
        -2.5 * math.log10(7.07236e-20) - 5.0 * math.log10(8059.7598) - 2.408,
        rel=1e-12,
    )
    assert reading.value("zeropoint") == pytest.approx(25.9365, abs=1e-4)
    assert reading.value("photometric_system") == "AB"
    assert reading.value("filter") == "F814W"

    # A product whose BUNIT already says ELECTRONS/S agrees with the rule.
    primary, image = _drizzled_hst_headers(
        bunit="ELECTRONS/S", filename="j8pu32010_drz.fits"
    )
    reading = read_header(image, primary_header=primary)

    assert reading["image_unit"].status == "convention"
    assert reading["zeropoint"].status == "derived"


def test_a_photflam_of_zero_leaves_the_drizzled_zeropoint_to_declare() -> None:
    primary, image = _drizzled_hst_headers(photflam=0.0, filename="j9ty22010_drz.fits")
    primary["PHOTCORR"] = "OMIT"

    reading = read_header(image, primary_header=primary)
    zeropoint = reading["zeropoint"]

    assert reading.value("image_unit") == "electron/s"
    assert zeropoint.status == "ambiguous"
    assert zeropoint.convention == "hst_drizzle"
    assert "PHOTCORR = OMIT" in zeropoint.note
    assert "not defined for these values" in zeropoint.note
    assert reading["photometric_system"].status == "ambiguous"

    declared = read_header(
        image,
        primary_header=primary,
        declarations={"zeropoint": 25.94, "photometric_system": "AB"},
    )

    assert declared["zeropoint"].status == "declared"
    assert declared["zeropoint"].header.convention == "hst_drizzle"


@pytest.mark.parametrize(
    ("bunit", "output_units", "status", "value"),
    [
        # Counts that are not electrons may be DN: the unit stays open.
        ("COUNTS", "cps", "ambiguous", None),
        # An image drizzled to counts keeps the unit of its header.
        ("ELECTRONS", "counts", "explicit", "ELECTRONS"),
    ],
)
def test_the_drizzle_rule_holds_only_for_electrons_drizzled_to_rates(
    bunit, output_units, status, value
) -> None:
    primary, image = _drizzled_hst_headers(bunit=bunit, output_units=output_units)

    reading = read_header(image, primary_header=primary)

    assert reading["image_unit"].status == status
    assert reading.value("image_unit") == value
    assert reading["zeropoint"].status == "ambiguous"


def _grizli_header() -> fits.Header:
    """Return the header of a DJA mosaic made by grizli, in 10 nJy."""
    header = fits.Header()
    header["TELESCOP"] = "JWST"
    header["INSTRUME"] = "NIRCAM"
    header["FILTER"] = "F277W"
    header["GRIZLIV"] = ("1.9.13.dev26+g7a14934", "Grizli code version")
    header["PHOTFNU"] = 1e-08
    header["OPHOTFNU"] = (9.31e-08, "Original PHOTFNU before scaling")
    header["ZP"] = 26.47788770106852
    header["PIXAR_SR"] = 9.31e-14
    header["BUNIT"] = "10.0*nanoJansky"
    _wcs_cards(header, 0.04)

    return header


def test_a_grizli_mosaic_takes_its_zeropoint_from_photfnu() -> None:
    """10 nJy per unit of the pixels is the AB zeropoint 28.9 of the DJA."""
    reading = read_header(_grizli_header())
    zeropoint = reading["zeropoint"]

    assert reading.conventions[:2] == ("grizli", "jwst")
    assert zeropoint.status == "derived"
    assert zeropoint.convention == "grizli"
    assert zeropoint.keyword == "PHOTFNU"
    assert zeropoint.value == pytest.approx(-2.5 * math.log10(1e-8 / 3631.0))
    assert zeropoint.value == pytest.approx(28.9, abs=1e-3)
    assert "not the pixels of the mosaic" in zeropoint.note
    assert reading.value("photometric_system") == "AB"
    assert reading.value("pipeline") == "grizli"
    assert reading.value("image_unit") == "10.0*nanoJansky"


def test_an_exposure_time_written_as_text_is_read_as_a_number() -> None:
    """SDSS frames write EXPTIME as a string; its number is still read."""
    header = fits.Header()
    header["NMGY"] = 0.0048
    header["EXPTIME"] = ("53.907456", "Exposure time (seconds)")

    reading = read_header(header)

    assert reading["exposure_time"].status == "convention"
    assert reading.value("exposure_time") == pytest.approx(53.907456)
    assert reading["exposure_time"].unit == "s"


# Values tabulated by a quantity read before, and the zeropoints of two High
# Level Science Products that document them per filter.


def test_a_convention_tabulates_a_quantity_by_one_read_before(tmp_path) -> None:
    """A table keyed by the filter gives the value of the filter it finds."""
    path = _write_convention(
        tmp_path / "survey.yaml",
        "survey",
        50,
        [
            "  zeropoint:",
            "    by: filter",
            "    table:",
            "      r: 27.1",
            "      I: 26.4",
            "    unit: mag",
        ],
    )
    conventions = load_header_conventions([path])
    header = _generic_header()

    reading = read_header(header, conventions=conventions)
    zeropoint = reading["zeropoint"]

    assert zeropoint.status == "convention"
    assert zeropoint.value == pytest.approx(27.1)
    assert zeropoint.unit == "mag"
    assert zeropoint.keyword == "FILTER"
    assert zeropoint.convention == "survey"
    assert "Tabulated for the filter 'r'." in zeropoint.note

    header["FILTER"] = " i "

    assert read_header(header, conventions=conventions).value("zeropoint") == (
        pytest.approx(26.4)
    )

    header["FILTER"] = "z"
    unlisted = read_header(header, conventions=conventions)["zeropoint"]

    assert unlisted.status == "ambiguous"
    assert "tabulates no value for the filter 'z'" in unlisted.note

    del header["FILTER"]
    unknown = read_header(header, conventions=conventions)["zeropoint"]

    assert unknown.status == "ambiguous"
    assert "tabulates it by the filter, which is not known" in unknown.note


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (["    by: filter", "    unit: mag"], "needs both by and table"),
        (
            ["    by: psf_fwhm", "    table: {r: 27.1}", "    unit: mag"],
            "keyed by a text quantity read before",
        ),
        (
            ["    by: exposure_time", "    table: {'300': 27.1}", "    unit: mag"],
            "keyed by a text quantity read before",
        ),
        (
            ["    by: filter", "    table: {on: 27.1}", "    unit: mag"],
            "quote the key True",
        ),
        (["    by: filter", "    table: {}", "    unit: mag"], "must map values"),
        (
            ["    by: filter", "    table: {r: 27.1, R: 27.2}", "    unit: mag"],
            "distinct and non-empty",
        ),
        (["    by: filter", "    table: {r: 27.1}"], "needs its unit"),
        (
            ["    by: filter", "    table: {r: 27.1}", "    keywords: [ZP]"],
            "neither keywords",
        ),
    ],
)
def test_malformed_tables_are_rejected(tmp_path, lines, message) -> None:
    path = _write_convention(
        tmp_path / "survey.yaml", "survey", 50, ["  zeropoint:", *lines]
    )

    with pytest.raises(ValueError, match=message):
        load_header_conventions([path])


def test_a_table_keyed_by_the_image_unit_compares_units_in_one_form(
    tmp_path,
) -> None:
    """A key spelled as BUNIT matches the unit a convention or a user gives."""
    path = _write_convention(
        tmp_path / "camera.yaml",
        "camera",
        70,
        [
            "  saturation:",
            "    by: image_unit",
            "    table:",
            "      ELECTRONS/S: 50.0",
            "    unit: image",
        ],
    )
    conventions = [*load_header_conventions(), *load_header_conventions([path])]
    primary, image = _drizzled_hst_headers()

    reading = read_header(image, primary_header=primary, conventions=conventions)

    assert reading.value("image_unit") == "electron/s"
    assert reading.value("saturation") == pytest.approx(50.0)

    # Without a BUNIT, the declared unit contradicts no unit of the header.
    header = _generic_header()
    del header["BUNIT"]
    declared = read_header(
        header, conventions=conventions, declarations={"image_unit": "e-/s"}
    )

    assert declared.value("saturation") == pytest.approx(50.0)


def _xdf_header(filter_name: str = "F850LP") -> fits.Header:
    """Return the header of an XDF mosaic, which records XDFVER."""
    header = fits.Header()
    header["TELESCOP"] = "HST"
    header["INSTRUME"] = "ACS"
    header["DETECTOR"] = "WFC"
    header["FILTER"] = filter_name
    header["BUNIT"] = "ELECTRONS/S"
    header["EXPTIME"] = 1.0
    header["XDFVER"] = "Hubble eXtreme Deep Field (ACS/WFC) V1.0. 22 March 2013"
    _wcs_cards(header, 0.03)

    return header


def test_an_xdf_mosaic_takes_the_zeropoint_of_its_release() -> None:
    """The XDF documents one AB zeropoint per filter; its header has none."""
    reading = read_header(_xdf_header())
    zeropoint = reading["zeropoint"]

    assert reading.conventions[:2] == ("xdf", "hst")
    assert zeropoint.status == "convention"
    assert zeropoint.value == pytest.approx(24.87)
    assert zeropoint.convention == "xdf"
    assert zeropoint.keyword == "FILTER"
    assert reading.value("photometric_system") == "AB"
    assert read_header(_xdf_header("F160W")).value("zeropoint") == pytest.approx(25.94)

    header = _xdf_header()
    header["BUNIT"] = "ELECTRONS"

    assert read_header(header)["zeropoint"].status == "ambiguous"


def _coma_headers(photflam: float = 0.0) -> tuple[fits.Header, fits.Header]:
    """Return the headers of an image of the Coma Treasury Survey (DR2)."""
    primary, image = _drizzled_hst_headers(
        photflam=photflam, filename="h_v22_F814W_ivm_drz_cl_ver2.fits"
    )
    primary["PROPOSID"] = 10861
    primary["PHOTCORR"] = "OMIT"

    return primary, image


def test_a_coma_treasury_image_takes_the_zeropoint_of_the_survey() -> None:
    """PHOTFLAM = 0 in the header; the survey documents 25.937 for F814W."""
    primary, image = _coma_headers()

    reading = read_header(image, primary_header=primary)
    zeropoint = reading["zeropoint"]

    assert reading.conventions[:2] == ("coma_treasury", "hst_drizzle")
    assert reading.value("image_unit") == "electron/s"
    assert zeropoint.status == "convention"
    assert zeropoint.value == pytest.approx(25.937)
    assert zeropoint.convention == "coma_treasury"
    assert zeropoint.keyword == "FILTER2"
    assert reading.value("photometric_system") == "AB"

    # A valid PHOTFLAM gives 25.9365, within 0.001 mag of the survey value,
    # which is kept, and the note names the other readings.
    primary, image = _coma_headers(photflam=7.0723600e-20)
    reading = read_header(image, primary_header=primary)

    assert reading.value("zeropoint") == pytest.approx(25.937)
    assert (
        "Other readings agree within 0.001 mag: hst_drizzle gives 25.9365 mag "
        "(PHOTFLAM, PHOTPLAM); hst gives 25.9365 mag (PHOTFLAM, PHOTPLAM)."
    ) in reading["zeropoint"].note

    # A PHOTFLAM whose zeropoint is 0.0015 mag lower, 0.002 mag from the
    # survey value, is overridden by it.
    primary, image = _coma_headers(photflam=7.0723600e-20 * 10 ** (0.4 * 0.0015))
    reading = read_header(image, primary_header=primary)

    assert reading.value("zeropoint") == pytest.approx(25.937)
    assert "It overrides another reading: hst_drizzle gives" in (
        reading["zeropoint"].note
    )

    # The file name alone is not enough: the program must be the survey's.
    primary, image = _coma_headers()
    primary["PROPOSID"] = 12345

    other = read_header(image, primary_header=primary)["zeropoint"]

    assert other.status == "ambiguous"
    assert other.convention == "coma_treasury"
    assert "PROPOSID" in other.note


# What the port fixed: defects that a review of the 0.6 opening against known
# answers found.


def _wings_vst_header() -> fits.Header:
    """Return the header of a WINGS VST coadd normalized to one second."""
    header = fits.Header()
    header["TELESCOP"] = "ESO-VST"
    header["FILTER"] = ("VJOHN", "Filter Name.")
    header["EXPTIME"] = (1.000000024214387, "[sec] Exposure time.")
    header["GAIN"] = (3224.999921908604, "[e-/ADU] Gain.")
    header["HIERARCH ESO DET OUT1 CONAD"] = (2.08, "Conversion from ADUs to electrons")
    header["HIERARCH ESO DET OUT1 GAIN"] = (0.48, "Conversion from electrons to ADU")
    header["HIERARCH ESO DPR CATG"] = ("SCIENCE", "Observation category")
    header["HIERARCH ESO INS FILT1 NAME"] = ("V_JOHN", "Filter name.")
    _wcs_cards(header, 0.2134)
    return header


def test_an_unusable_decision_names_the_values_other_conventions_read() -> None:
    """The ESO detector keywords decide the gain, but the GAIN of the coadd shows."""
    reading = read_header(_wings_vst_header())
    gain = reading["gain"]

    assert reading.conventions == ("eso_raw", "generic")
    assert gain.status == "conflicting"
    assert gain.value is None
    assert gain.convention == "eso_raw"
    assert "generic gives 3225 electron/adu (GAIN)" in gain.note
    assert "Declare data.calibration.gain to use a value." in gain.note
    assert any(
        keyword == "GAIN" and value == pytest.approx(3225.0, rel=1e-6)
        for keyword, value in gain.candidates
    )
    assert "generic gives 3225 electron/adu (GAIN)" in reading.report()


def test_an_unusable_decision_of_a_name_lists_the_others_without_a_remedy(
    tmp_path,
) -> None:
    """A quantity that data.calibration cannot declare gets no remedy."""
    path = _write_convention(
        tmp_path / "camera.yaml",
        "camera",
        50,
        [
            "  filter:",
            "    keywords: [FILTNAME]",
            "    requires:",
            "      - keyword: FILTWHL",
            "        present: true",
        ],
    )
    header = _generic_header()
    header["FILTNAME"] = "R"

    reading = read_header(header, conventions=load_header_conventions([path]))
    filter_ = reading["filter"]

    assert filter_.status == "ambiguous"
    assert filter_.convention == "camera"
    assert "another convention reads it: generic gives r (FILTER)." in filter_.note
    assert "Declare" not in filter_.note


def test_names_that_differ_only_in_spelling_agree() -> None:
    """V_JOHN and VJOHN are one filter, written by two conventions."""
    filter_ = read_header(_wings_vst_header())["filter"]

    assert filter_.status == "explicit"
    assert filter_.value == "V_JOHN"
    assert filter_.convention == "eso_raw"
    assert (
        "Another reading gives the same name with another spelling: generic gives "
        "VJOHN (FILTER)."
    ) in filter_.note

    assert canonical_name(" V_john ") == canonical_name("v-JOHN") == "vjohn"
    assert canonical_name("F814W") != canonical_name("F606W")


def test_keywords_of_one_convention_that_spell_one_name_differently_agree(
    tmp_path,
) -> None:
    path = _write_convention(
        tmp_path / "camera.yaml",
        "camera",
        50,
        ["  filter:", "    keywords: [FILT_A, FILT_B]"],
    )
    conventions = load_header_conventions([path])
    header = _generic_header()
    del header["FILTER"]
    header["FILT_A"] = "R_SDSS"
    header["FILT_B"] = "r-sdss"

    filter_ = read_header(header, conventions=conventions)["filter"]

    assert filter_.status == "explicit"
    assert filter_.value == "R_SDSS"
    assert "The keywords write the same name differently" in filter_.note

    header["FILT_B"] = "g"

    assert read_header(header, conventions=conventions)["filter"].status == (
        "conflicting"
    )


def test_a_celestial_wcs_that_astropy_cannot_build_is_reported() -> None:
    """The INT/WFC headers of WINGS write CUNIT1 = 'pixel' on a TAN projection."""
    header = _generic_header()
    header["CUNIT1"] = "pixel"

    reading = read_header(header)
    scale = reading["pixel_scale"]

    assert scale.status == "ambiguous"
    assert scale.value is None
    assert reading.celestial_wcs is None
    assert "CTYPE1 = 'RA---TAN'" in scale.note
    assert "Astropy cannot read it" in scale.note
    assert "CUNIT1" in scale.note
    assert "Declare data.pixel_scale" in scale.note
    assert reading.wcs_problem == scale.note
    assert reading.as_dict()["wcs_problem"] == scale.note


def _tangent_header_without_scale(**cards) -> fits.Header:
    """Return a TAN header with the given scale cards only."""
    header = fits.Header()
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRPIX1"] = 50.0
    header["CRPIX2"] = 50.0
    header["CRVAL1"] = 150.0
    header["CRVAL2"] = 2.0

    for keyword, value in cards.items():
        header[keyword] = value

    return header


@pytest.mark.parametrize(
    ("cards", "status"),
    [
        ({}, "ambiguous"),
        ({"PC1_1": 1.0, "PC2_2": 1.0}, "ambiguous"),
        ({"CDELT1": -5.5e-5}, "ambiguous"),
        ({"CDELT1": -5.5e-5, "CDELT2": 5.5e-5}, "derived"),
        ({"PC1_1": 1.0, "PC2_2": 1.0, "CDELT1": -5.5e-5, "CDELT2": 5.5e-5}, "derived"),
        ({"CD1_1": -5.5e-5, "CD2_2": 5.5e-5}, "derived"),
    ],
)
def test_a_celestial_wcs_without_its_scale_gives_no_pixel_scale(cards, status) -> None:
    """Without CDi_j or both CDELTi the standard takes one degree per pixel."""
    reading = read_header(_tangent_header_without_scale(**cards))
    scale = reading["pixel_scale"]

    assert scale.status == status

    if status == "derived":
        assert scale.value == pytest.approx(0.198)
        assert reading.wcs_problem is None
    else:
        assert scale.value is None
        assert reading.celestial_wcs is None
        assert "neither CDi_j nor CDELT1 and CDELT2" in scale.note


def test_a_header_without_a_celestial_wcs_has_no_pixel_scale_and_no_problem() -> None:
    header = fits.Header()
    header["BUNIT"] = "ADU"

    reading = read_header(header)

    assert reading["pixel_scale"].status == "missing"
    assert reading.wcs_problem is None
    assert read_celestial_wcs(header) == (None, None)

    header["CTYPE1"] = "LINEAR"
    header["CTYPE2"] = "LINEAR"

    assert read_celestial_wcs(header) == (None, None)


def test_a_keyword_written_twice_with_different_values_is_conflicting() -> None:
    """A pipeline that appends a corrected EXPTIME leaves the stale one first."""
    header = _generic_header()
    header.append(("EXPTIME", 200.0, "corrected exposure time"))

    exposure = read_header(header)["exposure_time"]

    assert exposure.status == "conflicting"
    assert exposure.value is None
    assert exposure.raw_value == 300.0
    assert "writes EXPTIME more than once, with the values 300.0, 200.0" in (
        exposure.note
    )
    assert ("EXPTIME", 200.0) in exposure.candidates

    header = _generic_header()
    header.append(("EXPTIME", 300.0, "the same value again"))

    assert read_header(header)["exposure_time"].status == "convention"


def test_repeated_commentary_cards_are_not_a_conflict() -> None:
    header = _ps1_header()
    header.add_history(
        "Pixel values converted from asinh-scaled flux to linear flux using the "
        "BSOFTEN/BOFFSET keywords."
    )
    header.add_history("A later step of the cutout.")

    assert read_header(header).value("pixel_encoding") == "linear"


@pytest.mark.parametrize(
    ("bunit", "status"),
    [
        (None, "ambiguous"),
        ("ADU", "convention"),
        ("ELECTRONS", "convention"),
        ("ADU/s", "ambiguous"),
        ("e-/s", "ambiguous"),
        ("nanomaggy", "ambiguous"),
    ],
)
def test_the_generic_saturation_holds_only_for_an_image_in_counts(
    bunit, status
) -> None:
    """A coadd normalized to one second keeps the SATURATE of one exposure."""
    header = fits.Header()
    header["SATURATE"] = (65535.0, "Highest value that is unsaturated")
    header["EXPTIME"] = 1
    header["NCOMBINE"] = 12

    if bunit is not None:
        header["BUNIT"] = bunit

    saturation = read_header(header)["saturation"]

    assert saturation.status == status

    if status == "convention":
        assert saturation.value == 65535.0
        assert saturation.unit == "image"
    else:
        assert saturation.value is None
        assert saturation.raw_value == 65535.0
        assert "the image unit is one of 'ADU'" in saturation.note


@pytest.mark.parametrize(
    ("rule", "message"),
    [
        ({"quantity": "image_unit", "one_of": []}, "non-empty list"),
        ({"quantity": "image_unit", "one_of": "ADU"}, "non-empty list"),
        (
            {"quantity": "image_unit", "one_of": ["ADU"], "equals": "ADU"},
            "exactly one",
        ),
    ],
)
def test_malformed_one_of_rules_are_rejected(tmp_path, rule, message) -> None:
    path = tmp_path / "convention.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "a",
                "title": "x",
                "priority": 1,
                "references": ["a"],
                "match": "always",
                "quantities": {
                    "saturation": {
                        "keywords": ["SATURATE"],
                        "unit": "image",
                        "requires": [rule],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_header_conventions([path])


def _dja_cutout_header() -> fits.Header:
    """Return the header of a user cutout of a DJA mosaic, without GRIZLIV."""
    header = fits.Header()
    header["TELESCOP"] = "JWST"
    header["INSTRUME"] = "NIRCAM"
    header["FILTER"] = "F444W"
    header["PHOTFNU"] = 1e-08
    header["ZP"] = 26.51773877645794
    header["BUNIT"] = "10.0*nanoJansky"
    _wcs_cards(header, 0.04)

    return header


def test_an_image_in_a_flux_density_takes_its_ab_zeropoint_from_the_unit() -> None:
    """10 nJy per unit of the pixels is the AB zeropoint 28.9, for any telescope."""
    reading = read_header(_dja_cutout_header())
    zeropoint = reading["zeropoint"]

    assert reading.conventions == ("jwst", "generic")
    assert zeropoint.status == "derived"
    assert zeropoint.convention == UNIT_ZEROPOINT_RULE
    assert zeropoint.keyword == "BUNIT"
    assert zeropoint.unit == "mag"
    assert zeropoint.value == pytest.approx(-2.5 * math.log10(1e-8 / 3631.0), rel=1e-12)
    assert zeropoint.value == pytest.approx(28.9, abs=1e-3)
    assert "PHOTFNU = 1e-08 Jy gives the same zeropoint." in zeropoint.note

    system = reading["photometric_system"]

    assert system.status == "derived"
    assert system.value == "AB"
    assert system.convention == UNIT_ZEROPOINT_RULE
    assert "It replaces the reading of jwst" in system.note

    # With PIXAR_SR the jwst convention finds the zeropoint and leaves it
    # ambiguous, since the image is not in MJy/sr; the unit settles it.
    header = _dja_cutout_header()
    header["PIXAR_SR"] = 9.31e-14
    zeropoint = read_header(header)["zeropoint"]

    assert zeropoint.status == "derived"
    assert zeropoint.value == pytest.approx(28.9, abs=1e-3)
    assert (
        "It replaces the reading of jwst, which leaves the zeropoint ambiguous: "
    ) in zeropoint.note
    assert "the image unit is 'MJy/sr'" in zeropoint.note


def test_a_photfnu_unlike_the_unit_makes_the_zeropoint_conflicting() -> None:
    header = _dja_cutout_header()
    header["PHOTFNU"] = 2e-08

    zeropoint = read_header(header)["zeropoint"]

    assert zeropoint.status == "conflicting"
    assert zeropoint.value is None
    assert zeropoint.raw_value == pytest.approx(28.9, abs=1e-3)
    assert "PHOTFNU = 2e-08 gives" in zeropoint.note
    assert "the two must agree" in zeropoint.note

    header["PHOTFNU"] = "bright"

    assert read_header(header)["zeropoint"].status == "conflicting"

    del header["PHOTFNU"]
    zeropoint = read_header(header)["zeropoint"]

    assert zeropoint.status == "derived"
    assert "PHOTFNU" not in zeropoint.note


def test_a_convention_zeropoint_must_agree_with_a_flux_density_unit(
    tmp_path,
) -> None:
    path = _write_convention(
        tmp_path / "survey.yaml",
        "survey",
        50,
        ["  zeropoint:", "    keywords: [MAGZP]", "    unit: mag"],
    )
    conventions = load_header_conventions([path])
    header = _generic_header()
    header["BUNIT"] = "nJy"
    header["MAGZP"] = 31.4

    agreed = read_header(header, conventions=conventions)["zeropoint"]

    assert agreed.status == "convention"
    assert agreed.convention == "survey"
    assert agreed.value == pytest.approx(31.4)
    assert "The image unit gives the same zeropoint" in agreed.note

    header["MAGZP"] = 30.0
    disagreed = read_header(header, conventions=conventions)["zeropoint"]

    assert disagreed.status == "conflicting"
    assert disagreed.value is None
    assert "The image unit disagrees: survey gives 30 mag (MAGZP)" in disagreed.note
    assert f"{UNIT_ZEROPOINT_RULE} gives 31.4001 mag (BUNIT)" in disagreed.note


def test_a_declared_flux_density_unit_gives_the_zeropoint() -> None:
    header = _generic_header()
    del header["BUNIT"]

    reading = read_header(header, declarations={"image_unit": "uJy"})

    assert reading["zeropoint"].status == "derived"
    assert reading.value("zeropoint") == pytest.approx(23.9, abs=1e-3)
    assert reading["zeropoint"].source == "configuration"
    assert reading["zeropoint"].note.startswith("The declared image unit 'uJy'")
    assert reading.value("photometric_system") == "AB"


def test_a_declared_zeropoint_takes_the_place_of_the_unit_zeropoint() -> None:
    reading = read_header(_dja_cutout_header(), declarations={"zeropoint": 28.0})

    assert reading["zeropoint"].status == "declared"
    assert reading.value("zeropoint") == 28.0
    assert reading["zeropoint"].header.convention == UNIT_ZEROPOINT_RULE
    assert reading["photometric_system"].status == "missing"


@pytest.mark.parametrize(
    ("unit", "flux"),
    [
        ("10.0*nanoJansky", 1e-8),
        ("nJy", 1e-9),
        ("uJy", 1e-6),
        ("nJy/pix", 1e-9),
        ("erg / (s cm2 Hz)", 1e23),
        ("Jy", 1.0),
        ("MJy/sr", None),
        ("ADU/s", None),
        ("ELECTRONS/S", None),
        ("electron/s", None),
        ("nanomaggy", None),
        ("Jy/beam", None),
        ("", None),
        ("not a unit", None),
    ],
)
def test_only_a_flux_density_per_pixel_is_read_as_one(unit, flux) -> None:
    result = unit_flux_density_jy(unit)

    if flux is None:
        assert result is None
    else:
        assert result == pytest.approx(flux, rel=1e-12)


def test_the_conventions_have_a_digest_that_follows_their_content(tmp_path) -> None:
    distributed = load_header_conventions()

    assert all(len(convention.digest) == 64 for convention in distributed)
    assert conventions_digest(distributed) == conventions_digest(
        list(reversed(distributed))
    )

    path = _write_convention(
        tmp_path / "camera.yaml",
        "camera",
        50,
        ["  zeropoint:", "    keywords: [MAGZP]", "    unit: mag"],
    )
    first = load_header_conventions([path])
    _write_convention(
        path,
        "camera",
        50,
        ["  zeropoint:", "    keywords: [MAGZP]", "    unit: mag", "    note: x"],
    )
    second = load_header_conventions([path])

    assert conventions_digest(first) != conventions_digest(second)
    assert conventions_digest(first) != conventions_digest(distributed)


def test_the_distributed_conventions_are_those_of_the_preparation() -> None:
    sources = {convention.source for convention in load_header_conventions()}

    assert all(
        source.startswith("baymorph/resources/preparation/header_conventions/")
        for source in sources
    )


def _sip_header(ctype_suffix: str) -> fits.Header:
    """Return a header with SIP terms of order 2 and the given CTYPE suffix."""
    header = fits.Header()
    _wcs_cards(header, 0.1)
    header["CTYPE1"] = "RA---TAN" + ctype_suffix
    header["CTYPE2"] = "DEC--TAN" + ctype_suffix
    header["A_ORDER"] = 2
    header["B_ORDER"] = 2
    header["A_2_0"] = 1.0e-6
    header["B_0_2"] = 1.0e-6

    return header


def test_sip_terms_without_their_suffix_get_a_note_instead_of_astropy_info() -> None:
    with astropy_log.log_to_list() as messages:
        reading = read_header(_sip_header(""))

    assert not any("SIP" in str(message.msg) for message in messages)
    assert "SIP distortion terms (A_ORDER = 2)" in reading["pixel_scale"].note
    assert "BayMorph leaves them out" in reading["pixel_scale"].note
    assert "Notes" in reading.report()


def test_sip_terms_with_their_suffix_or_no_sip_get_no_note() -> None:
    for header in (_sip_header("-SIP"), _generic_header()):
        with astropy_log.log_to_list() as messages:
            reading = read_header(header)

        assert not any("SIP" in str(message.msg) for message in messages)
        assert reading["pixel_scale"].status == "derived"
        assert "SIP" not in reading["pixel_scale"].note
