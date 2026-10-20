"""Tests for the calibration services that the observation advisor consults."""

from __future__ import annotations

import ast
import email.message
import http.client
import io
import json
import ssl
import tokenize
import urllib.error
import urllib.request
import urllib.response
from datetime import date
from pathlib import Path

import pytest
from astropy.io import fits

from baymorph.core.header_metadata import read_header
from baymorph.pipeline import calibration_bridges
from baymorph.pipeline.calibration_bridges import (
    ACS_ENDPOINT,
    BRIDGES,
    MAX_ANSWER_DEPTH,
    AcsZeropointBridge,
    BridgeAnswerError,
    BridgeOutcome,
    BridgeTransportError,
    _calendar_date,
    _check_bridges,
    bridged_declarations,
    consult_bridges,
)
from baymorph.pipeline.calibration_bridges import post_json as real_post_json

PACKAGE = Path(__file__).resolve().parents[1] / "baymorph"
DECLARE = (
    "The calculator was not asked; declare zeropoint and photometric_system "
    "in data.calibration (--zeropoint, --photometric-system)."
)


def _acs_headers(**changes) -> tuple[fits.Header, fits.Header]:
    """Return the headers of an ACS/WFC F814W image drizzled to cps.

    The dates and exposure time are those of a Coma Treasury Survey image;
    ``PHOTFLAM`` is 0, so the header gives no zeropoint. A change set to None
    removes the keyword.
    """
    primary = fits.Header()
    primary["TELESCOP"] = "HST"
    primary["INSTRUME"] = "ACS"
    primary["DETECTOR"] = "WFC"
    primary["FILTER1"] = "CLEAR1L"
    primary["FILTER2"] = "F814W"
    primary["FILENAME"] = "j9ty22010_drz.fits"
    primary["EXPSTART"] = 54122.6577005
    primary["EXPEND"] = 54122.74369596
    primary["EXPTIME"] = 1400.0
    image = fits.Header()
    image["BUNIT"] = "ELECTRONS"
    image["D001OUUN"] = "cps"
    image["PHOTFLAM"] = 0.0
    image["PHOTPLAM"] = 8045.0

    for keyword, value in changes.items():
        header = image if keyword in image else primary

        if value is None:
            header.remove(keyword, ignore_missing=True)
        else:
            header[keyword] = value

    return primary, image


def _reading(declarations=None, **changes):
    """Return the reading of the headers of :func:`_acs_headers`."""
    primary, image = _acs_headers(**changes)

    return read_header(image, primary_header=primary, declarations=declarations)


def test_the_acs_bridge_serves_only_hst_acs_readings() -> None:
    """Images of other instruments and telescopes are not served."""
    bridge = AcsZeropointBridge()

    assert bridge.serves(_reading())
    assert not bridge.serves(_reading(INSTRUME="WFC3"))
    assert not bridge.serves(_reading(TELESCOP="JWST"))
    assert not bridge.serves(_reading(TELESCOP=None))
    assert consult_bridges(_reading(INSTRUME="WFC3"), online=True) == ()


def test_a_usable_or_declared_zeropoint_needs_no_service(acs_calculator) -> None:
    """A zeropoint read from the header, or declared, is never asked for."""
    (derived,) = consult_bridges(_reading(PHOTFLAM=7.07236e-20), online=True)

    assert derived.status == "not_needed"
    assert derived.values == {}
    assert derived.detail.startswith("The header reading gives the zeropoint 25.9")
    assert "(derived, PHOTFLAM, PHOTPLAM; hst_drizzle)" in derived.detail
    assert derived.detail.endswith("so the calculator was not asked.")

    (declared,) = consult_bridges(
        _reading(declarations={"zeropoint": 25.94, "photometric_system": "AB"}),
        online=True,
    )

    assert declared.status == "not_needed"
    assert declared.detail == (
        "The zeropoint 25.94 is declared in data.calibration, so the "
        "calculator was not asked."
    )
    assert acs_calculator.calls == []


def test_a_conflicting_zeropoint_is_refused(tmp_path, acs_calculator) -> None:
    """A service never settles two readings of the header that disagree."""
    path = tmp_path / "camera.yaml"
    path.write_text(
        "\n".join(
            [
                "name: camera",
                "title: A camera",
                "priority: 70",
                "references: ['A manual.']",
                "match:",
                "  - keyword: TELESCOP",
                "    equals: HST",
                "quantities:",
                "  zeropoint:",
                "    keywords: [MAGZPT]",
                "    unit: mag",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    from baymorph.core.header_metadata import load_header_conventions

    primary, image = _acs_headers(PHOTFLAM=7.07236e-20)
    primary["MAGZPT"] = 25.0
    reading = read_header(
        image,
        primary_header=primary,
        conventions=[*load_header_conventions(), *load_header_conventions([path])],
    )

    assert reading["zeropoint"].status == "conflicting"

    (outcome,) = consult_bridges(reading, online=True)

    assert outcome.status == "refused"
    assert "does not settle a conflict of the header readings" in outcome.detail
    assert acs_calculator.calls == []


def test_the_request_asks_for_the_middle_of_the_observation() -> None:
    """The body is that of acstools; the date is the middle of the dates."""
    query = AcsZeropointBridge().plan(_reading())

    assert json.dumps(query.body) == (
        '{"date": "2007-01-22", "detector": "WFC", "filter": "F814W"}'
    )
    assert query.subject == "the AB zeropoint of WFC F814W on 2007-01-22"
    assert query.rule["span_days"] == pytest.approx(0.08599546)
    assert query.rule["middle_mjd"] == pytest.approx(54122.70069823)
    assert query.rule["max_span_days"] == 365.25
    assert query.rule["coverage_tolerance_s"] == 1.0
    assert query.check_bodies == ()
    assert query.rule["check_dates"] == []

    # Over four days the start, the middle and the end fall on three dates:
    # the middle one is asked for, the first and last ones check it.
    assert len({_calendar_date(mjd) for mjd in (54120.9, 54122.9, 54124.9)}) == 3

    spread = AcsZeropointBridge().plan(_reading(EXPSTART=54120.9, EXPEND=54124.9))

    assert spread.body["date"] == spread.rule["date"] == "2007-01-22"
    assert [body["date"] for body in spread.check_bodies] == [
        "2007-01-20",
        "2007-01-24",
    ]
    assert spread.rule["check_dates"] == ["2007-01-20", "2007-01-24"]
    assert spread.subject == (
        "the AB zeropoint of WFC F814W on 2007-01-22 and on 2007-01-20 and "
        "2007-01-24 to check that it holds for the whole observation"
    )

    # The first and last dates must be served too, not only the middle one.
    with pytest.raises(calibration_bridges.BridgeRefusal, match="2002-03-06 to"):
        AcsZeropointBridge().plan(_reading(EXPSTART=52339.5, EXPEND=52341.5))


def test_an_observation_over_several_dates_must_have_one_zeropoint(
    acs_calculator,
) -> None:
    """The zeropoints of the first and last dates must agree with the middle."""
    acs_calculator.by_date.update(
        {
            "2007-01-20": acs_calculator.answers["2007-01-22"],
            "2007-01-24": acs_calculator.answers["2007-01-22"],
        }
    )

    (steady,) = consult_bridges(_reading(EXPSTART=54120.9, EXPEND=54124.9), online=True)

    assert steady.status == "applied"
    assert steady.values["zeropoint"] == pytest.approx(25.931)
    assert steady.checks["ab_on_2007-01-20"] == pytest.approx(25.931)
    assert steady.checks["epoch_tolerance_mag"] == 0.002
    assert len(steady.check_responses) == 2
    assert [call[1]["date"] for call in acs_calculator.calls] == [
        "2007-01-22",
        "2007-01-20",
        "2007-01-24",
    ]

    # Across the step of July 2006 the zeropoint falls by 0.016 mag. The
    # answer of the middle date is synthetic: that of 2006-07-20.
    acs_calculator.calls.clear()
    acs_calculator.by_date["2006-07-05"] = acs_calculator.answers["2006-07-20"]
    start, end = 53906.5, 53936.5

    (straddling,) = consult_bridges(_reading(EXPSTART=start, EXPEND=end), online=True)

    assert [call[1]["date"] for call in acs_calculator.calls] == [
        "2006-07-05",
        "2006-06-20",
        "2006-07-20",
    ]
    assert straddling.status == "refused"
    assert straddling.values == {}
    assert "differ by up to 0.016 mag, more than 0.002" in straddling.detail
    assert straddling.detail.endswith(
        "The flux priors follow the rule without a zeropoint; declare zeropoint "
        "and photometric_system in data.calibration (--zeropoint, "
        "--photometric-system)."
    )
    assert straddling.run == {"asked_at": acs_calculator.asked_at, "error": None}
    assert len(straddling.check_responses) == 2


@pytest.mark.parametrize(
    ("changes", "date_asked"),
    [
        ({"EXPSTART": 54122.0, "EXPEND": 54122.0 + 365.25}, "2007-07-23"),
        ({"EXPSTART": 54122.0, "EXPEND": 54122.0 + 1399.5 / 86400.0}, "2007-01-22"),
        ({"EXPSTART": 52340.0, "EXPEND": 52340.1}, "2002-03-07"),
        ({"EXPSTART": 62501.8, "EXPEND": 62501.9}, "2029-12-31"),
    ],
)
def test_the_limits_of_the_rule_are_served(changes, date_asked) -> None:
    """A span of 365.25 days, the coverage tolerance and the served dates."""
    assert AcsZeropointBridge().plan(_reading(**changes)).body["date"] == date_asked


def test_the_documented_answer_gives_the_ab_zeropoint(acs_calculator) -> None:
    """The answer is checked against its PHOTFLAM and pivot wavelength."""
    notes: list[str] = []

    (outcome,) = consult_bridges(_reading(), online=True, notify=notes.append)

    assert outcome.status == "applied"
    assert outcome.values == {"zeropoint": 25.931, "photometric_system": "AB"}
    assert outcome.checks == {
        "ab_from_photflam_and_pivot": pytest.approx(25.93103),
        "st_from_photflam": pytest.approx(26.76671),
        "tolerance_mag": 0.002,
    }
    assert outcome.response == acs_calculator.answers["2007-01-22"]
    assert outcome.run == {"asked_at": acs_calculator.asked_at, "error": None}
    assert outcome.request == {
        "method": "POST",
        "url": ACS_ENDPOINT,
        "body": {"date": "2007-01-22", "detector": "WFC", "filter": "F814W"},
        "check_bodies": [],
        "timeout_seconds": 30.0,
    }
    assert acs_calculator.calls == [
        (
            ACS_ENDPOINT,
            {"date": "2007-01-22", "detector": "WFC", "filter": "F814W"},
            30.0,
        )
    ]
    assert notes == [
        "Asking the ACS Zeropoint Calculator of STScI for the AB zeropoint of "
        "WFC F814W on 2007-01-22 (timeout 30 s for each step of the connection)."
    ]
    assert "gave the AB zeropoint 25.931; its PHOTFLAM 7.1337e-20" in outcome.detail
    assert "its ST zeropoint 26.767 agrees with the 26.76671" in outcome.detail
    assert outcome.detail.endswith(
        "data.calibration declares zeropoint 25.931 and photometric_system AB."
    )

    declarations, sources = bridged_declarations({}, (outcome,))

    assert declarations == {"zeropoint": 25.931, "photometric_system": "AB"}
    assert sources == {
        "zeropoint": "stsci_acs_zeropoint",
        "photometric_system": "stsci_acs_zeropoint",
    }

    # The tile of COSMOS was observed in 2004: the calculator gives 25.952.
    (earlier,) = consult_bridges(
        _reading(EXPSTART=53110.31740397, EXPEND=53110.32328424, EXPTIME=500.0),
        online=True,
    )

    assert earlier.values["zeropoint"] == pytest.approx(25.952)
    assert acs_calculator.calls[-1][1]["date"] == "2004-04-15"


@pytest.mark.parametrize(
    ("changes", "declarations", "reason"),
    [
        ({"EXPSTART": None, "EXPEND": None}, None, "no usable observation dates"),
        ({"EXPEND": 54122.5}, None, "ends (EXPEND 54122.5000) before it starts"),
        ({"EXPEND": 54522.6577005}, None, "spans 400.0 days"),
        ({"EXPSTART": 54122.0, "EXPEND": 54122.0 + 365.26}, None, "more than 365.25"),
        (
            {"EXPSTART": 53110.31740397, "EXPEND": 53110.32328424, "EXPTIME": 4056.0},
            None,
            "the exposure time, 4056 s (EXPTIME), exceeds the 508 s",
        ),
        (
            {"EXPSTART": 53110.31740397, "EXPEND": 53110.32328424},
            {"exposure_time": 4056.0},
            "the exposure time, 4056 s (data.calibration), exceeds the 508 s",
        ),
        ({"DETECTOR": "SBC"}, None, "serves the detectors WFC and HRC"),
        ({"DETECTOR": None}, None, "does not give the detector (DETECTOR)"),
        ({"FILTER1": "F555W"}, None, "the filter is conflicting (FILTER1)"),
        ({"FILTER2": "F220W"}, None, "the filter is 'F220W' (FILTER2)"),
        ({"D001OUUN": None}, None, "the image unit is 'ELECTRONS' (explicit)"),
        (
            {
                "D001OUUN": None,
                "BUNIT": "ELECTRONS/S",
                "FILENAME": "j9ty22010_drz.fits",
            },
            None,
            "the zeropoints of the calculator hold for images drizzled",
        ),
        ({"EXPTIME": None}, None, "the exposure time is missing (EXPTIME)"),
        (
            {"EXPSTART": 52244.0, "EXPEND": 52244.1},
            None,
            "2001-12-01 to 2001-12-01, lies partly or wholly outside",
        ),
        (
            {"EXPSTART": 62503.0, "EXPEND": 62503.1},
            None,
            "2030-01-02 to 2030-01-02, lies partly or wholly outside",
        ),
        (
            {"EXPSTART": 3.0e9, "EXPEND": 3.0e9 + 0.05},
            None,
            "MJD 3000000000.0000 to MJD 3000000000.0500",
        ),
    ],
)
def test_refusals_are_stated_and_ask_nothing(
    acs_calculator, changes, declarations, reason
) -> None:
    """A reading that does not allow a request is refused with its reason."""
    (outcome,) = consult_bridges(
        _reading(declarations, **changes), online=True, notify=pytest.fail
    )

    assert outcome.status == "refused"
    assert reason in outcome.detail
    assert outcome.detail.endswith(DECLARE)
    assert outcome.request is None
    assert acs_calculator.calls == []


def test_an_hrc_image_asks_for_its_detector() -> None:
    """The second CCD detector of ACS is served with its own filters."""
    query = AcsZeropointBridge().plan(_reading(DETECTOR="HRC", FILTER2="F220W"))

    assert query.body == {"date": "2007-01-22", "detector": "HRC", "filter": "F220W"}


def test_offline_records_the_request_and_asks_nothing(acs_calculator) -> None:
    """Offline, the planned request is recorded and nothing is sent."""
    (outcome,) = consult_bridges(_reading(), online=False, notify=pytest.fail)

    assert outcome.status == "offline"
    assert outcome.request["body"]["date"] == "2007-01-22"
    assert outcome.run is None
    assert outcome.response is None
    assert "the request is recorded" in outcome.detail
    assert acs_calculator.calls == []

    with pytest.raises(TypeError, match="online must be True or False"):
        consult_bridges(_reading(), online="no")


def _cell(name: str, value):
    """Return a change of one cell of the row of an answer."""

    def change(answer: dict) -> dict:
        columns = [header.split()[0] for header in answer["headers"]]
        answer["rows"][0][columns.index(name)] = value

        return answer

    return change


def _renamed(old: str, new: str):
    """Return a change of the name of one column of an answer."""

    def change(answer: dict) -> dict:
        answer["headers"] = [
            new if header.split()[0] == old else header for header in answer["headers"]
        ]

        return answer

    return change


def _without(name: str):
    """Return a change that removes one column of an answer."""

    def change(answer: dict) -> dict:
        index = [header.split()[0] for header in answer["headers"]].index(name)
        del answer["headers"][index]
        del answer["rows"][0][index]

        return answer

    return change


@pytest.mark.parametrize(
    ("error", "change", "reason"),
    [
        (
            BridgeTransportError("did not answer within 30 s", "timed out"),
            None,
            "did not answer within 30 s",
        ),
        (
            BridgeTransportError("could not be reached", "nodename nor servname"),
            None,
            "could not be reached",
        ),
        (
            BridgeTransportError("answered with HTTP status 502", "Bad Gateway"),
            None,
            "answered with HTTP status 502",
        ),
        (
            BridgeAnswerError("gave an answer that is not JSON", "Expecting value"),
            None,
            "gave an answer that is not JSON",
        ),
        (None, lambda answer: {**answer, "rows": []}, "0 rows instead of one"),
        (
            None,
            lambda answer: {**answer, "rows": answer["rows"] * 2},
            "2 rows instead of one",
        ),
        (None, lambda answer: {**answer, "rows": ["x"]}, "row is not a list"),
        (
            None,
            lambda answer: {**answer, "rows": [answer["rows"][0][:-1]]},
            "its row and its headers differ in length",
        ),
        (
            None,
            lambda answer: {**answer, "rows": [answer["rows"][0] + ["x"]]},
            "its row and its headers differ in length",
        ),
        (None, lambda answer: {"rows": answer["rows"]}, "no list of headers and rows"),
        (None, lambda answer: [], "no list of headers and rows"),
        (None, lambda answer: "text", "no list of headers and rows"),
        (None, _cell("Filter", "F606W"), "its filter is 'F606W' instead of 'F814W'"),
        (None, _cell("Detector", "HRC"), "its detector is 'HRC' instead of 'WFC'"),
        (None, _cell("ABmag", "n/a"), "its ABmag 'n/a' is not a number"),
        (None, _cell("ABmag", "inf"), "its ABmag 'inf' is not finite"),
        (None, _cell("PHOTFLAM", "nan"), "its PHOTFLAM 'nan' is not finite"),
        (None, _cell("ABmag", True), "its ABmag True is not a number"),
        (None, _cell("ABmag", "25.941"), "differs from the 25.93103"),
        (None, _cell("STmag", "26.900"), "its ST zeropoint 26.9 differs from the"),
        (None, _cell("PHOTFLAM", "0"), "must be positive"),
        (None, _cell("PHOTLAM", "0"), "must be positive"),
        (
            None,
            _cell("PHOTFLAM", 10**309),
            "its PHOTFLAM 1000000000000000000000000000000000000... is not a number",
        ),
        (
            None,
            _cell("Filter", "F" * 500),
            "its filter is '" + "F" * 36 + "... instead of 'F814W'",
        ),
        (None, _without("ABmag"), "it has no column ABmag"),
        (None, _without("STmag"), "it has no column STmag"),
        (None, _renamed("STmag", "ST"), "it has no column STmag"),
        (
            None,
            lambda answer: {
                **answer,
                "headers": answer["headers"][:-2] + ["ABmag", "ABmag"],
            },
            "it names the column ABmag twice",
        ),
    ],
)
def test_failures_state_a_stable_reason_after_one_request(
    acs_calculator, error, change, reason
) -> None:
    """A failed request gives no value, a stable reason and the raw error."""
    acs_calculator.error = error

    if change is not None:
        acs_calculator.answer = change(
            json.loads(json.dumps(acs_calculator.answers["2007-01-22"]))
        )

    (outcome,) = consult_bridges(_reading(), online=True)

    assert outcome.status == "failed"
    assert outcome.values == {}
    assert reason in outcome.detail
    assert "propose again online" in outcome.detail
    assert outcome.run["asked_at"] == acs_calculator.asked_at
    assert len(acs_calculator.calls) == 1

    if error is not None:
        assert outcome.run["error"] == error.detail
        assert error.detail not in outcome.detail


def test_an_answer_naming_the_pivot_photplam_is_read(acs_calculator) -> None:
    """The pivot wavelength may be named as in the headers of HST images."""
    acs_calculator.answer = _renamed("PHOTLAM", "PHOTPLAM")(
        json.loads(json.dumps(acs_calculator.answers["2007-01-22"]))
    )

    (outcome,) = consult_bridges(_reading(), online=True)

    assert outcome.status == "applied"


def test_interpret_refuses_an_empty_answer() -> None:
    """A JSON null, which the fake service cannot send, is refused too."""
    query = AcsZeropointBridge().plan(_reading())

    with pytest.raises(BridgeAnswerError, match="no list of headers and rows"):
        AcsZeropointBridge().interpret(query, [None])


def _answer(body: bytes, *, code: int = 200, url: str = ACS_ENDPOINT, **headers):
    """Return a response of urllib to an HTTPS request, without a network."""
    message = email.message.Message()

    for name, value in headers.items():
        message[name] = value

    response = urllib.response.addinfourl(io.BytesIO(body), message, url, code)
    response.msg = "OK" if code == 200 else "Error"

    return response


def _transport(monkeypatch, *results) -> list[dict]:
    """Answer the HTTPS requests of urllib with ``results``, in order.

    A result is a response or an exception to raise. Every request is
    recorded with its timeout and the SSL context of the handler.
    """
    pending = list(results)
    sent: list[dict] = []

    def https_open(handler, request):
        sent.append(
            {
                "request": request,
                "timeout": request.timeout,
                "context": handler._context,
            }
        )
        result = pending.pop(0)

        if isinstance(result, BaseException):
            raise result

        return result

    monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", https_open)

    return sent


def test_post_json_posts_one_verified_json_request(monkeypatch) -> None:
    """One POST with a JSON body, a verified certificate and the timeout."""
    answer = {"headers": ["Filter"], "rows": [["F814W"]]}
    sent = _transport(monkeypatch, _answer(json.dumps(answer).encode("utf-8")))
    body = {"date": "2007-01-22", "detector": "WFC", "filter": "F814W"}

    assert real_post_json(ACS_ENDPOINT, body, timeout=30.0) == answer

    (record,) = sent
    request = record["request"]

    assert request.full_url == ACS_ENDPOINT
    assert request.get_method() == "POST"
    assert request.data == json.dumps(body).encode("utf-8")
    assert request.get_header("Content-type") == "application/json"
    assert request.get_header("Accept") == "application/json"
    assert request.get_header("User-agent").startswith("BayMorph/")
    assert record["timeout"] == 30.0
    assert record["context"].verify_mode == ssl.CERT_REQUIRED
    assert record["context"].check_hostname


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_post_json_follows_no_redirection(monkeypatch, code) -> None:
    """A redirection is refused at once: no second request is sent."""
    sent = _transport(
        monkeypatch, _answer(b"", code=code, Location="http://elsewhere.invalid/x")
    )

    with pytest.raises(BridgeTransportError) as raised:
        real_post_json(ACS_ENDPOINT, {"date": "2007-01-22"}, timeout=30.0)

    assert raised.value.reason == "redirected the request to another address"
    assert raised.value.detail == "http://elsewhere.invalid/x"
    assert len(sent) == 1


def test_a_redirection_is_reported_as_written(monkeypatch) -> None:
    """A long or malformed address of redirection is reported, not parsed."""
    long_address = "https://elsewhere.invalid/" + "x" * 300

    for location, detail in (
        ("https://[::1/x", "https://[::1/x"),
        (long_address, long_address[:117] + "..."),
    ):
        _transport(monkeypatch, _answer(b"", code=302, Location=location))

        with pytest.raises(BridgeTransportError) as raised:
            real_post_json(ACS_ENDPOINT, {"date": "2007-01-22"}, timeout=30.0)

        assert raised.value.reason == "redirected the request to another address"
        assert raised.value.detail == detail


def test_a_malformed_proxy_is_a_failure_of_the_transport(monkeypatch) -> None:
    """A proxy of the environment that urllib cannot use fails the request."""
    for name in ("https_proxy", "HTTPS_PROXY"):
        monkeypatch.setenv(name, "http:/proxy.invalid")

    sent = _transport(monkeypatch)

    with pytest.raises(BridgeTransportError, match="could not be reached"):
        real_post_json(ACS_ENDPOINT, {"date": "2007-01-22"}, timeout=30.0)

    assert sent == []


def test_post_json_asks_over_https_only(monkeypatch) -> None:
    """An address that does not use HTTPS is refused before any request."""
    sent = _transport(monkeypatch)

    for url in ("http://example.org/", "https://", "file:///tmp/answer.json"):
        with pytest.raises(ValueError, match="over HTTPS only"):
            real_post_json(url, {"date": "2007-01-22"}, timeout=30.0)

    assert sent == []


@pytest.mark.parametrize(
    ("result", "error", "reason"),
    [
        (_answer(b"x" * 65_537), BridgeAnswerError, "larger than 64 KiB"),
        (_answer(b"<html>"), BridgeAnswerError, "not JSON"),
        (_answer(b'{"rows": NaN}'), BridgeAnswerError, "not JSON"),
        (_answer(b'{"rows": 1e999}'), BridgeAnswerError, "not JSON"),
        (_answer(b'{"rows": Infinity}'), BridgeAnswerError, "not JSON"),
        (_answer(b'{"rows": -Infinity}'), BridgeAnswerError, "not JSON"),
        (
            _answer(b"[" * 30_000 + b"]" * 30_000),
            BridgeAnswerError,
            f"nested deeper than {MAX_ANSWER_DEPTH} levels",
        ),
        (
            _answer(b"[" * 600 + b"]" * 600),
            BridgeAnswerError,
            f"nested deeper than {MAX_ANSWER_DEPTH} levels",
        ),
        (
            _answer(b"{}", url="https://example.org/"),
            BridgeTransportError,
            "redirected",
        ),
        (
            _answer(b"Bad Gateway", code=502),
            BridgeTransportError,
            "answered with HTTP status 502",
        ),
        (
            urllib.error.URLError(ssl.SSLCertVerificationError("bad certificate")),
            BridgeTransportError,
            "certificate was not verified",
        ),
        (
            urllib.error.URLError(TimeoutError("timed out")),
            BridgeTransportError,
            "did not answer within 30 s",
        ),
        (TimeoutError("timed out"), BridgeTransportError, "did not answer within 30 s"),
        (
            urllib.error.URLError("nodename nor servname provided"),
            BridgeTransportError,
            "could not be reached",
        ),
        (
            http.client.RemoteDisconnected("closed"),
            BridgeTransportError,
            "closed the connection without answering",
        ),
        (
            http.client.IncompleteRead(b""),
            BridgeTransportError,
            "broke the connection while answering",
        ),
        (
            http.client.BadStatusLine("garbage"),
            BridgeTransportError,
            "did not give a valid HTTP answer",
        ),
        (
            ConnectionRefusedError("refused"),
            BridgeTransportError,
            "could not be reached",
        ),
    ],
)
def test_post_json_maps_each_failure_to_a_stable_reason(
    monkeypatch, result, error, reason
) -> None:
    """Every failure of the transport or of the answer has a stable reason."""
    sent = _transport(monkeypatch, result)

    with pytest.raises(error) as raised:
        real_post_json(ACS_ENDPOINT, {"date": "2007-01-22"}, timeout=30.0)

    assert reason in raised.value.reason
    assert len(sent) == 1


def test_an_answer_at_the_depth_limit_is_read(monkeypatch) -> None:
    """Sixteen levels of nesting are read; the ACS answers hold three."""
    nested = "[" * MAX_ANSWER_DEPTH + "]" * MAX_ANSWER_DEPTH
    _transport(monkeypatch, _answer(nested.encode("utf-8")))

    assert real_post_json(ACS_ENDPOINT, {}, timeout=30.0) is not None


def test_modified_julian_dates_become_calendar_dates() -> None:
    """The first and last days the calculator serves, and the Coma date."""
    assert _calendar_date(52340.0) == date(2002, 3, 7)
    assert _calendar_date(62501.99) == date(2029, 12, 31)
    assert _calendar_date(54122.70069823) == date(2007, 1, 22)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"endpoint": "http://example.org/"}, "must use HTTPS"),
        ({"endpoint": "https://"}, "must use HTTPS"),
        ({"supplies": ("pixel_scale",)}, "must supply declarable quantities"),
        ({"supplies": ()}, "must supply declarable quantities"),
        ({"reads": ("telescope", "colour")}, "must resolve and read quantities"),
        ({"references": ()}, "must cite its documentation"),
    ],
)
def test_the_registry_refuses_a_bad_bridge(change, message) -> None:
    """Every rule of the registry is checked, also for bridges given in a call."""
    _check_bridges(BRIDGES)
    broken = type("Broken", (AcsZeropointBridge,), change)()

    with pytest.raises(ValueError, match=message):
        _check_bridges([broken])

    with pytest.raises(ValueError, match=message):
        consult_bridges(_reading(), online=False, bridges=[broken])


def test_the_registry_needs_unique_names() -> None:
    """Two bridges of one name could not be told apart in a report."""
    with pytest.raises(ValueError, match="unique, non-empty names"):
        _check_bridges([AcsZeropointBridge(), AcsZeropointBridge()])


def test_a_service_never_replaces_a_declaration() -> None:
    """The precedence of the user is an invariant of the declarations."""
    (outcome,) = consult_bridges(_reading(), online=False)
    applied = BridgeOutcome(
        name="stsci_acs_zeropoint",
        title="ACS Zeropoint Calculator of STScI",
        status="applied",
        detail="",
        references=outcome.references,
        inputs=outcome.inputs,
        values={"zeropoint": 25.931},
    )

    with pytest.raises(RuntimeError, match="which the user declared"):
        bridged_declarations({"zeropoint": 26.0}, (applied,))


def _names_and_strings(path: Path) -> set[str]:
    """Return the names and the string constants of a source file."""
    found: set[str] = set()
    source = path.read_text(encoding="utf-8")

    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in (tokenize.NAME, tokenize.STRING):
            found.add(token.string)

    return found


def _imported_roots(path: Path) -> set[str]:
    """Return the top-level modules a source file imports.

    A relative import stays inside BayMorph, so its root is ``baymorph``.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            roots.add("baymorph" if node.level else str(node.module).split(".")[0])

    return roots


NETWORK_MODULES = frozenset(
    {
        "aiohttp",
        "ftplib",
        "http",
        "httpx",
        "imaplib",
        "poplib",
        "requests",
        "smtplib",
        "socket",
        "socketserver",
        "ssl",
        "urllib",
        "urllib3",
        "websocket",
        "websockets",
        "xmlrpc",
    }
)


def test_only_the_advisor_reaches_the_network() -> None:
    """Only the bridges import network modules, and only the advisor names them.

    The names are searched in the tokens of every module, so an import
    through the package, a relative import or a string given to importlib is
    found as well as an absolute import.
    """
    owners = {"pipeline/advisor.py", "pipeline/calibration_bridges.py"}
    bridge_names = {
        "calibration_bridges",
        "consult_bridges",
        "bridged_declarations",
        "post_json",
    }

    for path in sorted(PACKAGE.rglob("*.py")):
        relative = path.relative_to(PACKAGE).as_posix()

        if relative != "pipeline/calibration_bridges.py":
            assert not _imported_roots(path) & NETWORK_MODULES, relative

        if relative not in owners:
            tokens = _names_and_strings(path)

            assert not any(
                name in token for token in tokens for name in bridge_names
            ), relative


def test_the_network_scan_sees_every_form_of_import(tmp_path) -> None:
    """The scan of the previous test catches the forms that bypass an AST."""
    forms = [
        "from .calibration_bridges import consult_bridges\n",
        "from baymorph.pipeline import calibration_bridges\n",
        "from . import calibration_bridges\n",
        "import importlib\n"
        "importlib.import_module('baymorph.pipeline.calibration_bridges')\n",
    ]

    for form in forms:
        path = tmp_path / "module.py"
        path.write_text(form, encoding="utf-8")

        assert "calibration_bridges" in " ".join(_names_and_strings(path)), form


def test_the_fixture_blocks_the_transport_of_urllib() -> None:
    """Without its own answers, a test cannot send a request."""
    assert calibration_bridges.post_json is not real_post_json

    for url in ("https://guard.invalid/", "http://guard.invalid/"):
        with pytest.raises(pytest.fail.Exception, match="tried to send a request"):
            urllib.request.urlopen(url, timeout=1.0)

    with pytest.raises(pytest.fail.Exception, match="tried to send a request"):
        real_post_json(ACS_ENDPOINT, {"date": "2007-01-22"}, timeout=30.0)
