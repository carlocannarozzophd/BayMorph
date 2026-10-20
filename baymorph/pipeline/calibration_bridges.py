"""Calibration services that the observation advisor consults (AD-054).

A calibration bridge asks an online service for a calibration quantity that
the user does not declare and the header does not give usably, such as the
zeropoint of an HST/ACS image whose ``PHOTFLAM`` is 0. Five rules hold:

1. Bridges belong to the preparation phase. Only the observation advisor
   consults them, and it writes their values to ``data.calibration`` of the
   proposed configuration; the fit reads them from there and never reaches
   the network.
2. A bridge reads the quantities of a header reading, never raw keywords, so
   every input it uses has a convention, a status and a source.
3. A value that a fixed, documented table gives belongs to a header
   convention, not to a bridge; a bridge is for values that only a service
   gives.
4. A declared value, or a usable value of the header, is never replaced, and
   a service never settles a conflict between readings of the header.
5. Each request that a bridge plans is sent once, with no retry, and no
   redirection is followed. A failure leaves the quantity unusable, and the
   outcome says why.

This is the only module of BayMorph that reaches the network.
"""

from __future__ import annotations

import http.client
import json
import math
import ssl
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from types import MappingProxyType
from typing import Any, ClassVar

from baymorph.core.header_metadata import (
    DECLARABLE_QUANTITIES,
    DERIVED_QUANTITIES,
    QUANTITIES,
    HeaderQuantity,
    HeaderReading,
    canonical_unit,
    validate_declarations,
)

BRIDGE_STATUSES = ("applied", "not_needed", "refused", "offline", "failed")
"""Outcomes of a consultation: the service gave the values; the quantity was
already known; the image does not allow a request; the advisor was offline;
the request or its answer failed."""

TIMEOUT_SECONDS = 30.0
"""Longest wait, in seconds, for each operation on the connection."""

MAX_ANSWER_BYTES = 65_536
"""Largest answer read from a service, in bytes."""

MAX_ANSWER_DEPTH = 16
"""Deepest nesting of lists and objects accepted in an answer."""


class BridgeRefusal(ValueError):
    """The image does not allow a request; ``reason`` says why."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class BridgeAnswerError(ValueError):
    """The service answered with something the bridge cannot use.

    ``reason`` is a stable sentence; ``detail`` holds the verbatim error.
    """

    def __init__(self, reason: str, detail: str | None = None) -> None:
        super().__init__(reason if detail is None else f"{reason} ({detail})")
        self.reason = reason
        self.detail = detail


class BridgeTransportError(OSError):
    """The service was not reached, or did not answer.

    ``reason`` is a stable sentence; ``detail`` holds the verbatim error.
    """

    def __init__(self, reason: str, detail: str | None = None) -> None:
        super().__init__(reason if detail is None else f"{reason} ({detail})")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class BridgeQuery:
    """A request that a bridge plans from a header reading, without sending it.

    Parameters
    ----------
    body : dict
        JSON body of the request, with its keys in the order of the service.
    subject : str
        What is asked, such as "the AB zeropoint of WFC F814W on 2007-01-22".
    basis : str
        How the inputs of the reading became the body.
    rule : dict
        The numbers of the rule that chose the body.
    check_bodies : tuple of dict
        Bodies of further requests whose answers check the first one, such as
        the zeropoints of the first and last dates of an observation.
    """

    body: dict[str, Any]
    subject: str
    basis: str
    rule: dict[str, Any]
    check_bodies: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class BridgeAnswer:
    """The values a bridge takes from an answer, and the checks they passed.

    Parameters
    ----------
    values : dict
        Declarable values by quantity, such as ``{"zeropoint": 25.931,
        "photometric_system": "AB"}``.
    checks : dict of str to float
        The numbers the answer was checked with.
    statement : str
        What the service gave and how it was checked, completing a sentence
        whose subject is the service.
    """

    values: dict[str, Any]
    checks: dict[str, float]
    statement: str


@dataclass(frozen=True)
class BridgeOutcome:
    """The outcome of the consultation of one calibration service.

    Parameters
    ----------
    name, title : str
        Name and title of the bridge.
    status : str
        One of ``BRIDGE_STATUSES``.
    detail : str
        What was done and why, in words that do not change between two
        consultations with the same answer.
    references : tuple of str
        Documentation of the service.
    inputs : dict
        The conventions matched and the quantities of the reading the bridge
        read.
    values : dict
        Values the service gave; empty unless the status is ``applied``.
    rule, request : dict, optional
        The numbers of the rule and the requests, once the bridge planned
        them; a refusal after the service was asked, because its answers
        hold no single value, records them too.
    response : object, optional
        The decoded answer of the service to the first request, verbatim.
    check_responses : list, optional
        The decoded answers to the requests that check the first one.
    checks : dict, optional
        The numbers the answer was checked with, when it was accepted.
    run : dict, optional
        When the service was asked, ``asked_at``, and the verbatim error of a
        failed request, ``error``: the only part that changes between two
        consultations with the same answer.
    """

    name: str
    title: str
    status: str
    detail: str
    references: tuple[str, ...]
    inputs: dict[str, Any]
    values: dict[str, Any] = field(default_factory=dict)
    rule: dict[str, Any] | None = None
    request: dict[str, Any] | None = None
    response: Any = None
    check_responses: list[Any] | None = None
    checks: dict[str, float] | None = None
    run: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable description of the outcome."""
        return {
            "name": self.name,
            "title": self.title,
            "status": self.status,
            "detail": self.detail,
            "values": dict(self.values),
            "references": list(self.references),
            "inputs": deepcopy(self.inputs),
            "rule": deepcopy(self.rule),
            "request": deepcopy(self.request),
            "response": deepcopy(self.response),
            "check_responses": deepcopy(self.check_responses),
            "checks": deepcopy(self.checks),
            "run": deepcopy(self.run),
        }


class CalibrationBridge(ABC):
    """An online calibration service that gives declarable quantities.

    A bridge recognizes the images it serves from their header reading,
    plans its requests from that reading and interprets the answers. All
    three are pure, so :func:`consult_bridges` alone sends the requests.

    Attributes
    ----------
    name : str
        Identifier written as the source of the values it gives.
    title : str
        Name of the service for a reader.
    service : str
        Short name of the service in a sentence, such as "the calculator".
    endpoint : str
        HTTPS address that the request is posted to.
    supplies : tuple of str
        Declarable quantities it gives; the first is the one it resolves.
    reads : tuple of str
        Quantities of the reading it uses, recorded as its inputs.
    fallback : str
        What the proposal does without the values, in a sentence.
    references : tuple of str
        Documentation of the service.
    """

    name: ClassVar[str]
    title: ClassVar[str]
    service: ClassVar[str]
    endpoint: ClassVar[str]
    supplies: ClassVar[tuple[str, ...]]
    reads: ClassVar[tuple[str, ...]]
    fallback: ClassVar[str]
    references: ClassVar[tuple[str, ...]]

    @abstractmethod
    def serves(self, reading: HeaderReading) -> bool:
        """Return whether the service gives values for this image."""

    @abstractmethod
    def plan(self, reading: HeaderReading) -> BridgeQuery:
        """Return the request for this image.

        Raises
        ------
        BridgeRefusal
            If the reading does not allow a request.
        """

    @abstractmethod
    def interpret(self, query: BridgeQuery, answers: Sequence[Any]) -> BridgeAnswer:
        """Return the values of the answers to ``query``.

        ``answers`` holds the answer to ``query.body`` first, then those to
        ``query.check_bodies``, in order.

        Raises
        ------
        BridgeAnswerError
            If an answer cannot be read or fails a check.
        BridgeRefusal
            If the answers are valid but hold no single value for the image.
        """


def consult_bridges(
    reading: HeaderReading,
    *,
    online: bool,
    bridges: Sequence[CalibrationBridge] | None = None,
    notify: Callable[[str], None] | None = None,
) -> tuple[BridgeOutcome, ...]:
    """Consult every calibration service that serves an image.

    A service is asked only for a quantity that the reading leaves ambiguous
    or missing and that no earlier service gave; a conflicting reading is
    refused, and so is an image whose reading does not allow a request, or
    whose answers hold no single value. Each service is consulted at most
    once, and each request it plans is sent once, with no retry.

    Parameters
    ----------
    reading : HeaderReading
        Reading of the header with the declarations of the user.
    online : bool
        Whether a service may be asked. Offline, the planned request is
        recorded and nothing is sent.
    bridges : sequence of CalibrationBridge, optional
        The bridges to consult, in order; ``BRIDGES`` by default.
    notify : callable, optional
        Called with one line before a request is sent.

    Returns
    -------
    tuple of BridgeOutcome
        One outcome per bridge that serves the image, in order.
    """
    if not isinstance(online, bool):
        raise TypeError("online must be True or False.")

    if bridges is None:
        bridges = BRIDGES
    else:
        _check_bridges(bridges)

    outcomes: list[BridgeOutcome] = []
    supplied: set[str] = set()

    for bridge in bridges:
        if not bridge.serves(reading):
            continue

        outcome = _consult(
            bridge, reading, online=online, notify=notify, supplied=supplied
        )
        outcomes.append(outcome)
        supplied.update(outcome.values)

    return tuple(outcomes)


def bridged_declarations(
    declared: Mapping[str, Any] | None, outcomes: Sequence[BridgeOutcome]
) -> tuple[dict[str, Any], dict[str, str]]:
    """Return the declarations of the user completed by the services.

    Parameters
    ----------
    declared : mapping or None
        Declarations of the user, as ``data.calibration`` holds them.
    outcomes : sequence of BridgeOutcome
        Outcomes of :func:`consult_bridges`.

    Returns
    -------
    tuple of dict
        The declarations, validated and in the order of
        ``DECLARABLE_QUANTITIES``, and the source of each: ``"user"`` or the
        name of the bridge that gave it.

    Raises
    ------
    RuntimeError
        If a service would replace a declaration of the user.
    """
    combined = validate_declarations(declared)
    sources = {name: "user" for name in combined}

    for outcome in outcomes:
        for name, value in outcome.values.items():
            if name in combined:
                owner = (
                    "the user declared"
                    if sources[name] == "user"
                    else f"{sources[name]} gave"
                )
                raise RuntimeError(
                    f"The calibration service {outcome.name} would replace "
                    f"data.calibration.{name}, which {owner}."
                )

            combined[name] = value
            sources[name] = outcome.name

    validated = validate_declarations(combined)

    return validated, {name: sources[name] for name in validated}


def _consult(
    bridge: CalibrationBridge,
    reading: HeaderReading,
    *,
    online: bool,
    notify: Callable[[str], None] | None,
    supplied: set[str],
) -> BridgeOutcome:
    """Consult one bridge that serves the image."""
    resolved = bridge.supplies[0]
    target = reading[resolved]
    label = resolved.replace("_", " ")
    common: dict[str, Any] = {
        "name": bridge.name,
        "title": bridge.title,
        "references": bridge.references,
        "inputs": _inputs(bridge, reading),
    }

    if target.usable or resolved in supplied:
        return BridgeOutcome(
            status="not_needed",
            detail=f"{_known(target, label, resolved in supplied)}, so "
            f"{bridge.service} was not asked.",
            **common,
        )

    state = _state(target, label)
    declare = (
        "declare "
        + " and ".join(bridge.supplies)
        + " in data.calibration ("
        + ", ".join("--" + name.replace("_", "-") for name in bridge.supplies)
        + ")"
    )

    if target.status == "conflicting":
        return BridgeOutcome(
            status="refused",
            detail=f"{state}, and a calibration service does not settle a "
            f"conflict of the header readings. {_capital(bridge.service)} was "
            f"not asked; {declare}.",
            **common,
        )

    try:
        query = bridge.plan(reading)
    except BridgeRefusal as refusal:
        return BridgeOutcome(
            status="refused",
            detail=f"{state}, and {refusal.reason}. {_capital(bridge.service)} "
            f"was not asked; {declare}.",
            **common,
        )

    planned: dict[str, Any] = {
        "rule": deepcopy(query.rule),
        "request": {
            "method": "POST",
            "url": bridge.endpoint,
            "body": dict(query.body),
            "check_bodies": [dict(body) for body in query.check_bodies],
            "timeout_seconds": TIMEOUT_SECONDS,
        },
    }

    if not online:
        return BridgeOutcome(
            status="offline",
            detail=f"{state}. Offline, the advisor did not ask {bridge.service} "
            f"for {query.subject}; the request is recorded. "
            f"{_capital(bridge.fallback)}.",
            **common,
            **planned,
        )

    if notify is not None:
        notify(
            f"Asking the {bridge.title} for {query.subject} "
            f"(timeout {TIMEOUT_SECONDS:g} s for each step of the connection)."
        )

    asked_at = _utc_now()
    responses: list[Any] = []

    def received() -> dict[str, Any]:
        return {
            "response": responses[0] if responses else None,
            "check_responses": responses[1:],
        }

    try:
        for body in (query.body, *query.check_bodies):
            responses.append(post_json(bridge.endpoint, body, timeout=TIMEOUT_SECONDS))

        answer = bridge.interpret(query, responses)
        values = _declarable_values(bridge, answer.values)
    except BridgeRefusal as refusal:
        return BridgeOutcome(
            status="refused",
            detail=f"{state}. Asked for {query.subject}, {bridge.service} "
            f"{refusal.reason}. {_capital(bridge.fallback)}; {declare}.",
            run={"asked_at": asked_at, "error": None},
            **received(),
            **common,
            **planned,
        )
    except (BridgeTransportError, BridgeAnswerError) as error:
        return BridgeOutcome(
            status="failed",
            detail=f"{state}. Asked for {query.subject}, {bridge.service} "
            f"{error.reason}. {_capital(bridge.fallback)}; propose again "
            f"online, or {declare}.",
            run={"asked_at": asked_at, "error": error.detail},
            **received(),
            **common,
            **planned,
        )

    declared = " and ".join(
        f"{name} {_format(value)}" for name, value in values.items()
    )

    return BridgeOutcome(
        status="applied",
        detail=f"{state}. Asked for {query.basis}, {bridge.service} "
        f"{answer.statement}. data.calibration declares {declared}.",
        values=values,
        checks=dict(answer.checks),
        run={"asked_at": asked_at, "error": None},
        **received(),
        **common,
        **planned,
    )


def _declarable_values(
    bridge: CalibrationBridge, values: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the values of an answer as declarations."""
    extra = sorted(set(values) - set(bridge.supplies))

    if extra or bridge.supplies[0] not in values:
        raise RuntimeError(
            f"The calibration bridge {bridge.name} gave {sorted(values)}, which "
            f"must include {bridge.supplies[0]} and lie within "
            f"{list(bridge.supplies)}."
        )

    try:
        return validate_declarations(values)
    except ValueError as error:
        raise BridgeAnswerError(
            "gave a value that cannot be declared", str(error)
        ) from None


def _inputs(bridge: CalibrationBridge, reading: HeaderReading) -> dict[str, Any]:
    """Return the conventions and the quantities of the reading a bridge read."""
    quantities = {}

    for name in bridge.reads:
        described = reading[name].as_dict()
        quantities[name] = {
            key: described[key]
            for key in ("status", "value", "unit", "keyword", "convention")
        }

    return {"conventions": list(reading.conventions), "quantities": quantities}


def _known(quantity: HeaderQuantity, label: str, supplied: bool) -> str:
    """Say where a quantity that needs no service is known from."""
    if supplied:
        return f"An earlier calibration service gave the {label}"

    value = _format(quantity.value)

    if quantity.status == "declared":
        return f"The {label} {value} is declared in data.calibration"

    unit = f" {quantity.unit}" if quantity.unit else ""
    provenance = ", ".join(part for part in (quantity.status, quantity.keyword) if part)

    if quantity.convention:
        provenance += f"; {quantity.convention}"

    return f"The header reading gives the {label} {value}{unit} ({provenance})"


def _state(quantity: HeaderQuantity, label: str) -> str:
    """Say what the header gives of a quantity that a service may give."""
    if quantity.status == "missing":
        return f"The header gives no {label}"

    origin = "; ".join(part for part in (quantity.keyword, quantity.convention) if part)
    suffix = f" ({origin})" if origin else ""

    if quantity.status == "conflicting":
        return f"The readings of the {label} of the header disagree{suffix}"

    return f"The {label} of the header is {quantity.status}{suffix}"


def _capital(text: str) -> str:
    """Return a sentence fragment with its first letter in upper case."""
    return text[:1].upper() + text[1:]


def _format(value: Any) -> str:
    """Format a value in a sentence."""
    if isinstance(value, float):
        return f"{value:.6g}"

    return str(value)


def post_json(url: str, body: Mapping[str, Any], *, timeout: float) -> Any:
    """Post a JSON body to an HTTPS address and return the decoded answer.

    The certificate of the server is always verified, and a redirection is
    never followed, so exactly one request is sent. At most
    ``MAX_ANSWER_BYTES`` are read, and an answer nested deeper than
    ``MAX_ANSWER_DEPTH`` levels, or holding a number that is not finite, is
    refused.

    Parameters
    ----------
    url : str
        HTTPS address.
    body : mapping
        Body of the request, written as JSON.
    timeout : float
        Longest wait, in seconds, for each operation on the connection.

    Returns
    -------
    object
        The answer decoded from JSON.

    Raises
    ------
    ValueError
        If the address does not use HTTPS.
    BridgeTransportError
        If the server is not reached, does not answer in time, answers with
        an HTTP error or redirects the request.
    BridgeAnswerError
        If the answer is too large, too deeply nested, or not JSON.
    """
    _require_https(url)
    request = urllib.request.Request(
        url,
        data=json.dumps(dict(body), allow_nan=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": _user_agent(),
        },
        method="POST",
    )
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
        _NoRedirection,
    )

    try:
        with opener.open(request, timeout=timeout) as response:
            if response.geturl() != url:
                raise BridgeTransportError(
                    "redirected the request to another address",
                    _shortened(response.geturl()),
                )

            raw = response.read(MAX_ANSWER_BYTES + 1)
    except BridgeTransportError:
        raise
    except urllib.error.HTTPError as error:
        if error.code in _REDIRECTION_CODES:
            location = error.headers.get("Location") if error.headers else None
            raise BridgeTransportError(
                "redirected the request to another address",
                _shortened(str(location or error)),
            ) from None

        raise BridgeTransportError(
            f"answered with HTTP status {error.code}", _error_body(error)
        ) from None
    except urllib.error.URLError as error:
        raise _transport_error(error.reason, timeout) from None
    except (OSError, http.client.HTTPException, ValueError) as error:
        # A ValueError comes from an address urllib cannot use, such as a
        # malformed proxy of the environment.
        raise _transport_error(error, timeout) from None

    if len(raw) > MAX_ANSWER_BYTES:
        raise BridgeAnswerError(
            f"gave an answer larger than {MAX_ANSWER_BYTES // 1024} KiB"
        )

    too_deep = f"gave an answer nested deeper than {MAX_ANSWER_DEPTH} levels"

    try:
        answer = json.loads(
            raw.decode("utf-8"),
            parse_constant=_refuse_constant,
            parse_float=_finite_float,
        )
    except RecursionError:
        raise BridgeAnswerError(too_deep) from None
    except ValueError as error:
        raise BridgeAnswerError(
            "gave an answer that is not JSON", _shortened(str(error))
        ) from None

    if _depth(answer) > MAX_ANSWER_DEPTH:
        raise BridgeAnswerError(too_deep)

    return answer


_REDIRECTION_CODES = frozenset({301, 302, 303, 307, 308})


class _NoRedirection(urllib.request.HTTPRedirectHandler):
    """Follow no redirection: its HTTP status reaches the caller as an error.

    The address of the redirection is not even parsed, so a malformed one
    cannot raise here; the caller reports it as it was written.
    """

    def http_error_302(self, req, fp, code, msg, headers):
        """Decline the redirection, so that no second request is sent."""
        return None

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


def _require_https(url: str) -> None:
    """Refuse an address that does not use HTTPS or names no host."""
    parts = urllib.parse.urlsplit(url)

    if parts.scheme != "https" or not parts.hostname:
        raise ValueError(
            f"Calibration services are asked over HTTPS only, not at {url!r}."
        )


def _depth(value: Any) -> int:
    """Return the nesting depth of the lists and objects of a decoded answer."""
    deepest = 0
    pending = [(value, 1)]

    while pending:
        item, level = pending.pop()

        if not isinstance(item, (list, dict)):
            continue

        deepest = max(deepest, level)

        if deepest > MAX_ANSWER_DEPTH:
            break

        children = item.values() if isinstance(item, dict) else item
        pending.extend((child, level + 1) for child in children)

    return deepest


def _transport_error(reason: Any, timeout: float) -> BridgeTransportError:
    """Return the transport error, with its stable reason, for a cause."""
    detail = _shortened(str(reason) or type(reason).__name__)

    if isinstance(reason, ssl.SSLCertVerificationError):
        return BridgeTransportError(
            "could not prove its identity: its certificate was not verified", detail
        )

    if isinstance(reason, TimeoutError):
        return BridgeTransportError(f"did not answer within {timeout:g} s", detail)

    if isinstance(reason, http.client.RemoteDisconnected):
        return BridgeTransportError("closed the connection without answering", detail)

    if isinstance(reason, http.client.IncompleteRead):
        return BridgeTransportError("broke the connection while answering", detail)

    if isinstance(reason, http.client.HTTPException):
        return BridgeTransportError("did not give a valid HTTP answer", detail)

    return BridgeTransportError("could not be reached", detail)


def _error_body(error: urllib.error.HTTPError) -> str:
    """Return the start of the body of an HTTP error, or the error itself."""
    try:
        text = error.read(200).decode("utf-8", "replace").strip()
    except (OSError, http.client.HTTPException):
        text = ""

    return text or str(error)


def _refuse_constant(name: str) -> float:
    """Refuse NaN and the infinities, which are not numbers of JSON."""
    raise ValueError(f"{name} is not a number of JSON")


def _finite_float(text: str) -> float:
    """Decode a JSON number, refusing one that overflows to infinity."""
    number = float(text)

    if not math.isfinite(number):
        raise ValueError(f"{_shortened(text)} is not a finite number")

    return number


def _shortened(text: str, limit: int = 120) -> str:
    """Return a text cut to ``limit`` characters, for a sentence or a record."""
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _shown(value: Any) -> str:
    """Return the representation of a header or answer value in a sentence."""
    return _shortened(repr(value), 40)


def _user_agent() -> str:
    """Return the User-Agent of the requests of BayMorph."""
    from baymorph import __version__

    return f"BayMorph/{__version__}"


def _utc_now() -> str:
    """Return the present time in UTC, to the second."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_MJD_ORIGIN = date(1858, 11, 17)
"""Calendar date of the Modified Julian Date 0."""


def _calendar_date(mjd: float) -> date:
    """Return the calendar date of a Modified Julian Date."""
    return _MJD_ORIGIN + timedelta(days=math.floor(mjd))


def _date_text(mjd: float) -> str:
    """Return the calendar date of a Modified Julian Date, or the MJD itself
    when it lies beyond the calendar of Python."""
    try:
        return _calendar_date(mjd).isoformat()
    except OverflowError:
        return f"MJD {mjd:.4f}"


ACS_ENDPOINT = (
    "https://vtbopx9sf3.execute-api.us-east-1.amazonaws.com"
    "/main/single-filter-resource"
)
"""Address that acstools 3.8.2 posts a query for one filter to."""

ACS_DATES = (date(2002, 3, 7), date(2029, 12, 31))
"""First and last dates that the ACS Zeropoint Calculator serves, as acstools
3.8.2 checks them: the installation of ACS and the end of its table."""

ACS_FILTERS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "WFC": (
            "F435W", "F475W", "F502N", "F550M", "F555W", "F606W", "F625W",
            "F658N", "F660N", "F775W", "F814W", "F850LP", "F892N",
        ),
        "HRC": (
            "F220W", "F250W", "F330W", "F344N", "F435W", "F475W", "F502N",
            "F550M", "F555W", "F606W", "F625W", "F658N", "F660N", "F775W",
            "F814W", "F850LP", "F892N",
        ),
    }
)  # fmt: skip
"""Filters of the two CCD detectors for which acstools 3.8.2 asks the
calculator."""

ACS_MAX_SPAN_DAYS = 365.25
"""Longest observation, in days, served with the zeropoint of its middle
date."""

ACS_EPOCH_TOLERANCE_MAG = 0.002
"""Largest difference, in magnitudes, between the zeropoint of the middle
date of an observation and those of its first and last dates. The calculator
gave for WFC F814W 25.948 on 2006-06-20 and 25.932 on 2006-07-20, a step of
0.016 mag within a month, against 25.952 on 2004-04-15 and 25.931 on
2007-01-22: an observation over a step has no single zeropoint. If the
zeropoint changes monotonically within the observation, every exposure lies
within this tolerance of the middle date."""

COVERAGE_TOLERANCE_S = 1.0
"""Rounding allowed, in seconds, when the dates must cover the exposure time."""

ANSWER_TOLERANCE_MAG = 0.002
"""Largest difference, in magnitudes, between a zeropoint of the answer and
the one its PHOTFLAM and pivot wavelength give."""


class AcsZeropointBridge(CalibrationBridge):
    """The ACS Zeropoint Calculator of STScI, for drizzled ACS images.

    The sensitivity of ACS changes with time, and STScI gives its zeropoints
    through the calculator, not as a table. For an image of the WFC or HRC
    detector drizzled to electrons per second, the bridge asks for the AB
    zeropoint of its detector and filter on the calendar date of the middle
    of the observation, the mean of ``EXPSTART`` and ``EXPEND``, when these
    dates span at most ``ACS_MAX_SPAN_DAYS`` and cover the exposure time.
    When the observation runs over several dates, it also asks for the
    zeropoints of its first and last dates, and refuses the image when one
    of them differs from that of the middle date by more than
    ``ACS_EPOCH_TOLERANCE_MAG``. It accepts an answer only when it holds one
    row for that detector and filter, its AB zeropoint agrees with the one
    its PHOTFLAM and pivot wavelength give, and its ST zeropoint with the one
    its PHOTFLAM gives, within ``ANSWER_TOLERANCE_MAG``.
    """

    name = "stsci_acs_zeropoint"
    title = "ACS Zeropoint Calculator of STScI"
    service = "the calculator"
    endpoint = ACS_ENDPOINT
    supplies = ("zeropoint", "photometric_system")
    reads = (
        "telescope",
        "instrument",
        "detector",
        "filter",
        "image_unit",
        "exposure_time",
        "observation_start",
        "observation_end",
        "zeropoint",
    )
    fallback = "the flux priors follow the rule without a zeropoint"
    references = (
        "STScI, ACS zeropoints: the zeropoints depend on time and are given by "
        "the ACS Zeropoints Calculator, https://acszeropoints.stsci.edu, or "
        "the acszpt module of acstools; ZP_AB = -2.5 log10(PHOTFLAM) - 5 "
        "log10(PHOTPLAM) - 2.408 and ZP_ST = -2.5 log10(PHOTFLAM) - 21.10: "
        "https://www.stsci.edu/hst/instrumentation/acs/data-analysis/zeropoints",
        "STScI, acstools 3.8.2, acszpt: the address, the body (date, "
        "detector, filter), the filters of each detector and the dates "
        "2002-03-07 to 2029-12-31: https://github.com/spacetelescope/acstools/"
        "blob/3.8.2/acstools/acszpt.py",
    )

    def serves(self, reading: HeaderReading) -> bool:
        """Return whether the image is an HST/ACS image."""
        return _text(reading["telescope"]) == "HST" and (
            _text(reading["instrument"]) == "ACS"
        )

    def plan(self, reading: HeaderReading) -> BridgeQuery:
        """Return the request for the zeropoint of the middle date."""
        detector = _text(reading["detector"])

        if detector is None:
            raise BridgeRefusal(_unusable(reading["detector"], "DETECTOR"))

        if detector not in ACS_FILTERS:
            raise BridgeRefusal(
                "the bridge serves the detectors "
                + " and ".join(ACS_FILTERS)
                + f", and the detector is {_shown(reading['detector'].value)} "
                f"({reading['detector'].keyword})"
            )

        filter_quantity = reading["filter"]
        filter_name = _text(filter_quantity)

        if filter_name is None:
            raise BridgeRefusal(_unusable(filter_quantity, "FILTER, FILTER1, FILTER2"))

        if filter_name not in ACS_FILTERS[detector]:
            raise BridgeRefusal(
                f"the calculator gives {detector} zeropoints for "
                + ", ".join(ACS_FILTERS[detector])
                + f", and the filter is {_shown(filter_quantity.value)} "
                f"({filter_quantity.keyword})"
            )

        unit = reading["image_unit"]

        if not unit.usable or canonical_unit(unit.value) != "electron/s":
            shown = unit.value if unit.usable else unit.raw_value
            raise BridgeRefusal(
                "the calculator gives zeropoints for count rates in electrons "
                f"per second, and the image unit is {_shown(shown)} ({unit.status})"
            )

        if "hst_drizzle" not in reading.conventions:
            raise BridgeRefusal(
                "the zeropoints of the calculator hold for images drizzled to "
                "count rates, which D001OUUN = 'cps' shows, and the header does "
                "not show it; flt and flc images need the pixel area map first"
            )

        start, end = reading["observation_start"], reading["observation_end"]

        if not all(item.usable and item.unit == "MJD" for item in (start, end)):
            raise BridgeRefusal(
                "the header gives no usable observation dates (EXPSTART "
                f"{start.status}, EXPEND {end.status})"
            )

        first, last = float(start.value), float(end.value)

        if last < first:
            raise BridgeRefusal(
                f"the observation ends (EXPEND {last:.4f}) before it starts "
                f"(EXPSTART {first:.4f})"
            )

        exposure = reading["exposure_time"]

        if not exposure.usable or exposure.unit != "s":
            raise BridgeRefusal(
                f"the exposure time is {exposure.status} (EXPTIME), so the "
                "advisor cannot check that EXPSTART and EXPEND cover every "
                "exposure"
            )

        exposure_s = float(exposure.value)
        span_days = last - first

        if exposure_s > span_days * 86400.0 + COVERAGE_TOLERANCE_S:
            source = exposure.keyword or "data.calibration"
            raise BridgeRefusal(
                f"the exposure time, {exposure_s:g} s ({source}), exceeds the "
                f"{span_days * 86400.0:.0f} s between EXPSTART and EXPEND, so "
                "these dates do not cover every exposure of the image"
            )

        if span_days > ACS_MAX_SPAN_DAYS:
            raise BridgeRefusal(
                f"the observation spans {span_days:.1f} days (EXPSTART "
                f"{first:.4f} to EXPEND {last:.4f} MJD), more than "
                f"{ACS_MAX_SPAN_DAYS:g}, so it combines epochs whose zeropoints "
                "differ"
            )

        middle = 0.5 * (first + last)
        served = [(item - _MJD_ORIGIN).days for item in ACS_DATES]

        if not all(served[0] <= math.floor(mjd) <= served[1] for mjd in (first, last)):
            raise BridgeRefusal(
                f"the observation, {_date_text(first)} to {_date_text(last)}, "
                "lies partly or wholly outside the dates the calculator "
                f"serves, {ACS_DATES[0].isoformat()} to {ACS_DATES[1].isoformat()}"
            )

        day = _calendar_date(middle).isoformat()
        check_days = sorted(
            {_calendar_date(first).isoformat(), _calendar_date(last).isoformat()}
            - {day}
        )
        body = {"date": day, "detector": detector, "filter": filter_name}
        checked = (
            " and on " + " and ".join(check_days) + " to check that it holds for "
            "the whole observation"
            if check_days
            else ""
        )

        return BridgeQuery(
            body=body,
            subject=(f"the AB zeropoint of {detector} {filter_name} on {day}{checked}"),
            basis=(
                f"{detector}, {filter_name} ({filter_quantity.keyword}) and "
                f"{day}, the date of the middle of the observation "
                f"(EXPSTART {first:.4f} to EXPEND {last:.4f} MJD, "
                f"{span_days:.3g} days, exposure time {exposure_s:g} s)"
                + (", and its first and last dates" if check_days else "")
            ),
            rule={
                "date_of": "middle of observation_start and observation_end",
                "middle_mjd": round(middle, 8),
                "span_days": round(span_days, 8),
                "max_span_days": ACS_MAX_SPAN_DAYS,
                "exposure_time_s": exposure_s,
                "coverage_tolerance_s": COVERAGE_TOLERANCE_S,
                "date": day,
                "check_dates": check_days,
                "epoch_tolerance_mag": ACS_EPOCH_TOLERANCE_MAG,
                "dates_served": [item.isoformat() for item in ACS_DATES],
            },
            check_bodies=tuple({**body, "date": other} for other in check_days),
        )

    def interpret(self, query: BridgeQuery, answers: Sequence[Any]) -> BridgeAnswer:
        """Return the AB zeropoint of the middle date, after checking it.

        Every answer is checked on its own; the zeropoints of the first and
        last dates must then agree with that of the middle date.
        """
        bodies = (query.body, *query.check_bodies)

        if len(answers) != len(bodies):
            raise RuntimeError(
                f"The ACS bridge needs {len(bodies)} answers and received "
                f"{len(answers)}."
            )

        read = [_read_acs_answer(body, answer) for body, answer in zip(bodies, answers)]
        middle = read[0]
        checks = {
            "ab_from_photflam_and_pivot": round(middle["ab_check"], 5),
            "st_from_photflam": round(middle["st_check"], 5),
            "tolerance_mag": ANSWER_TOLERANCE_MAG,
        }
        statement = (
            f"gave the AB zeropoint {middle['ab']:g}; its PHOTFLAM "
            f"{middle['photflam']:.5g} and pivot wavelength {middle['photlam']:g} "
            f"A give {middle['ab_check']:.5f}, and its ST zeropoint "
            f"{middle['st']:g} agrees with the {middle['st_check']:.5f} of its "
            f"PHOTFLAM, both within {ANSWER_TOLERANCE_MAG:g} mag"
        )

        if query.check_bodies:
            others = {
                body["date"]: item["ab"]
                for body, item in zip(query.check_bodies, read[1:])
            }
            given = " and ".join(
                f"{value:g} on {date}" for date, value in others.items()
            )
            worst = max(abs(value - middle["ab"]) for value in others.values())

            if worst > ACS_EPOCH_TOLERANCE_MAG + 1e-9:
                raise BridgeRefusal(
                    f"gave the AB zeropoint {middle['ab']:g} on "
                    f"{query.body['date']} and {given}, which differ by up to "
                    f"{worst:.3f} mag, more than {ACS_EPOCH_TOLERANCE_MAG:g}, so "
                    "no single zeropoint holds for the exposures of the "
                    "observation"
                )

            checks.update({f"ab_on_{date}": value for date, value in others.items()})
            checks["epoch_tolerance_mag"] = ACS_EPOCH_TOLERANCE_MAG
            statement += (
                f"; it gives {given}, within {ACS_EPOCH_TOLERANCE_MAG:g} mag of it"
            )

        return BridgeAnswer(
            values={"zeropoint": middle["ab"], "photometric_system": "AB"},
            checks=checks,
            statement=statement,
        )


def _read_acs_answer(body: Mapping[str, Any], answer: Any) -> dict[str, float]:
    """Return the checked numbers of one answer of the ACS calculator.

    Raises
    ------
    BridgeAnswerError
        If the answer cannot be read, is not the one asked for, or is not
        consistent with itself.
    """
    if not isinstance(answer, Mapping) or not all(
        isinstance(answer.get(key), list) for key in ("headers", "rows")
    ):
        raise _unreadable("it holds no list of headers and rows")

    headers, rows = answer["headers"], answer["rows"]
    required = ("Detector", "Filter", "PHOTLAM", "PHOTFLAM", "STmag", "ABmag")
    columns: dict[str, int] = {}

    for index, header in enumerate(headers):
        words = str(header).split()
        token = words[0] if words else ""
        name = "PHOTLAM" if token == "PHOTPLAM" else token

        if name in required and name in columns:
            raise _unreadable(f"it names the column {name} twice")

        columns.setdefault(name, index)

    absent = [name for name in required if name not in columns]

    if absent:
        raise _unreadable("it has no column " + ", ".join(absent))

    if len(rows) != 1:
        raise _unreadable(f"it holds {len(rows)} rows instead of one")

    row = rows[0]

    if not isinstance(row, list):
        raise _unreadable("its row is not a list of values")

    if len(row) != len(headers):
        raise _unreadable("its row and its headers differ in length")

    for name in ("Detector", "Filter"):
        asked = str(body[name.lower()])

        if str(row[columns[name]]).strip().upper() != asked.upper():
            raise _unreadable(
                f"its {name.lower()} is {_shown(row[columns[name]])} instead of "
                f"{asked!r}"
            )

    numbers = {
        name: _number(row[columns[name]], name)
        for name in ("PHOTLAM", "PHOTFLAM", "STmag", "ABmag")
    }

    if numbers["PHOTFLAM"] <= 0.0 or numbers["PHOTLAM"] <= 0.0:
        raise _unreadable("its PHOTFLAM and pivot wavelength must be positive")

    ab, st = numbers["ABmag"], numbers["STmag"]
    ab_check = (
        -2.5 * math.log10(numbers["PHOTFLAM"])
        - 5.0 * math.log10(numbers["PHOTLAM"])
        - 2.408
    )
    st_check = -2.5 * math.log10(numbers["PHOTFLAM"]) - 21.10

    if abs(ab - ab_check) > ANSWER_TOLERANCE_MAG:
        raise _unreadable(
            f"its AB zeropoint {ab:g} differs from the {ab_check:.5f} of its "
            f"PHOTFLAM and pivot wavelength by more than "
            f"{ANSWER_TOLERANCE_MAG:g} mag"
        )

    if abs(st - st_check) > ANSWER_TOLERANCE_MAG:
        raise _unreadable(
            f"its ST zeropoint {st:g} differs from the {st_check:.5f} of its "
            f"PHOTFLAM by more than {ANSWER_TOLERANCE_MAG:g} mag"
        )

    return {
        "ab": ab,
        "st": st,
        "ab_check": ab_check,
        "st_check": st_check,
        "photflam": numbers["PHOTFLAM"],
        "photlam": numbers["PHOTLAM"],
    }


def _text(quantity: HeaderQuantity) -> str | None:
    """Return a usable text value in upper case without spaces, or None."""
    if not quantity.usable or quantity.value is None:
        return None

    return "".join(str(quantity.value).split()).upper()


def _unusable(quantity: HeaderQuantity, keywords: str) -> str:
    """Say that a quantity the bridge needs is not usable."""
    label = quantity.name.replace("_", " ")

    if quantity.status == "missing":
        return f"the header does not give the {label} ({keywords})"

    return f"the {label} is {quantity.status} ({quantity.keyword or keywords})"


def _unreadable(check: str) -> BridgeAnswerError:
    """Return the error of an answer that fails a check."""
    return BridgeAnswerError(f"gave an answer the bridge cannot read: {check}")


def _number(value: Any, name: str) -> float:
    """Return a finite number of an answer."""
    if isinstance(value, bool):
        raise _unreadable(f"its {name} {_shown(value)} is not a number")

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise _unreadable(f"its {name} {_shown(value)} is not a number") from None

    if not math.isfinite(number):
        raise _unreadable(f"its {name} {_shown(value)} is not finite")

    return number


BRIDGES: tuple[CalibrationBridge, ...] = (AcsZeropointBridge(),)
"""The calibration bridges of BayMorph, in the order they are consulted."""


def _check_bridges(bridges: Sequence[CalibrationBridge]) -> None:
    """Refuse a registry of bridges that breaks the rules of this module."""
    names = [bridge.name for bridge in bridges]

    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Calibration bridges need unique, non-empty names.")

    readable = set(QUANTITIES) | set(DERIVED_QUANTITIES)

    for bridge in bridges:
        prefix = f"Calibration bridge {bridge.name}:"

        try:
            _require_https(bridge.endpoint)
        except ValueError:
            raise ValueError(f"{prefix} its endpoint must use HTTPS.") from None

        if not bridge.supplies or not set(bridge.supplies) <= set(
            DECLARABLE_QUANTITIES
        ):
            raise ValueError(f"{prefix} it must supply declarable quantities.")

        if bridge.supplies[0] not in QUANTITIES or not set(bridge.reads) <= readable:
            raise ValueError(
                f"{prefix} it must resolve and read quantities of the header "
                "reading."
            )

        if not bridge.references:
            raise ValueError(f"{prefix} it must cite its documentation.")


_check_bridges(BRIDGES)

__all__ = [
    "ACS_DATES",
    "ACS_ENDPOINT",
    "ACS_EPOCH_TOLERANCE_MAG",
    "ACS_FILTERS",
    "ACS_MAX_SPAN_DAYS",
    "ANSWER_TOLERANCE_MAG",
    "BRIDGES",
    "BRIDGE_STATUSES",
    "COVERAGE_TOLERANCE_S",
    "MAX_ANSWER_BYTES",
    "TIMEOUT_SECONDS",
    "AcsZeropointBridge",
    "BridgeAnswer",
    "BridgeAnswerError",
    "BridgeOutcome",
    "BridgeQuery",
    "BridgeRefusal",
    "BridgeTransportError",
    "CalibrationBridge",
    "bridged_declarations",
    "consult_bridges",
    "post_json",
]
