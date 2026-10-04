"""Observational metadata read from FITS headers through conventions (AD-049).

A convention is a data file that describes, for one instrument, survey or
pipeline, how it is recognized, the keyword and unit of each quantity BayMorph
reads, and the documentation it rests on. The conventions distributed with
BayMorph live in ``baymorph/resources/header_conventions``; a user may add
files of the same form.

Every quantity of a reading carries its provenance, the keyword and the
convention, and one status:

- ``explicit``: the header itself declares the value and, for a number, its
  unit;
- ``convention``: a matched convention documents the meaning and unit;
- ``derived``: computed from other header content, such as the WCS;
- ``ambiguous``: present, but its unit or meaning is not established;
- ``conflicting``: present under several keywords with different values;
- ``missing``: absent.

No unit is ever inferred from a value. Only the first three statuses give a
usable value; the others keep the raw header value for inspection only.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import yaml
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_scales

STATUSES = (
    "explicit",
    "convention",
    "derived",
    "ambiguous",
    "conflicting",
    "missing",
)
"""Statuses of a quantity read from a header, from the most to the least
certain."""

USABLE_STATUSES = frozenset({"explicit", "convention", "derived"})
"""Statuses whose value may be used without a declaration of the user."""

IMAGE_UNIT = "image"
"""Unit of a quantity expressed in the units of the image pixels (BUNIT)."""


@dataclass(frozen=True)
class QuantityDefinition:
    """One quantity that a convention may read.

    Parameters
    ----------
    name : str
        Identifier of the quantity.
    kind : {"string", "number", "integer"}
        Type of the value.
    description : str
        Meaning of the quantity.
    """

    name: str
    kind: str
    description: str


QUANTITIES: dict[str, QuantityDefinition] = {
    definition.name: definition
    for definition in (
        QuantityDefinition("telescope", "string", "Telescope."),
        QuantityDefinition("instrument", "string", "Instrument."),
        QuantityDefinition("filter", "string", "Filter."),
        QuantityDefinition(
            "pipeline", "string", "Software that produced the image."
        ),
        QuantityDefinition("image_unit", "string", "Unit of the pixel values."),
        QuantityDefinition(
            "exposure_time", "number", "Exposure time of the image."
        ),
        QuantityDefinition(
            "total_exposure_time",
            "number",
            "Sum of the exposure times of the combined exposures.",
        ),
        QuantityDefinition(
            "n_combined", "integer", "Number of combined exposures."
        ),
        QuantityDefinition("gain", "number", "Gain of the detector or coadd."),
        QuantityDefinition("read_noise", "number", "Read noise."),
        QuantityDefinition(
            "saturation", "number", "Level at which pixels saturate."
        ),
        QuantityDefinition("zeropoint", "number", "Photometric zeropoint."),
        QuantityDefinition(
            "zeropoint_error",
            "number",
            "One-sigma uncertainty of the photometric zeropoint.",
        ),
        QuantityDefinition(
            "photometric_system", "string", "Photometric system, AB or Vega."
        ),
        QuantityDefinition(
            "magnitude_limit",
            "number",
            "Five-sigma limiting magnitude for point sources.",
        ),
        QuantityDefinition(
            "psf_fwhm", "number", "Full width at half maximum of the PSF."
        ),
        QuantityDefinition(
            "sky_level",
            "number",
            "Sky level subtracted before the image reached BayMorph.",
        ),
        QuantityDefinition(
            "header_background",
            "number",
            "Background value recorded in the header, of unspecified meaning.",
        ),
        QuantityDefinition(
            "astrometric_error",
            "number",
            "Root-mean-square error of the astrometric solution.",
        ),
        QuantityDefinition(
            "resampling_kernel", "string", "Kernel used to resample the image."
        ),
        QuantityDefinition(
            "combine_type", "string", "Method that combined the exposures."
        ),
    )
}
"""Quantities that conventions may read, by name."""

DERIVED_QUANTITIES = ("pixel_scale",)
"""Quantities derived by BayMorph from the header rather than read from it."""

_UNIT_TOKENS: dict[str, frozenset[str]] = {
    "arcsec": frozenset({"arcsec", "arcsecond", "arcseconds", "asec"}),
    "pixels": frozenset({"pix", "pixel", "pixels", "px"}),
    "s": frozenset({"s", "sec", "second", "seconds"}),
}

_SPEC_KEYS = frozenset(
    {"keywords", "unit", "unit_keywords", "unit_from_comment", "units", "note"}
)
_CONVENTION_KEYS = frozenset(
    {"name", "title", "priority", "references", "match", "quantities"}
)
_RULE_KEYS = frozenset(
    {"keyword", "keyword_prefix", "present", "equals", "startswith", "contains"}
)
_RELATIVE_AGREEMENT = 1e-9


@dataclass(frozen=True)
class QuantitySpec:
    """How one convention reads one quantity.

    Parameters
    ----------
    keywords : tuple of str
        Keywords in priority order.
    unit : str, optional
        Unit that the convention documents. ``"image"`` means the units of the
        image pixels.
    unit_keywords : tuple of str
        Keywords whose values may declare the unit.
    unit_from_comment : bool
        Whether the comment of the keyword may declare the unit.
    units : tuple of str
        Units that a declaration may name, for ``unit_keywords`` and
        ``unit_from_comment``.
    note : str, optional
        What the documentation says about the keyword.
    """

    keywords: tuple[str, ...]
    unit: str | None = None
    unit_keywords: tuple[str, ...] = ()
    unit_from_comment: bool = False
    units: tuple[str, ...] = ()
    note: str | None = None


@dataclass(frozen=True)
class MatchRule:
    """One rule that recognizes a convention in a header."""

    keyword: str | None = None
    keyword_prefix: str | None = None
    present: bool = False
    equals: str | None = None
    startswith: str | None = None
    contains: str | None = None

    def matches(self, lookup: _HeaderLookup) -> bool:
        """Return whether the header satisfies the rule."""
        if self.keyword_prefix is not None:
            prefix = self.keyword_prefix.upper()
            return any(keyword.startswith(prefix) for keyword in lookup.keywords())

        found = lookup.find(self.keyword)

        if found is None:
            return False

        if self.present:
            return True

        text = str(found.value).strip().lower()

        if self.equals is not None:
            return text == self.equals.strip().lower()

        if self.startswith is not None:
            return text.startswith(self.startswith.strip().lower())

        return self.contains.strip().lower() in text


@dataclass(frozen=True)
class HeaderConvention:
    """A convention of FITS headers.

    Parameters
    ----------
    name : str
        Unique identifier.
    title : str
        Human-readable name.
    priority : int
        Order among the matched conventions: higher values are consulted
        first.
    references : tuple of str
        Documentation the convention rests on.
    rules : tuple of MatchRule, optional
        Rules of which any one recognizes the convention. ``None`` matches
        every header.
    quantities : mapping of str to QuantitySpec
        How the convention reads each quantity.
    source : str
        File the convention was loaded from.
    """

    name: str
    title: str
    priority: int
    references: tuple[str, ...]
    rules: tuple[MatchRule, ...] | None
    quantities: Mapping[str, QuantitySpec]
    source: str

    def matches(self, lookup: _HeaderLookup) -> bool:
        """Return whether the convention applies to a header."""
        if self.rules is None:
            return True

        return any(rule.matches(lookup) for rule in self.rules)


@dataclass(frozen=True)
class HeaderQuantity:
    """One quantity read from a header.

    Parameters
    ----------
    name : str
        Identifier of the quantity.
    status : str
        One of ``STATUSES``.
    value : object or None
        The value, present only when the status is usable.
    unit : str, optional
        Unit of the value. ``"image"`` means the units of the image pixels.
    keyword : str, optional
        Keyword the value was read from.
    source : {"image", "primary"}, optional
        Header that holds the keyword: the header of the image HDU, or the
        primary header inherited by an image stored in an extension.
    convention : str, optional
        Convention that read the value.
    raw_value : object or None
        The value as written in the header, kept for inspection whatever the
        status.
    note : str, optional
        Explanation of the status.
    candidates : tuple of (str, object)
        Every keyword of the convention found in the header, with its raw
        value, in priority order.
    """

    name: str
    status: str
    value: Any = None
    unit: str | None = None
    keyword: str | None = None
    source: str | None = None
    convention: str | None = None
    raw_value: Any = None
    note: str | None = None
    candidates: tuple[tuple[str, Any], ...] = ()

    @property
    def usable(self) -> bool:
        """Whether the value may be used without a declaration of the user."""
        return self.status in USABLE_STATUSES

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable description of the quantity."""
        return {
            "status": self.status,
            "value": _json_value(self.value),
            "unit": self.unit,
            "keyword": self.keyword,
            "source": self.source,
            "convention": self.convention,
            "raw_value": _json_value(self.raw_value),
            "note": self.note,
        }


@dataclass(frozen=True)
class HeaderReading:
    """The quantities of an observation read from its FITS header.

    Parameters
    ----------
    conventions : tuple of str
        Matched conventions, in the order they were consulted.
    quantities : mapping of str to HeaderQuantity
        Every quantity of ``QUANTITIES`` and ``DERIVED_QUANTITIES``.
    """

    conventions: tuple[str, ...]
    quantities: Mapping[str, HeaderQuantity]

    def __getitem__(self, name: str) -> HeaderQuantity:
        """Return one quantity by name."""
        return self.quantities[name]

    def value(self, name: str) -> Any:
        """Return the value of a quantity when its status is usable, else None."""
        quantity = self.quantities[name]

        return quantity.value if quantity.usable else None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable description of the reading."""
        return {
            "conventions": list(self.conventions),
            "quantities": {
                name: quantity.as_dict()
                for name, quantity in self.quantities.items()
            },
        }

    def report(self) -> str:
        """Return a table of the reading for a human reader."""
        rows = [("Quantity", "Value", "Unit", "Status", "Source")]

        for name, quantity in self.quantities.items():
            if quantity.status == "missing":
                continue

            shown = quantity.value if quantity.usable else quantity.raw_value
            source = quantity.keyword or "-"

            if quantity.convention is not None:
                source = f"{source} ({quantity.convention})"

            rows.append(
                (
                    name,
                    _format_value(shown),
                    quantity.unit or "-",
                    quantity.status,
                    source,
                )
            )

        widths = [max(len(row[index]) for row in rows) for index in range(5)]
        lines = [
            "Conventions: " + (", ".join(self.conventions) or "none"),
            "",
        ]
        lines.extend(
            "  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip()
            for row in rows
        )

        notes = [
            f"- {name}: {quantity.note}"
            for name, quantity in self.quantities.items()
            if quantity.note and quantity.status != "missing"
        ]

        if notes:
            lines.extend(["", "Notes", *notes])

        return "\n".join(lines)


@dataclass(frozen=True)
class _Found:
    """One keyword found in the image header or the inherited primary header."""

    keyword: str
    value: Any
    comment: str
    source: str


class _HeaderLookup:
    """Keyword lookup in an image header, then in its primary header."""

    def __init__(
        self,
        header: fits.Header | None,
        primary_header: fits.Header | None,
    ) -> None:
        self._headers = tuple(
            (source, candidate)
            for source, candidate in (
                ("image", header),
                ("primary", primary_header),
            )
            if candidate is not None
        )

    def find(self, keyword: str) -> _Found | None:
        """Return the first card with the keyword, or None."""
        for source, header in self._headers:
            if keyword in header:
                try:
                    comment = header.comments[keyword]
                except (KeyError, IndexError, TypeError):
                    comment = ""

                return _Found(
                    keyword=keyword,
                    value=header[keyword],
                    comment=str(comment or ""),
                    source=source,
                )

        return None

    def keywords(self) -> Iterable[str]:
        """Yield every keyword of the headers, in upper case."""
        for _, header in self._headers:
            for keyword in header.keys():
                yield str(keyword).upper()


def read_header(
    header: fits.Header | None,
    *,
    primary_header: fits.Header | None = None,
    conventions: Sequence[HeaderConvention] | None = None,
) -> HeaderReading:
    """Read the quantities of an observation from its FITS header.

    Parameters
    ----------
    header : astropy.io.fits.Header or None
        Header of the HDU that holds the image.
    primary_header : astropy.io.fits.Header, optional
        Primary header, when the image is stored in an extension. Its keywords
        are inherited where the image header lacks them.
    conventions : sequence of HeaderConvention, optional
        Conventions to consult. The conventions distributed with BayMorph are
        used by default.

    Returns
    -------
    HeaderReading
        Every quantity, with its provenance and status.
    """
    if conventions is None:
        conventions = load_header_conventions()

    lookup = _HeaderLookup(header, primary_header)
    matched = sorted(
        (convention for convention in conventions if convention.matches(lookup)),
        key=lambda convention: (-convention.priority, convention.name),
    )

    quantities: dict[str, HeaderQuantity] = {}

    for name, definition in QUANTITIES.items():
        quantities[name] = _read_quantity(definition, matched, lookup)

    quantities["pixel_scale"] = _read_pixel_scale_quantity(header)

    return HeaderReading(
        conventions=tuple(convention.name for convention in matched),
        quantities=quantities,
    )


def _read_quantity(
    definition: QuantityDefinition,
    conventions: Sequence[HeaderConvention],
    lookup: _HeaderLookup,
) -> HeaderQuantity:
    """Read one quantity through the first convention that finds it."""
    for convention in conventions:
        spec = convention.quantities.get(definition.name)

        if spec is None:
            continue

        found = [
            card
            for card in (lookup.find(keyword) for keyword in spec.keywords)
            if card is not None and not _is_blank(card.value)
        ]

        if not found:
            continue

        if definition.kind == "string":
            return _resolve_string(definition, spec, found, convention)

        return _resolve_number(definition, spec, found, convention, lookup)

    return HeaderQuantity(name=definition.name, status="missing")


def _resolve_string(
    definition: QuantityDefinition,
    spec: QuantitySpec,
    found: Sequence[_Found],
    convention: HeaderConvention,
) -> HeaderQuantity:
    """Resolve a string quantity: the header states it explicitly."""
    first = found[0]
    value = str(first.value).strip()
    distinct = {str(card.value).strip() for card in found}
    candidates = tuple((card.keyword, card.value) for card in found)

    if len(distinct) > 1:
        return HeaderQuantity(
            name=definition.name,
            status="conflicting",
            keyword=first.keyword,
            source=first.source,
            convention=convention.name,
            raw_value=value,
            note=_join_notes(
                spec.note,
                "Keywords disagree: "
                + ", ".join(f"{card.keyword}={card.value!r}" for card in found)
                + ".",
            ),
            candidates=candidates,
        )

    return HeaderQuantity(
        name=definition.name,
        status="explicit",
        value=value,
        keyword=first.keyword,
        source=first.source,
        convention=convention.name,
        raw_value=value,
        note=spec.note,
        candidates=candidates,
    )


def _resolve_number(
    definition: QuantityDefinition,
    spec: QuantitySpec,
    found: Sequence[_Found],
    convention: HeaderConvention,
    lookup: _HeaderLookup,
) -> HeaderQuantity:
    """Resolve a numerical quantity and the certainty of its unit."""
    candidates = tuple((card.keyword, card.value) for card in found)
    valid = [
        (card, number)
        for card, number in (
            (card, _finite_number(card.value, definition.kind)) for card in found
        )
        if number is not None
    ]

    if not valid:
        first = found[0]
        return HeaderQuantity(
            name=definition.name,
            status="ambiguous",
            keyword=first.keyword,
            source=first.source,
            convention=convention.name,
            raw_value=first.value,
            note=_join_notes(
                spec.note,
                f"{first.keyword} is not a finite "
                + ("integer." if definition.kind == "integer" else "number."),
            ),
            candidates=candidates,
        )

    first_card, first_number = valid[0]
    common = {
        "name": definition.name,
        "keyword": first_card.keyword,
        "source": first_card.source,
        "convention": convention.name,
        "raw_value": first_number,
        "candidates": candidates,
    }

    if any(not _agree(first_number, number) for _, number in valid[1:]):
        return HeaderQuantity(
            status="conflicting",
            note=_join_notes(
                spec.note,
                "Keywords disagree: "
                + ", ".join(f"{card.keyword}={number:.10g}" for card, number in valid)
                + ".",
            ),
            **common,
        )

    if definition.kind == "integer":
        # A count has no unit: the header states it as it is.
        return HeaderQuantity(
            status="explicit",
            value=first_number,
            note=spec.note,
            **common,
        )

    if spec.unit is not None:
        return HeaderQuantity(
            status="convention",
            value=first_number,
            unit=spec.unit,
            note=spec.note,
            **common,
        )

    if spec.unit_keywords or spec.unit_from_comment:
        declarations = _declared_units(spec, first_card, lookup)

        if len(declarations) == 1:
            return HeaderQuantity(
                status="explicit",
                value=first_number,
                unit=declarations.pop(),
                note=spec.note,
                **common,
            )

        reason = (
            "The header declares conflicting units."
            if declarations
            else "The header declares no unit."
        )

        return HeaderQuantity(
            status="ambiguous",
            note=_join_notes(spec.note, reason),
            **common,
        )

    return HeaderQuantity(
        status="ambiguous",
        note=_join_notes(spec.note, "The convention documents no unit."),
        **common,
    )


def _declared_units(
    spec: QuantitySpec,
    card: _Found,
    lookup: _HeaderLookup,
) -> set[str]:
    """Return the units that the header declares for one keyword."""
    declarations = set()
    texts = []

    for keyword in spec.unit_keywords:
        unit_card = lookup.find(keyword)

        if unit_card is not None:
            texts.append(str(unit_card.value))

    if spec.unit_from_comment:
        texts.append(card.comment)

    for text in texts:
        unit = normalize_declared_unit(text, allowed=spec.units)

        if unit is not None:
            declarations.add(unit)

    return declarations


def normalize_declared_unit(
    text: Any,
    *,
    allowed: Sequence[str],
) -> str | None:
    """Return the one unit among ``allowed`` that a text declares, or None.

    The text is split into alphabetic tokens; a unit is recognized when one
    of its tokens appears. A text naming no allowed unit, or several, declares
    none.

    Parameters
    ----------
    text : object
        Value of a unit keyword or comment of a card.
    allowed : sequence of str
        Units that may be recognized, among ``"arcsec"``, ``"pixels"`` and
        ``"s"``.

    Returns
    -------
    str or None
        The recognized unit.
    """
    if text is None:
        return None

    normalized = "".join(
        character.lower() if character.isalpha() else " " for character in str(text)
    )
    tokens = set(normalized.split())
    recognized = {unit for unit in allowed if tokens & _UNIT_TOKENS[unit]}

    if len(recognized) != 1:
        return None

    return recognized.pop()


def _read_pixel_scale_quantity(header: fits.Header | None) -> HeaderQuantity:
    """Derive the mean pixel scale from the celestial WCS."""
    scale_x, scale_y, scale_mean = read_pixel_scales(read_celestial_wcs(header))

    if scale_mean is None:
        return HeaderQuantity(name="pixel_scale", status="missing")

    return HeaderQuantity(
        name="pixel_scale",
        status="derived",
        value=scale_mean,
        unit="arcsec/pixel",
        keyword="WCS",
        source="image",
        raw_value=scale_mean,
        note=(
            f"Mean of the projected scales {scale_x:.6g} and {scale_y:.6g} "
            "arcsec per pixel along the two image axes."
        ),
    )


def read_celestial_wcs(header: fits.Header | None) -> WCS | None:
    """Construct a valid two-dimensional celestial WCS when possible.

    Parameters
    ----------
    header : astropy.io.fits.Header or None
        Header of the image.

    Returns
    -------
    astropy.wcs.WCS or None
        The celestial WCS, or None when the header holds none.
    """
    if header is None:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FITSFixedWarning)
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


def read_pixel_scales(
    wcs: WCS | None,
) -> tuple[float | None, float | None, float | None]:
    """Return the projected celestial scales in arcseconds per pixel.

    Parameters
    ----------
    wcs : astropy.wcs.WCS or None
        Celestial WCS.

    Returns
    -------
    tuple of float or None
        Scales along the first and second image axes and their mean, or
        three None values when the WCS gives no valid scale.
    """
    if wcs is None:
        return None, None, None

    try:
        scales_deg = np.asarray(proj_plane_pixel_scales(wcs), dtype=float)
    except Exception:
        return None, None, None

    if scales_deg.shape != (2,):
        return None, None, None

    scales_arcsec = np.abs(scales_deg) * 3600.0

    if not np.all(np.isfinite(scales_arcsec)) or np.any(scales_arcsec <= 0.0):
        return None, None, None

    return (
        float(scales_arcsec[0]),
        float(scales_arcsec[1]),
        float(np.mean(scales_arcsec)),
    )


def load_header_conventions(
    paths: Iterable[str | Path] = (),
) -> tuple[HeaderConvention, ...]:
    """Load the distributed conventions and, optionally, files of the user.

    Parameters
    ----------
    paths : iterable of str or pathlib.Path
        Convention files of the user, added to the distributed ones.

    Returns
    -------
    tuple of HeaderConvention
        The conventions, sorted by name.

    Raises
    ------
    ValueError
        If a file is malformed, names an unknown quantity or field, or
        repeats the name of another convention.
    """
    conventions = list(_distributed_conventions())

    for path in paths:
        path = Path(path).expanduser()
        conventions.append(
            _parse_convention(
                yaml.safe_load(path.read_text(encoding="utf-8")),
                source=str(path),
            )
        )

    names = [convention.name for convention in conventions]
    repeated = sorted({name for name in names if names.count(name) > 1})

    if repeated:
        raise ValueError(
            "Header conventions must have unique names; repeated: "
            + ", ".join(repeated)
            + "."
        )

    return tuple(sorted(conventions, key=lambda convention: convention.name))


@lru_cache(maxsize=1)
def _distributed_conventions() -> tuple[HeaderConvention, ...]:
    """Parse the conventions distributed with BayMorph, once per process."""
    directory = resources.files("baymorph.resources").joinpath("header_conventions")

    return tuple(
        _parse_convention(
            yaml.safe_load(entry.read_text(encoding="utf-8")),
            source=f"baymorph/resources/header_conventions/{entry.name}",
        )
        for entry in sorted(directory.iterdir(), key=lambda item: item.name)
        if entry.name.endswith(".yaml")
    )


def _parse_convention(document: Any, *, source: str) -> HeaderConvention:
    """Validate one convention document and build it."""
    if not isinstance(document, Mapping):
        raise ValueError(f"Header convention {source} must be a mapping.")

    unknown = sorted(set(document) - _CONVENTION_KEYS)
    missing = sorted(_CONVENTION_KEYS - set(document))

    if unknown or missing:
        raise ValueError(
            f"Header convention {source}: unknown fields {unknown}, "
            f"missing fields {missing}."
        )

    name = document["name"]

    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Header convention {source} needs a non-empty name.")

    priority = document["priority"]

    if isinstance(priority, bool) or not isinstance(priority, int):
        raise ValueError(f"Header convention {source}: priority must be an integer.")

    references = document["references"]

    if (
        not isinstance(references, list)
        or not references
        or not all(isinstance(item, str) and item.strip() for item in references)
    ):
        raise ValueError(
            f"Header convention {source} must cite at least one reference."
        )

    quantities = document["quantities"]

    if not isinstance(quantities, Mapping) or not quantities:
        raise ValueError(
            f"Header convention {source} must read at least one quantity."
        )

    specs = {}

    for quantity, spec in quantities.items():
        if quantity not in QUANTITIES:
            raise ValueError(
                f"Header convention {source} names an unknown quantity "
                f"{quantity!r}; known: {', '.join(sorted(QUANTITIES))}."
            )

        specs[quantity] = _parse_spec(spec, source=f"{source}: {quantity}")

    return HeaderConvention(
        name=name.strip(),
        title=str(document["title"]),
        priority=priority,
        references=tuple(str(item) for item in references),
        rules=_parse_rules(document["match"], source=source),
        quantities=MappingProxyType(specs),
        source=source,
    )


def _parse_rules(match: Any, *, source: str) -> tuple[MatchRule, ...] | None:
    """Validate the match rules of a convention."""
    if match == "always":
        return None

    if not isinstance(match, list) or not match:
        raise ValueError(
            f"Header convention {source}: match must be 'always' or a "
            "non-empty list of rules."
        )

    rules = []

    for rule in match:
        if not isinstance(rule, Mapping) or set(rule) - _RULE_KEYS:
            raise ValueError(f"Header convention {source}: malformed rule {rule!r}.")

        if "keyword_prefix" in rule:
            if set(rule) != {"keyword_prefix"}:
                raise ValueError(
                    f"Header convention {source}: a keyword_prefix rule takes "
                    "no other field."
                )
            rules.append(MatchRule(keyword_prefix=str(rule["keyword_prefix"])))
            continue

        tests = set(rule) & {"present", "equals", "startswith", "contains"}

        if "keyword" not in rule or len(tests) != 1:
            raise ValueError(
                f"Header convention {source}: a rule needs a keyword and exactly "
                "one of present, equals, startswith or contains."
            )

        if "present" in rule and rule["present"] is not True:
            raise ValueError(
                f"Header convention {source}: present must be true when given."
            )

        rules.append(
            MatchRule(
                keyword=str(rule["keyword"]),
                present="present" in rule,
                equals=_optional_text(rule.get("equals")),
                startswith=_optional_text(rule.get("startswith")),
                contains=_optional_text(rule.get("contains")),
            )
        )

    return tuple(rules)


def _parse_spec(spec: Any, *, source: str) -> QuantitySpec:
    """Validate how a convention reads one quantity."""
    if not isinstance(spec, Mapping) or set(spec) - _SPEC_KEYS:
        raise ValueError(
            f"Header convention {source}: unknown fields "
            f"{sorted(set(spec) - _SPEC_KEYS) if isinstance(spec, Mapping) else spec}."
        )

    keywords = spec.get("keywords")

    if (
        not isinstance(keywords, list)
        or not keywords
        or not all(isinstance(item, str) and item.strip() for item in keywords)
    ):
        raise ValueError(f"Header convention {source}: keywords must be listed.")

    units = tuple(spec.get("units", ()))
    unknown_units = sorted(set(units) - set(_UNIT_TOKENS))

    if unknown_units:
        raise ValueError(
            f"Header convention {source}: units {unknown_units} cannot be "
            f"recognized; known: {', '.join(sorted(_UNIT_TOKENS))}."
        )

    declares = bool(spec.get("unit_keywords")) or bool(spec.get("unit_from_comment"))

    if declares and not units:
        raise ValueError(
            f"Header convention {source}: a unit declared by the header needs "
            "the list of units it may name."
        )

    if declares and spec.get("unit") is not None:
        raise ValueError(
            f"Header convention {source}: a unit is either documented by the "
            "convention or declared by the header, not both."
        )

    return QuantitySpec(
        keywords=tuple(item.strip() for item in keywords),
        unit=_optional_text(spec.get("unit")),
        unit_keywords=tuple(str(item) for item in spec.get("unit_keywords", ())),
        unit_from_comment=bool(spec.get("unit_from_comment", False)),
        units=units,
        note=_optional_text(spec.get("note")),
    )


def _optional_text(value: Any) -> str | None:
    """Return a stripped string, or None."""
    if value is None:
        return None

    return str(value).strip()


def _is_blank(value: Any) -> bool:
    """Return whether a header value is empty."""
    return value is None or (isinstance(value, str) and not value.strip())


def _finite_number(value: Any, kind: str) -> float | int | None:
    """Return a finite number, or None for anything else."""
    if isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(number):
        return None

    if kind == "integer":
        if not number.is_integer():
            return None
        return int(number)

    return number


def _agree(first: float, second: float) -> bool:
    """Return whether two header numbers state the same value."""
    return math.isclose(first, second, rel_tol=_RELATIVE_AGREEMENT, abs_tol=0.0)


def _join_notes(*notes: str | None) -> str | None:
    """Join the notes that are present."""
    present = [note for note in notes if note]

    return " ".join(present) if present else None


def _json_value(value: Any) -> Any:
    """Return a JSON-serializable form of a header value."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    return str(value)


def _format_value(value: Any) -> str:
    """Format a value for the report."""
    if value is None:
        return "-"

    if isinstance(value, float):
        return f"{value:.6g}"

    return str(value)


__all__ = [
    "DERIVED_QUANTITIES",
    "IMAGE_UNIT",
    "QUANTITIES",
    "STATUSES",
    "USABLE_STATUSES",
    "HeaderConvention",
    "HeaderQuantity",
    "HeaderReading",
    "QuantityDefinition",
    "QuantitySpec",
    "load_header_conventions",
    "normalize_declared_unit",
    "read_celestial_wcs",
    "read_header",
    "read_pixel_scales",
]
