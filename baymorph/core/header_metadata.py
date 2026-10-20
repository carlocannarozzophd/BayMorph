"""Observational metadata read from FITS headers through conventions (AD-049).

A convention is a data file that describes, for one instrument, survey or
pipeline, how it is recognized, the keyword and unit of each quantity BayMorph
reads, and the documentation it rests on. The conventions distributed with
BayMorph live in ``baymorph/resources/header_conventions``; a user may add
files of the same form.

A convention reads a quantity from keywords, states its value as a constant
of the convention, derives it from keywords through a formula, or tabulates
it by a quantity read before, such as the zeropoints of a product by filter,
and it may require conditions on the header, such as the unit of the image,
for its reading to hold.

Every quantity of a reading carries its provenance, the keyword and the
convention, and one status:

- ``declared``: the configuration declares the value (AD-053), which takes
  the place of the reading of the header; that reading is kept beside it;
- ``explicit``: the header itself declares the value and, for a number, its
  unit;
- ``convention``: a matched convention documents the meaning and unit, or
  states the value;
- ``derived``: computed from other header content, such as the WCS or the
  keywords of a formula;
- ``ambiguous``: present, but its unit or meaning is not established;
- ``conflicting``: present under several keywords, or read by several
  conventions, with different values;
- ``missing``: absent.

No unit is ever inferred from a value. Only the first four statuses give a
usable value; the others keep the raw header value for inspection only.

A convention may require the value of a quantity read before, such as the
unit of the image, so that a declared unit reaches the quantities that
depend on it.

The first matched convention that finds a quantity decides it. When its
reading is usable, it is compared with the usable readings of the other
matched conventions, and a disagreement makes the quantity conflicting,
unless the deciding convention declares that it overrides the other one; the
value it overrides is then recorded in the note of the reading. Two zeropoints
agree when they differ by at most ``ZEROPOINT_AGREEMENT_MAG``, and the note
names the other one.
"""

from __future__ import annotations

import ast
import math
import operator
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
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
    "declared",
    "explicit",
    "convention",
    "derived",
    "ambiguous",
    "conflicting",
    "missing",
)
"""Statuses of a quantity read from a header, from the most to the least
certain."""

USABLE_STATUSES = frozenset({"declared", "explicit", "convention", "derived"})
"""Statuses whose value may be used."""

IMAGE_UNIT = "image"
"""Unit of a quantity expressed in the units of the image pixels (BUNIT)."""

DECLARABLE_QUANTITIES: Mapping[str, str | None] = MappingProxyType(
    {
        "image_unit": None,
        "zeropoint": "mag",
        "photometric_system": None,
        "exposure_time": "s",
        "gain": "electron/adu",
        "saturation": IMAGE_UNIT,
    }
)
"""Quantities that a configuration may declare in ``data.calibration``
(AD-053), with the unit of a declared value."""

PHOTOMETRIC_SYSTEMS = ("AB", "Vega")
"""Photometric systems that a declared zeropoint may name."""

LINEAR_ENCODING = "linear"
"""Encoding of pixel values that are linear in flux, the only one fitted."""


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
        QuantityDefinition(
            "detector", "string", "Detector or channel of the instrument."
        ),
        QuantityDefinition("filter", "string", "Filter."),
        QuantityDefinition("pipeline", "string", "Software that produced the image."),
        QuantityDefinition("image_unit", "string", "Unit of the pixel values."),
        QuantityDefinition("exposure_time", "number", "Exposure time of the image."),
        QuantityDefinition(
            "total_exposure_time",
            "number",
            "Sum of the exposure times of the combined exposures.",
        ),
        QuantityDefinition("n_combined", "integer", "Number of combined exposures."),
        QuantityDefinition(
            "observation_start", "number", "Start of the first exposure of the image."
        ),
        QuantityDefinition(
            "observation_end", "number", "End of the last exposure of the image."
        ),
        QuantityDefinition("gain", "number", "Gain of the detector or coadd."),
        QuantityDefinition("read_noise", "number", "Read noise."),
        QuantityDefinition("saturation", "number", "Level at which pixels saturate."),
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
        QuantityDefinition(
            "pixel_encoding",
            "string",
            "Encoding of the pixel values: linear in flux, or a documented "
            "compression such as asinh.",
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
    {
        "keywords",
        "value",
        "formula",
        "requires",
        "unit",
        "unit_keywords",
        "unit_from_comment",
        "units",
        "note",
        "overrides",
        "ignore_prefixes",
        "by",
        "table",
    }
)
_FORMULA_FUNCTIONS: Mapping[str, Callable[[float], float]] = MappingProxyType(
    {"log10": math.log10, "ln": math.log, "exp": math.exp, "sqrt": math.sqrt}
)
_FORMULA_OPERATORS: Mapping[type, Callable[[float, float], float]] = MappingProxyType(
    {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }
)
_CONVENTION_KEYS = frozenset(
    {"name", "title", "priority", "references", "match", "quantities"}
)
_RULE_KEYS = frozenset(
    {
        "keyword",
        "keyword_prefix",
        "quantity",
        "present",
        "equals",
        "startswith",
        "contains",
    }
)
_RELATIVE_AGREEMENT = 1e-9

ZEROPOINT_AGREEMENT_MAG = 0.001
"""Largest difference, in magnitudes, between two readings of the zeropoint
by different conventions that agree (AD-054); 0.001 mag changes a flux by
0.09 percent."""


@dataclass(frozen=True)
class Formula:
    """An arithmetic expression over header keywords.

    The expression may use numbers, header keywords as names, the operators
    ``+``, ``-``, ``*``, ``/`` and ``**``, parentheses, and the functions
    ``log10``, ``ln``, ``exp`` and ``sqrt``. A keyword can enter an
    expression only when its name is a valid identifier.

    Parameters
    ----------
    text : str
        The expression as the convention writes it.
    keywords : tuple of str
        Keywords the expression reads, in order of first appearance.
    """

    text: str
    keywords: tuple[str, ...]
    tree: ast.Expression = field(repr=False, compare=False)

    @classmethod
    def parse(cls, text: Any) -> Formula:
        """Parse and validate an expression.

        Parameters
        ----------
        text : str
            The expression.

        Returns
        -------
        Formula
            The validated expression.

        Raises
        ------
        ValueError
            If the expression is empty, reads no keyword, or uses anything
            beyond numbers, keywords, the arithmetic operators and the
            allowed functions.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("a formula must be a non-empty expression")

        keywords: list[str] = []

        try:
            tree = ast.parse(text.strip(), mode="eval")
            _check_formula_node(tree.body, keywords, text)
        except SyntaxError as error:
            raise ValueError(
                f"formula {text!r} is not an expression: {error.msg}"
            ) from None
        except RecursionError:
            raise ValueError(f"formula {text[:40]!r}... is nested too deeply") from None

        if not keywords:
            raise ValueError(
                f"formula {text!r} reads no keyword; state a constant as a value"
            )

        return cls(text=text.strip(), keywords=tuple(keywords), tree=tree)

    def evaluate(self, values: Mapping[str, float]) -> float | None:
        """Return the value of the expression, or None where it is undefined.

        Parameters
        ----------
        values : mapping of str to float
            Value of every keyword of the expression.

        Returns
        -------
        float or None
            The finite real value, or None when the expression is undefined
            for these values, such as the logarithm of a negative number.
        """
        try:
            result = _evaluate_formula_node(self.tree.body, values)
        except (ArithmeticError, TypeError, ValueError):
            return None

        if isinstance(result, complex) or not math.isfinite(result):
            return None

        return float(result)


def _check_formula_node(node: ast.AST, keywords: list[str], text: str) -> None:
    """Accept only the nodes a formula may use, collecting its keywords."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"formula {text!r} may hold only numbers as constants")
        return

    if isinstance(node, ast.Name):
        if node.id in _FORMULA_FUNCTIONS:
            raise ValueError(f"formula {text!r} uses the function {node.id} as a value")
        if node.id not in keywords:
            keywords.append(node.id)
        return

    if isinstance(node, ast.BinOp) and type(node.op) in _FORMULA_OPERATORS:
        _check_formula_node(node.left, keywords, text)
        _check_formula_node(node.right, keywords, text)
        return

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        _check_formula_node(node.operand, keywords, text)
        return

    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FORMULA_FUNCTIONS
        and len(node.args) == 1
        and not node.keywords
    ):
        _check_formula_node(node.args[0], keywords, text)
        return

    raise ValueError(
        f"formula {text!r} may use only numbers, keywords, the operators "
        "+ - * / **, parentheses and the functions " + ", ".join(_FORMULA_FUNCTIONS)
    )


def _evaluate_formula_node(node: ast.AST, values: Mapping[str, float]) -> float:
    """Evaluate a validated formula node."""
    if isinstance(node, ast.Constant):
        return float(node.value)

    if isinstance(node, ast.Name):
        return values[node.id]

    if isinstance(node, ast.BinOp):
        return _FORMULA_OPERATORS[type(node.op)](
            _evaluate_formula_node(node.left, values),
            _evaluate_formula_node(node.right, values),
        )

    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_formula_node(node.operand, values)
        return -operand if isinstance(node.op, ast.USub) else operand

    return _FORMULA_FUNCTIONS[node.func.id](
        _evaluate_formula_node(node.args[0], values)
    )


@dataclass(frozen=True)
class QuantitySpec:
    """How one convention reads one quantity.

    A quantity is read from ``keywords``, stated by ``value``, derived by
    ``formula``, or looked up in ``table`` by the value of the quantity
    ``by``. A value given with keywords applies only when one of them is
    present: they are the evidence the convention relies on.

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
    value : object, optional
        Value that the convention states.
    formula : Formula, optional
        Expression that derives the value from header keywords.
    requires : tuple of MatchRule
        Rules that the header must satisfy for the reading to hold, such as
        the unit of the image. When one fails, the quantity is ambiguous.
    overrides : tuple of str
        Conventions whose disagreeing reading this one supersedes, because
        its documentation corrects them, such as the image unit of an HST
        image drizzled to count rates, whose ``BUNIT`` may keep the unit of
        its inputs.
    ignore_prefixes : tuple of str
        Prefixes of keyword values that mean the keyword holds nothing, such
        as a ``CLEAR`` element of a filter wheel; such a value is treated as
        blank. The comparison ignores case.
    by : str, optional
        Text quantity read before, such as ``filter``, whose usable value
        selects the entry of ``table``.
    table : mapping, optional
        Values of the quantity by value of ``by``. The lookup ignores case and
        surrounding spaces, and compares image units in canonical form.
    """

    keywords: tuple[str, ...] = ()
    unit: str | None = None
    unit_keywords: tuple[str, ...] = ()
    unit_from_comment: bool = False
    units: tuple[str, ...] = ()
    note: str | None = None
    value: Any = None
    formula: Formula | None = None
    requires: tuple[MatchRule, ...] = ()
    overrides: tuple[str, ...] = ()
    ignore_prefixes: tuple[str, ...] = ()
    by: str | None = None
    table: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class MatchRule:
    """One rule on a header, or on a quantity resolved before.

    A rule with ``quantity`` tests the usable value of a quantity already
    declared or read, instead of a keyword, and appears only among the
    requirements of a later quantity. The unit of the image is compared in
    the form of :func:`canonical_unit`, so ``ELECTRONS/S`` and
    ``electron/s`` agree.
    """

    keyword: str | None = None
    keyword_prefix: str | None = None
    quantity: str | None = None
    present: bool = False
    equals: str | None = None
    startswith: str | None = None
    contains: str | None = None

    def matches(
        self,
        lookup: _HeaderLookup,
        resolved: Mapping[str, HeaderQuantity] | None = None,
    ) -> bool:
        """Return whether the header, or a resolved quantity, satisfies the rule."""
        if self.keyword_prefix is not None:
            prefix = self.keyword_prefix.upper()
            return any(keyword.startswith(prefix) for keyword in lookup.keywords())

        if self.quantity is not None:
            quantity = (resolved or {}).get(self.quantity)

            if quantity is None or not quantity.usable:
                return False

            return self.present or self._test(quantity.value)

        found = lookup.find(self.keyword)

        if found is None:
            return False

        return self.present or self._test(found.value)

    def _test(self, value: Any) -> bool:
        """Return whether a value passes the test of the rule."""
        text = self._comparable(value)

        if self.equals is not None:
            return text == self._comparable(self.equals)

        if self.startswith is not None:
            return text.startswith(self._comparable(self.startswith))

        return self._comparable(self.contains) in text

    def _comparable(self, value: Any) -> str:
        """Return the form in which the rule compares a value."""
        if self.quantity == "image_unit":
            return canonical_unit(value)

        return str(value).strip().lower()

    def describe(self) -> str:
        """Describe the rule in words, for the notes of a reading."""
        if self.keyword_prefix is not None:
            return f"a keyword starts with {self.keyword_prefix!r}"

        subject = (
            self.keyword
            if self.quantity is None
            else "the " + self.quantity.replace("_", " ")
        )

        if self.present:
            return f"{subject} is present"

        if self.equals is not None:
            return f"{subject} is {self.equals!r}"

        if self.startswith is not None:
            return f"{subject} starts with {self.startswith!r}"

        return f"{subject} contains {self.contains!r}"

    def observe(
        self,
        lookup: _HeaderLookup,
        resolved: Mapping[str, HeaderQuantity] | None = None,
    ) -> str:
        """Describe what the header holds for the keyword or quantity of the rule."""
        if self.quantity is not None:
            subject = "the " + self.quantity.replace("_", " ")
            quantity = (resolved or {}).get(self.quantity)

            if quantity is None or quantity.status == "missing":
                return f"{subject} is not known"

            if not quantity.usable:
                return f"{subject} is {quantity.status}"

            return f"{subject} is {quantity.value!r} ({quantity.status})"

        if self.keyword is None:
            return f"no keyword starts with {self.keyword_prefix!r}"

        found = lookup.find(self.keyword)

        if found is None:
            return f"the header has no {self.keyword}"

        return f"the header has {self.keyword} = {found.value!r}"


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
        Keyword the value was read from; for a derived value, the keywords
        of its formula, separated by commas.
    source : {"image", "primary", "image, primary", "configuration"}, optional
        Header that holds the keyword: the header of the image HDU, or the
        primary header inherited by an image stored in an extension; both
        when the keywords of a formula come from both. ``"configuration"``
        for a declared value.
    convention : str, optional
        Convention that read the value.
    raw_value : object or None
        The value as written in the header, kept for inspection whatever the
        status. None for a value that a convention states or derives.
    note : str, optional
        Explanation of the status.
    candidates : tuple of (str, object)
        Every keyword of the convention found in the header, with its raw
        value, in priority order.
    header : HeaderQuantity, optional
        For a declared value, the reading of the header that it replaced.
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
    header: HeaderQuantity | None = None

    @property
    def usable(self) -> bool:
        """Whether the value may be used."""
        return self.status in USABLE_STATUSES

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable description of the quantity.

        A declared value carries the reading of the header it replaced under
        ``header``.
        """
        description = {
            "status": self.status,
            "value": _json_value(self.value),
            "unit": self.unit,
            "keyword": self.keyword,
            "source": self.source,
            "convention": self.convention,
            "raw_value": _json_value(self.raw_value),
            "note": self.note,
        }

        if self.header is not None:
            description["header"] = self.header.as_dict()

        return description


@dataclass(frozen=True)
class HeaderReading:
    """The quantities of an observation read from its FITS header.

    Parameters
    ----------
    conventions : tuple of str
        Matched conventions, in the order they were consulted.
    quantities : mapping of str to HeaderQuantity
        Every quantity of ``QUANTITIES`` and ``DERIVED_QUANTITIES``.
    celestial_wcs : astropy.wcs.WCS, optional
        Celestial WCS of the image header, from which the pixel scale is
        derived; None when the header holds none. It is not part of
        :meth:`as_dict`.
    """

    conventions: tuple[str, ...]
    quantities: Mapping[str, HeaderQuantity]
    celestial_wcs: WCS | None = field(default=None, compare=False, repr=False)

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
                name: quantity.as_dict() for name, quantity in self.quantities.items()
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

            if quantity.status == "declared":
                source = "data.calibration"
            elif quantity.convention is not None:
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
        lines = ["Conventions: " + (", ".join(self.conventions) or "none"), ""]
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
        self, header: fits.Header | None, primary_header: fits.Header | None
    ) -> None:
        self._headers = tuple(
            (source, candidate)
            for source, candidate in (("image", header), ("primary", primary_header))
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
    declarations: Mapping[str, Any] | None = None,
) -> HeaderReading:
    """Read the quantities of an observation from its FITS header.

    Quantities are read in the order of ``QUANTITIES``, so a requirement of a
    convention may test a quantity read or declared before. A declared value
    takes the place of the reading of the header, which it keeps under
    ``header`` (AD-053). Two consequences follow. A declared image unit that
    differs from a usable unit of the header leaves no header value in the
    units of the image established. A declared zeropoint takes its
    photometric system only from the declarations, and the zeropoint error
    of the header, which belongs to the zeropoint of the header, is set
    aside.

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
    declarations : mapping, optional
        Values of ``data.calibration``, validated by
        :func:`validate_declarations`.

    Returns
    -------
    HeaderReading
        Every quantity, with its provenance and status, and the celestial
        WCS of ``header``.

    Raises
    ------
    ValueError
        If the declarations are not valid.
    """
    if conventions is None:
        conventions = load_header_conventions()

    declared = validate_declarations(declarations)
    lookup = _HeaderLookup(header, primary_header)
    matched = sorted(
        (convention for convention in conventions if convention.matches(lookup)),
        key=lambda convention: (-convention.priority, convention.name),
    )

    quantities: dict[str, HeaderQuantity] = {}

    for name, definition in QUANTITIES.items():
        reading = _read_quantity(definition, matched, lookup, quantities)

        if name in declared:
            reading = _declared_quantity(name, declared[name], reading)
        elif name in _ZEROPOINT_QUALIFIERS and "zeropoint" in declared:
            reading = HeaderQuantity(
                name=name,
                status="missing",
                note=(
                    "The zeropoint is declared, and this quantity of the header "
                    "belongs to the zeropoint of the header."
                    + (
                        " Declare it in data.calibration.photometric_system."
                        if name == "photometric_system"
                        else ""
                    )
                ),
                header=reading,
            )

        quantities[name] = reading

    _set_aside_image_unit_values(quantities)

    celestial_wcs = read_celestial_wcs(header)
    quantities["pixel_scale"] = _read_pixel_scale_quantity(celestial_wcs)

    return HeaderReading(
        conventions=tuple(convention.name for convention in matched),
        quantities=quantities,
        celestial_wcs=celestial_wcs,
    )


_ZEROPOINT_QUALIFIERS = ("zeropoint_error", "photometric_system")
"""Quantities of the header that belong to the zeropoint of the header."""


def validate_declarations(declarations: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate the declarations of ``data.calibration`` (AD-053).

    Parameters
    ----------
    declarations : mapping or None
        Declared values by quantity: ``image_unit`` (text), ``zeropoint`` (a
        finite number, in magnitudes), ``photometric_system`` (AB or Vega,
        only together with ``zeropoint``), and the positive numbers
        ``exposure_time`` (s), ``gain`` (electrons per ADU of the detector)
        and ``saturation`` (in the units of the image).

    Returns
    -------
    dict
        The declarations in the order of ``DECLARABLE_QUANTITIES``, with
        numbers as floats and the photometric system written as in
        ``PHOTOMETRIC_SYSTEMS``.

    Raises
    ------
    ValueError
        If a quantity cannot be declared or its value is not valid.
    """
    if declarations is None:
        return {}

    if not isinstance(declarations, Mapping):
        raise ValueError(
            "data.calibration must map quantities to their declared values."
        )

    unknown = sorted(
        str(name) for name in set(declarations) - set(DECLARABLE_QUANTITIES)
    )

    if unknown:
        raise ValueError(
            f"data.calibration cannot declare {unknown}; it declares "
            + ", ".join(DECLARABLE_QUANTITIES)
            + "."
        )

    validated: dict[str, Any] = {}

    for name in DECLARABLE_QUANTITIES:
        if name not in declarations:
            continue

        value = declarations[name]
        key = f"data.calibration.{name}"

        if name == "image_unit":
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a non-empty unit, such as 'ADU/s'.")

            validated[name] = value.strip()
        elif name == "photometric_system":
            systems = {system.lower(): system for system in PHOTOMETRIC_SYSTEMS}

            if not isinstance(value, str) or value.strip().lower() not in systems:
                raise ValueError(
                    f"{key} must be one of " + ", ".join(PHOTOMETRIC_SYSTEMS) + "."
                )

            if "zeropoint" not in declarations:
                raise ValueError(
                    f"{key} qualifies a declared zeropoint; declare "
                    "data.calibration.zeropoint with it."
                )

            validated[name] = systems[value.strip().lower()]
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be a number.")

            number = float(value)

            if not math.isfinite(number):
                raise ValueError(f"{key} must be finite.")

            if name != "zeropoint" and number <= 0.0:
                raise ValueError(f"{key} must be positive.")

            validated[name] = number

    return validated


def canonical_unit(text: Any) -> str:
    """Return a unit written in one form, for comparing units.

    Case and spaces are ignored, ``electrons`` and ``e-`` read as
    ``electron``, and a second after a slash, written ``sec`` or ``second``,
    as ``s``; so ``ELECTRONS/S``, ``electrons / sec`` and ``e-/s`` agree.

    Parameters
    ----------
    text : object
        A unit as a header or a declaration writes it.

    Returns
    -------
    str
        The unit in canonical form.
    """
    unit = str(text).strip().lower().replace(" ", "")
    unit = unit.replace("electrons", "electron").replace("e-", "electron")

    for spelling in ("/seconds", "/second", "/sec"):
        unit = unit.replace(spelling, "/s")

    return unit


def _declared_quantity(name: str, value: Any, header: HeaderQuantity) -> HeaderQuantity:
    """Return a declared quantity, which keeps the reading of the header."""
    if header.status == "missing":
        was = "the header does not hold it"
    else:
        shown = header.value if header.usable else header.raw_value
        was = f"the header reading is {shown!r} ({header.status}"
        was += f", {header.keyword})" if header.keyword else ")"

    return HeaderQuantity(
        name=name,
        status="declared",
        value=value,
        unit=DECLARABLE_QUANTITIES[name],
        source="configuration",
        raw_value=header.raw_value if header.raw_value is not None else header.value,
        note=f"Declared in data.calibration; {was}.",
        header=header,
    )


def _set_aside_image_unit_values(quantities: dict[str, HeaderQuantity]) -> None:
    """Make ambiguous the header values in image units under another unit.

    A declared image unit that differs from a usable unit of the header says
    that the pixels are not in the unit the header wrote, so a value that the
    header gives in the units of the image is no longer established.
    """
    image_unit = quantities["image_unit"]

    if image_unit.status != "declared" or image_unit.header is None:
        return

    header_unit = image_unit.header

    if not header_unit.usable or canonical_unit(header_unit.value) == canonical_unit(
        image_unit.value
    ):
        return

    for name, quantity in quantities.items():
        if (
            quantity.status in ("explicit", "convention", "derived")
            and quantity.unit == IMAGE_UNIT
        ):
            quantities[name] = replace(
                quantity,
                status="ambiguous",
                value=None,
                raw_value=(
                    quantity.value if quantity.raw_value is None else quantity.raw_value
                ),
                note=_join_notes(
                    quantity.note,
                    f"The image unit is declared as {image_unit.value!r} while "
                    f"the header gives {header_unit.value!r}, so this value in "
                    "the units of the image is no longer established.",
                ),
            )


def ensure_linear_pixels(reading: HeaderReading, *, image: str) -> None:
    """Refuse an image whose pixel values are not linear in flux (AD-049).

    An image whose header declares no encoding is taken as linear, as
    BayMorph always has.

    Parameters
    ----------
    reading : HeaderReading
        Reading of the header of the image.
    image : str
        Description of the image for the error message.

    Raises
    ------
    ValueError
        If the reading states an encoding other than linear, or cannot
        establish the encoding its convention reads.
    """
    quantity = reading["pixel_encoding"]

    if quantity.status == "missing":
        return

    if quantity.usable and quantity.value == LINEAR_ENCODING:
        return

    declared = (
        f"declares the {quantity.value} encoding"
        if quantity.usable
        else "does not establish the encoding"
    )

    raise ValueError(
        f"BayMorph refuses {image}: its header {declared} of the pixel values "
        f"({quantity.keyword or 'no keyword'}, convention "
        f"{quantity.convention}), and BayMorph fits only pixel values linear "
        "in flux until a documented linearization is implemented (AD-049)."
        + ("" if quantity.note is None else f" {quantity.note}")
    )


def _read_quantity(
    definition: QuantityDefinition,
    conventions: Sequence[HeaderConvention],
    lookup: _HeaderLookup,
    resolved: Mapping[str, HeaderQuantity],
) -> HeaderQuantity:
    """Read one quantity through the conventions that find it.

    A convention finds a quantity when the header holds one of its keywords,
    or every keyword of its formula; a value it states without keywords is
    found as soon as the convention matches. The first convention that finds
    the quantity decides it, and its requirements decide whether the reading
    holds; a requirement may test a quantity in ``resolved``, read or
    declared before. A usable decision is compared with the usable reading
    of every later convention that finds the quantity: a value or unit that
    disagrees makes the quantity conflicting, unless the deciding convention
    overrides the other one, and then the overridden value is recorded in the
    note of the decision. An unusable decision is kept, so a convention that
    leaves a quantity ambiguous is not bypassed by a less specific one.
    """
    decision: HeaderQuantity | None = None
    overrides: tuple[str, ...] = ()
    close: list[HeaderQuantity] = []

    for convention in conventions:
        reading = _convention_reading(definition, convention, lookup, resolved)

        if reading is None:
            continue

        if decision is None:
            if not reading.usable:
                return reading

            decision = reading
            overrides = convention.quantities[definition.name].overrides
            continue

        if not reading.usable:
            continue

        if _readings_agree(definition, decision, reading):
            if not _readings_agree(definition, decision, reading, exact=True):
                close.append(reading)

            continue

        if convention.name not in overrides:
            return _conflicting_readings(definition, decision, reading)

        decision = replace(
            decision,
            note=_join_notes(
                decision.note,
                f"It overrides another reading: {_describe_reading(reading)}.",
            ),
        )

    if decision is None:
        return HeaderQuantity(name=definition.name, status="missing")

    if close:
        decision = replace(
            decision,
            note=_join_notes(
                decision.note,
                (
                    "Another reading agrees"
                    if len(close) == 1
                    else "Other readings agree"
                )
                + f" within {ZEROPOINT_AGREEMENT_MAG:g} mag: "
                + "; ".join(_describe_reading(item) for item in close)
                + ".",
            ),
        )

    return decision


def _convention_reading(
    definition: QuantityDefinition,
    convention: HeaderConvention,
    lookup: _HeaderLookup,
    resolved: Mapping[str, HeaderQuantity],
) -> HeaderQuantity | None:
    """Read one quantity through one convention, or None if it finds none."""
    spec = convention.quantities.get(definition.name)

    if spec is None:
        return None

    names = spec.keywords if spec.formula is None else spec.formula.keywords
    found = [
        card
        for card in (lookup.find(keyword) for keyword in names)
        if card is not None
        and not _is_blank(card.value)
        and not _is_ignored(card.value, spec.ignore_prefixes)
    ]

    if names and not found:
        return None

    if spec.formula is not None and len(found) < len(names):
        return None

    unmet = next(
        (rule for rule in spec.requires if not rule.matches(lookup, resolved)), None
    )

    if unmet is not None:
        return _resolve_unmet(
            definition, spec, found, convention, unmet, lookup, resolved
        )

    if spec.formula is not None:
        return _resolve_formula(definition, spec, found, convention)

    if spec.table is not None:
        return _resolve_table(definition, spec, convention, resolved)

    if spec.value is not None:
        return _resolve_value(definition, spec, found, convention)

    if definition.kind == "string":
        return _resolve_string(definition, spec, found, convention)

    return _resolve_number(definition, spec, found, convention, lookup)


def _table_key(value: Any, by: str) -> str:
    """Return the form in which a table compares the value of its key.

    An image unit is compared in the form of :func:`canonical_unit`, as every
    other comparison of units; any other text ignores case and surrounding
    spaces.
    """
    if by == "image_unit":
        return canonical_unit(value)

    return str(value).strip().upper()


def _resolve_table(
    definition: QuantityDefinition,
    spec: QuantitySpec,
    convention: HeaderConvention,
    resolved: Mapping[str, HeaderQuantity],
) -> HeaderQuantity:
    """Resolve a quantity that the convention tabulates by another quantity.

    The value rests on the keyword of the quantity that selects it, such as
    ``FILTER2`` for a zeropoint tabulated by filter.
    """
    subject = "the " + spec.by.replace("_", " ")
    key = resolved.get(spec.by)
    common = {
        "name": definition.name,
        "convention": convention.name,
        "keyword": None if key is None else key.keyword,
        "source": None if key is None else key.source,
    }

    if key is None or not key.usable:
        state = "not known" if key is None or key.status == "missing" else key.status

        return HeaderQuantity(
            status="ambiguous",
            note=_join_notes(
                spec.note,
                f"The convention tabulates it by {subject}, which is {state}.",
            ),
            **common,
        )

    entry = _table_key(key.value, spec.by)

    if entry not in spec.table:
        return HeaderQuantity(
            status="ambiguous",
            note=_join_notes(
                spec.note,
                f"The convention tabulates no value for {subject} {key.value!r}.",
            ),
            **common,
        )

    return HeaderQuantity(
        status="convention",
        value=spec.table[entry],
        unit=spec.unit,
        note=_join_notes(spec.note, f"Tabulated for {subject} {key.value!r}."),
        **common,
    )


def _is_ignored(value: Any, prefixes: Sequence[str]) -> bool:
    """Return whether a keyword value starts with a prefix meaning nothing."""
    text = str(value).strip().lower()

    return any(text.startswith(prefix.lower()) for prefix in prefixes)


def _readings_agree(
    definition: QuantityDefinition,
    first: HeaderQuantity,
    second: HeaderQuantity,
    *,
    exact: bool = False,
) -> bool:
    """Return whether two usable readings give the same value and unit.

    Two zeropoints agree within ``ZEROPOINT_AGREEMENT_MAG`` unless ``exact``.
    """
    if definition.kind == "string":
        compare = (
            canonical_unit
            if definition.name == "image_unit"
            else (lambda value: str(value).strip().lower())
        )
        same_value = compare(first.value) == compare(second.value)
    elif definition.name == "zeropoint" and not exact:
        # Two zeropoints written 0.001 mag apart differ by slightly more or
        # less in binary; the relative slack keeps them agreeing.
        difference = abs(float(first.value) - float(second.value))
        same_value = difference <= ZEROPOINT_AGREEMENT_MAG or math.isclose(
            difference, ZEROPOINT_AGREEMENT_MAG, rel_tol=_RELATIVE_AGREEMENT
        )
    else:
        same_value = _agree(float(first.value), float(second.value))

    units = [
        None if item.unit is None else canonical_unit(item.unit)
        for item in (first, second)
    ]

    return same_value and units[0] == units[1]


def _conflicting_readings(
    definition: QuantityDefinition, first: HeaderQuantity, second: HeaderQuantity
) -> HeaderQuantity:
    """Return a quantity on which two conventions disagree."""
    return HeaderQuantity(
        name=definition.name,
        status="conflicting",
        keyword=first.keyword,
        source=first.source,
        convention=first.convention,
        raw_value=first.value,
        note=_join_notes(
            first.note,
            "Conventions disagree: "
            + "; ".join(_describe_reading(item) for item in (first, second))
            + ".",
        ),
        candidates=first.candidates + second.candidates,
    )


def _describe_reading(quantity: HeaderQuantity) -> str:
    """Describe the value a convention gives, for the notes of a reading."""
    text = f"{quantity.convention} gives {_format_value(quantity.value)}"

    if quantity.unit == IMAGE_UNIT:
        text += " in the units of the image"
    elif quantity.unit:
        text += f" {quantity.unit}"

    if quantity.keyword:
        text += f" ({quantity.keyword})"

    return text


def _resolve_unmet(
    definition: QuantityDefinition,
    spec: QuantitySpec,
    found: Sequence[_Found],
    convention: HeaderConvention,
    rule: MatchRule,
    lookup: _HeaderLookup,
    resolved: Mapping[str, HeaderQuantity],
) -> HeaderQuantity:
    """Report a quantity whose convention does not hold for this header."""
    raw_value = None

    if found and spec.formula is None and spec.value is None:
        raw_value = found[0].value

        if definition.kind == "string":
            raw_value = str(raw_value).strip()
        else:
            number = _finite_number(raw_value, definition.kind)
            raw_value = raw_value if number is None else number

    return HeaderQuantity(
        name=definition.name,
        status="ambiguous",
        keyword=_keywords_of(found, spec),
        source=_sources_of(found, spec),
        convention=convention.name,
        raw_value=raw_value,
        note=_join_notes(
            spec.note,
            f"The convention reads it only when {rule.describe()}, and "
            f"{rule.observe(lookup, resolved)}.",
        ),
        candidates=tuple((card.keyword, card.value) for card in found),
    )


def _resolve_value(
    definition: QuantityDefinition,
    spec: QuantitySpec,
    found: Sequence[_Found],
    convention: HeaderConvention,
) -> HeaderQuantity:
    """Resolve a quantity whose value the convention states."""
    return HeaderQuantity(
        name=definition.name,
        status="convention",
        value=spec.value,
        unit=spec.unit,
        keyword=_keywords_of(found, spec),
        source=_sources_of(found, spec),
        convention=convention.name,
        note=spec.note,
        candidates=tuple((card.keyword, card.value) for card in found),
    )


def _resolve_formula(
    definition: QuantityDefinition,
    spec: QuantitySpec,
    found: Sequence[_Found],
    convention: HeaderConvention,
) -> HeaderQuantity:
    """Derive a number through the formula of a convention."""
    formula = spec.formula
    common = {
        "name": definition.name,
        "keyword": _keywords_of(found, spec),
        "source": _sources_of(found, spec),
        "convention": convention.name,
        "candidates": tuple((card.keyword, card.value) for card in found),
    }
    values = {}

    for card in found:
        number = _finite_number(card.value, "number")

        if number is None:
            return HeaderQuantity(
                status="ambiguous",
                note=_join_notes(spec.note, f"{card.keyword} is not a finite number."),
                **common,
            )

        values[card.keyword] = number

    result = formula.evaluate(values)

    if result is None:
        return HeaderQuantity(
            status="ambiguous",
            note=_join_notes(
                spec.note,
                f"The formula {formula.text} is not defined for these values.",
            ),
            **common,
        )

    return HeaderQuantity(
        status="derived",
        value=result,
        unit=spec.unit,
        note=_join_notes(spec.note, f"Derived as {formula.text}."),
        **common,
    )


def _keywords_of(found: Sequence[_Found], spec: QuantitySpec) -> str | None:
    """Return the keyword a value rests on, or those of its formula."""
    if not found:
        return None

    if spec.formula is not None:
        return ", ".join(card.keyword for card in found)

    return found[0].keyword


def _sources_of(found: Sequence[_Found], spec: QuantitySpec) -> str | None:
    """Return the header that holds the keywords a value rests on."""
    if not found:
        return None

    if spec.formula is None:
        return found[0].source

    sources = {card.source for card in found}

    return ", ".join(source for source in ("image", "primary") if source in sources)


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
            status="explicit", value=first_number, note=spec.note, **common
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
            status="ambiguous", note=_join_notes(spec.note, reason), **common
        )

    return HeaderQuantity(
        status="ambiguous",
        note=_join_notes(spec.note, "The convention documents no unit."),
        **common,
    )


def _declared_units(
    spec: QuantitySpec, card: _Found, lookup: _HeaderLookup
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


def normalize_declared_unit(text: Any, *, allowed: Sequence[str]) -> str | None:
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


def _read_pixel_scale_quantity(celestial_wcs: WCS | None) -> HeaderQuantity:
    """Derive the mean pixel scale from the celestial WCS."""
    scale_x, scale_y, scale_mean = read_pixel_scales(celestial_wcs)

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
                yaml.safe_load(path.read_text(encoding="utf-8")), source=str(path)
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
        raise ValueError(f"Header convention {source} must read at least one quantity.")

    specs = {}

    for quantity, spec in quantities.items():
        if quantity not in QUANTITIES:
            raise ValueError(
                f"Header convention {source} names an unknown quantity "
                f"{quantity!r}; known: {', '.join(sorted(QUANTITIES))}."
            )

        specs[quantity] = _parse_spec(
            spec, definition=QUANTITIES[quantity], source=f"{source}: {quantity}"
        )

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

    return tuple(_parse_rule(rule, source=source) for rule in match)


def _parse_rule(
    rule: Any, *, source: str, before: Sequence[str] | None = None
) -> MatchRule:
    """Validate one rule on a header, of a match or of a requirement.

    ``before`` lists the quantities read before the one whose requirement
    the rule is; only such a requirement may test a quantity.
    """
    if not isinstance(rule, Mapping) or set(rule) - _RULE_KEYS:
        raise ValueError(f"Header convention {source}: malformed rule {rule!r}.")

    if "keyword_prefix" in rule:
        if set(rule) != {"keyword_prefix"}:
            raise ValueError(
                f"Header convention {source}: a keyword_prefix rule takes "
                "no other field."
            )
        return MatchRule(keyword_prefix=str(rule["keyword_prefix"]))

    tests = set(rule) & {"present", "equals", "startswith", "contains"}
    subjects = set(rule) & {"keyword", "quantity"}

    if len(subjects) != 1 or len(tests) != 1:
        raise ValueError(
            f"Header convention {source}: a rule needs a keyword or a quantity "
            "and exactly one of present, equals, startswith or contains."
        )

    if "quantity" in rule:
        if before is None:
            raise ValueError(
                f"Header convention {source}: only a requirement may test a "
                "quantity."
            )

        if rule["quantity"] not in before:
            raise ValueError(
                f"Header convention {source}: a requirement may test only a "
                f"quantity read before, among {', '.join(before)}; got "
                f"{rule['quantity']!r}."
            )

    if "present" in rule and rule["present"] is not True:
        raise ValueError(
            f"Header convention {source}: present must be true when given."
        )

    return MatchRule(
        keyword=None if "keyword" not in rule else str(rule["keyword"]),
        quantity=None if "quantity" not in rule else str(rule["quantity"]),
        present="present" in rule,
        equals=_optional_text(rule.get("equals")),
        startswith=_optional_text(rule.get("startswith")),
        contains=_optional_text(rule.get("contains")),
    )


def _parse_spec(
    spec: Any, *, definition: QuantityDefinition, source: str
) -> QuantitySpec:
    """Validate how a convention reads one quantity."""
    if not isinstance(spec, Mapping) or set(spec) - _SPEC_KEYS:
        raise ValueError(
            f"Header convention {source}: unknown fields "
            f"{sorted(set(spec) - _SPEC_KEYS) if isinstance(spec, Mapping) else spec}."
        )

    keywords = spec.get("keywords", [])
    has_value = "value" in spec
    formula_text = spec.get("formula")
    has_table = "table" in spec or "by" in spec

    if formula_text is not None and (keywords or has_value):
        raise ValueError(
            f"Header convention {source}: a formula takes neither keywords nor "
            "a value."
        )

    if has_table and (keywords or has_value or formula_text is not None):
        raise ValueError(
            f"Header convention {source}: a table takes neither keywords, a "
            "value nor a formula."
        )

    if (
        not isinstance(keywords, list)
        or (not keywords and not has_value and formula_text is None and not has_table)
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

    if declares and (has_value or formula_text is not None or has_table):
        raise ValueError(
            f"Header convention {source}: a stated, derived or tabulated value "
            "takes its unit from the convention, not from the header."
        )

    unit = _optional_text(spec.get("unit"))
    value = None
    formula = None

    if has_value:
        value = _parse_value(spec["value"], definition, unit, source)

    if formula_text is not None:
        if definition.kind != "number":
            raise ValueError(
                f"Header convention {source}: only a number can be derived by "
                "a formula."
            )

        if unit is None:
            raise ValueError(
                f"Header convention {source}: a derived number needs its unit."
            )

        try:
            formula = Formula.parse(formula_text)
        except ValueError as error:
            raise ValueError(f"Header convention {source}: {error}.") from None

    by, table = (
        _parse_table(spec, definition, unit, source) if has_table else (None, None)
    )
    requires = spec.get("requires", [])

    if not isinstance(requires, list):
        raise ValueError(
            f"Header convention {source}: requires must be a list of rules."
        )

    overrides = spec.get("overrides", [])
    ignore_prefixes = spec.get("ignore_prefixes", [])

    for key, items in (("overrides", overrides), ("ignore_prefixes", ignore_prefixes)):
        if not isinstance(items, list) or not all(
            isinstance(item, str) and item.strip() for item in items
        ):
            raise ValueError(
                f"Header convention {source}: {key} must be a list of names."
            )

    if ignore_prefixes and not keywords:
        raise ValueError(
            f"Header convention {source}: ignore_prefixes applies only to "
            "values read from keywords."
        )

    return QuantitySpec(
        keywords=tuple(item.strip() for item in keywords),
        unit=unit,
        unit_keywords=tuple(str(item) for item in spec.get("unit_keywords", ())),
        unit_from_comment=bool(spec.get("unit_from_comment", False)),
        units=units,
        note=_optional_text(spec.get("note")),
        value=value,
        formula=formula,
        requires=tuple(
            _parse_rule(rule, source=source, before=_quantities_before(definition))
            for rule in requires
        ),
        overrides=tuple(item.strip() for item in overrides),
        ignore_prefixes=tuple(item.strip() for item in ignore_prefixes),
        by=by,
        table=table,
    )


def _parse_table(
    spec: Mapping[str, Any],
    definition: QuantityDefinition,
    unit: str | None,
    source: str,
) -> tuple[str, Mapping[str, Any]]:
    """Validate a table of values keyed by a text quantity read before."""
    if "by" not in spec or "table" not in spec:
        raise ValueError(
            f"Header convention {source}: a table needs both by and table."
        )

    by = str(spec["by"]).strip()
    allowed = tuple(
        name
        for name in _quantities_before(definition)
        if QUANTITIES[name].kind == "string"
    )

    if by not in allowed:
        raise ValueError(
            f"Header convention {source}: a table is keyed by a text quantity "
            f"read before, among {', '.join(allowed)}; got {by!r}."
        )

    table = spec["table"]

    if not isinstance(table, Mapping) or not table:
        raise ValueError(
            f"Header convention {source}: table must map values of {by} to " "values."
        )

    entries: dict[str, Any] = {}

    for key, value in table.items():
        if not isinstance(key, str):
            raise ValueError(
                f"Header convention {source}: the keys of the table are text; "
                f"quote the key {key!r}, which YAML did not read as text."
            )

        name = _table_key(key, by)

        if not name or name in entries:
            raise ValueError(
                f"Header convention {source}: the keys of the table must be "
                "distinct and non-empty."
            )

        entries[name] = _parse_value(value, definition, unit, source)

    return by, MappingProxyType(entries)


def _quantities_before(definition: QuantityDefinition) -> tuple[str, ...]:
    """Return the quantities read before one, in the order of reading."""
    names = tuple(QUANTITIES)

    return names[: names.index(definition.name)]


def _parse_value(
    value: Any, definition: QuantityDefinition, unit: str | None, source: str
) -> str | float | int:
    """Validate a value that a convention states."""
    if definition.kind == "string":
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Header convention {source}: the value must be a non-empty " "string."
            )

        if unit is not None:
            raise ValueError(f"Header convention {source}: a string takes no unit.")

        return value.strip()

    number = None if isinstance(value, str) else _finite_number(value, definition.kind)

    if number is None:
        raise ValueError(
            f"Header convention {source}: the value must be a finite "
            + ("integer." if definition.kind == "integer" else "number.")
        )

    if definition.kind == "number" and unit is None:
        raise ValueError(
            f"Header convention {source}: a number that the convention states "
            "needs its unit."
        )

    return number


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
    "DECLARABLE_QUANTITIES",
    "DERIVED_QUANTITIES",
    "IMAGE_UNIT",
    "LINEAR_ENCODING",
    "PHOTOMETRIC_SYSTEMS",
    "QUANTITIES",
    "STATUSES",
    "USABLE_STATUSES",
    "Formula",
    "HeaderConvention",
    "HeaderQuantity",
    "HeaderReading",
    "QuantityDefinition",
    "QuantitySpec",
    "canonical_unit",
    "ensure_linear_pixels",
    "load_header_conventions",
    "normalize_declared_unit",
    "read_celestial_wcs",
    "read_header",
    "read_pixel_scales",
    "validate_declarations",
]
