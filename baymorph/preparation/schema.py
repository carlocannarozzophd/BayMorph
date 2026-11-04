"""The configuration of the new preparation: one table of every key.

The sections ``object``, ``data``, ``outputs``, ``preparation`` and ``psf``
of a configuration are described once, below, as a table: each key with its
check, its default and its meaning. Reading a configuration checks every key
before any stage runs. An unknown key is refused with the valid names, a
value outside its domain is refused with its key, and each resolved value
records whether it came from the user or from a default.

The key names and the meaning of a valid configuration are those of
BayMorph 0.6 (decision D7 of the rewrite): what 0.6 accepted and used as
written stays valid; what 0.6 ignored, misread or could not run is refused
when the configuration is read. The sections ``fit``, ``diagnostics`` and
``simulation`` belong to other readers and are not checked here.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

from baymorph.core.correlated_noise import (
    DEFAULT_SKY_CORRELATION,
    FIELD_SEPARATED_GROWTH_PIXELS,
    SKY_CORRELATION_ESTIMATORS,
)
from baymorph.pipeline.opening import validate_region
from baymorph.pipeline.psf_candidate_screen import (
    DEFAULT_POINT_SOURCE_SCREEN,
    POINT_SOURCE_SCREEN_RULES,
)
from baymorph.preparation.header import validate_declarations

SCHEMA_VERSION = 1
"""Version of the table of keys."""

USER = "user"
"""Source of a value the configuration declares."""

DEFAULT = "default"
"""Source of a value the table supplies."""

Check = Callable[[Any, str], Any]
"""A check takes a value and its dotted name, and returns the value in
canonical form or raises ValueError naming the key."""


class _Required:
    """Marker of a key that has no default."""

    def __repr__(self) -> str:
        return "REQUIRED"


REQUIRED = _Required()


@dataclass(frozen=True)
class Derived:
    """A default that depends on other resolved values.

    Parameters
    ----------
    rule : callable
        Function of the resolved configuration (a nested mapping) that
        returns the default.
    description : str
        The rule in words, for the documentation of the key.
    """

    rule: Callable[[Mapping[str, Any]], Any]
    description: str


@dataclass(frozen=True)
class Key:
    """One key of a section.

    Parameters
    ----------
    name : str
        Name of the key.
    check : callable
        Check of a declared value.
    default : Any
        Default value, a :class:`Derived` default, or :data:`REQUIRED`.
    meaning : str
        One-line meaning, with its unit.
    """

    name: str
    check: Check
    default: Any
    meaning: str


@dataclass(frozen=True)
class Section:
    """One section of the configuration.

    Parameters
    ----------
    name : str
        Name of the section.
    entries : tuple of Key or Section
        Keys and sub-sections, in documentation order.
    meaning : str
        One-line meaning.
    refused : Mapping of str to str, optional
        Retired or misspelled keys, each with the message that refuses it.
    """

    name: str
    entries: tuple[Key | Section, ...]
    meaning: str
    refused: Mapping[str, str] = field(default_factory=dict)

    def names(self) -> list[str]:
        """Return the names of the entries, in order."""
        return [entry.name for entry in self.entries]


# ---------------------------------------------------------------------------
# Checks


def _finite_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{where} must be a number; got {value!r}.")

    number = float(value)

    if not math.isfinite(number):
        raise ValueError(f"{where} must be finite; got {value!r}.")

    return number


def number(
    *,
    above: float | None = None,
    at_least: float | None = None,
    at_most: float | None = None,
) -> Check:
    """Return a check of a finite number within bounds."""

    def check(value: Any, where: str) -> float:
        result = _finite_number(value, where)

        if above is not None and not result > above:
            raise ValueError(f"{where} must be greater than {above:g}; got {value!r}.")

        if at_least is not None and not result >= at_least:
            raise ValueError(f"{where} must be at least {at_least:g}; got {value!r}.")

        if at_most is not None and not result <= at_most:
            raise ValueError(f"{where} must be at most {at_most:g}; got {value!r}.")

        return result

    return check


def integer(*, at_least: int | None = None) -> Check:
    """Return a check of an integer of at least ``at_least``."""

    def check(value: Any, where: str) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{where} must be an integer; got {value!r}.")

        if at_least is not None and value < at_least:
            raise ValueError(f"{where} must be at least {at_least}; got {value!r}.")

        return int(value)

    return check


def boolean(value: Any, where: str) -> bool:
    """Check a Boolean."""
    if not isinstance(value, bool):
        raise ValueError(f"{where} must be true or false; got {value!r}.")

    return value


def text(value: Any, where: str) -> str:
    """Check a non-empty text."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a non-empty text; got {value!r}.")

    return value


def path_text(value: Any, where: str) -> str:
    """Check a path, written as text, and return it in canonical form."""
    return str(Path(text(value, where)))


def choice(*options: str) -> Check:
    """Return a check of one of ``options``, ignoring case and spaces."""

    def check(value: Any, where: str) -> str:
        if isinstance(value, str) and value.strip().lower() in options:
            return value.strip().lower()

        raise ValueError(f"{where} must be one of {list(options)}; got {value!r}.")

    return check


def optional(inner: Check) -> Check:
    """Return a check that also accepts null."""

    def check(value: Any, where: str) -> Any:
        return None if value is None else inner(value, where)

    return check


def or_auto(inner: Check) -> Check:
    """Return a check that also accepts the text ``auto``."""

    def check(value: Any, where: str) -> Any:
        if isinstance(value, str) and value.strip().lower() == "auto":
            return "auto"

        return inner(value, where)

    return check


def object_name(value: Any, where: str) -> str:
    """Check the name of the object, a text or an integer."""
    if isinstance(value, Integral) and not isinstance(value, bool):
        return str(value)

    return text(value, where)


def hdu(value: Any, where: str) -> int | str | None:
    """Check an HDU: null, a non-negative index or a non-empty name."""
    if value is None:
        return None

    if isinstance(value, Integral) and not isinstance(value, bool) and value >= 0:
        return int(value)

    if isinstance(value, str) and value.strip():
        return value

    raise ValueError(
        f"{where} must be a non-negative HDU index or a non-empty extension name; "
        f"got {value!r}."
    )


def pixel_scale(value: Any, where: str) -> float | str | None:
    """Check a pixel scale: null or ``auto`` (from the header), or arcsec."""
    if value is None:
        return None

    return or_auto(number(above=0.0))(value, where)


def target_position(value: Any, where: str) -> dict[str, float] | str | None:
    """Check a target position: null, ``center``, ``{x, y}`` or ``[x, y]``.

    The position is in pixels of the image, as AD-055 defines it.
    """
    if value is None:
        return None

    if isinstance(value, str):
        if value.strip().lower() == "center":
            return "center"

        raise ValueError(f"{where} must be 'center' or two coordinates; got {value!r}.")

    if isinstance(value, Mapping):
        if "ra" in value or "dec" in value:
            raise ValueError(
                f"{where} gives the target in pixels, {{x: ..., y: ...}}; baymorph "
                "config --target-ra and --target-dec, or the observation advisor, "
                "convert a position on the sky."
            )

        if set(value) != {"x", "y"}:
            raise ValueError(f"{where} must have exactly the entries x and y.")

        coordinates = (value["x"], value["y"])
    elif isinstance(value, Sequence) and not isinstance(value, str) and len(value) == 2:
        coordinates = (value[0], value[1])
    else:
        raise ValueError(f"{where} must be 'center' or two coordinates; got {value!r}.")

    x = _finite_number(coordinates[0], f"{where}.x")
    y = _finite_number(coordinates[1], f"{where}.y")

    return {"x": x, "y": y}


def region(value: Any, where: str) -> Any:
    """Check a working region (AD-053) with the rule of the opening."""
    return validate_region(value)


def calibration(value: Any, where: str) -> dict[str, Any] | None:
    """Check the declared calibration (AD-053) with the rule of the header engine."""
    if value is None:
        return None

    return validate_declarations(value)


def exclusion_box(value: Any, where: str) -> dict[str, float] | None:
    """Check a box ``{xmin, xmax, ymin, ymax}`` in pixels, or null."""
    if value is None:
        return None

    names = ("xmin", "xmax", "ymin", "ymax")

    if not isinstance(value, Mapping) or set(value) != set(names):
        raise ValueError(f"{where} must be null or a mapping with entries {names}.")

    box = {name: _finite_number(value[name], f"{where}.{name}") for name in names}

    if not (box["xmin"] < box["xmax"] and box["ymin"] < box["ymax"]):
        raise ValueError(f"{where} needs xmin < xmax and ymin < ymax.")

    return box


# ---------------------------------------------------------------------------
# The table


def _cutout_mode_default(root: Mapping[str, Any]) -> str:
    return "fixed" if root["data"]["cutout"]["size"] is not None else "auto"


def _growth_default(root: Mapping[str, Any]) -> int:
    if root["data"]["mask_mode"] == "union":
        return FIELD_SEPARATED_GROWTH_PIXELS

    return 0


_POSITIVE = number(above=0.0)

OBJECT = Section(
    "object",
    (Key("name", object_name, REQUIRED, "Name of the object, used in outputs."),),
    "The object observed.",
)

CUTOUT = Section(
    "cutout",
    (
        Key(
            "mode",
            choice("fixed", "auto"),
            Derived(_cutout_mode_default, "fixed when a size is declared, else auto"),
            "Fixed side, or automatic side from the extent of the galaxy (AD-055).",
        ),
        Key(
            "size",
            optional(integer(at_least=1)),
            None,
            "Side of a fixed cutout, in pixels.",
        ),
        Key(
            "min_size",
            optional(integer(at_least=1)),
            None,
            "Smallest side of an automatic cutout, in pixels.",
        ),
        Key(
            "max_size",
            optional(integer(at_least=1)),
            None,
            "Largest side of an automatic cutout, in pixels.",
        ),
        Key(
            "initial_factor",
            number(above=1.0),
            4.0,
            "Side of the first automatic cutout over the side of the target segment.",
        ),
        Key(
            "sky_margin",
            number(at_least=0.0),
            20.0,
            "Sky beyond the extent of the galaxy kept in the cutout, in pixels.",
        ),
    ),
    "The cutout about the target.",
)

DATA = Section(
    "data",
    (
        Key("image_path", path_text, REQUIRED, "FITS file of the observation."),
        Key("hdu", hdu, None, "HDU holding the image; null for the first image."),
        Key(
            "calibration",
            calibration,
            None,
            "Declared calibration quantities (AD-053).",
        ),
        Key("region", region, None, "Working region read from the file (AD-053)."),
        Key(
            "pixel_scale",
            pixel_scale,
            None,
            "Pixel scale in arcsec, or null/auto from the header.",
        ),
        Key(
            "target_position",
            target_position,
            None,
            "Centre of the galaxy in pixels, or 'center' (AD-055).",
        ),
        CUTOUT,
        Key(
            "mask_path",
            optional(path_text),
            None,
            "Curated mask, True = excluded.",
        ),
        Key(
            "mask_mode",
            choice("union", "exact", "replace"),
            "union",
            "How the curated mask combines with the automatic one.",
        ),
    ),
    "The observation and the target.",
    refused={
        "noise_map_path": (
            "data.noise_map_path is not supported yet: the external weight map of "
            "AD-047 is not implemented."
        ),
        "psf_path": (
            "data.psf_path is not supported: declare the PSF under psf (a PSF from "
            "file is planned, decision D3)."
        ),
        "external_mask_path": (
            "data.external_mask_path is no longer supported. Declare the curated "
            "mask under data.mask_path."
        ),
    },
)

OUTPUTS = Section(
    "outputs",
    (
        Key("root", path_text, REQUIRED, "Folder of the output namespaces."),
        Key(
            "overwrite",
            boolean,
            False,
            "Rebuild existing products instead of reusing them (AD-042).",
        ),
    ),
    "Where the products are written.",
)

PREPARATION = Section(
    "preparation",
    (
        Section(
            "background",
            (
                Key(
                    "box_size",
                    integer(at_least=1),
                    30,
                    "Box of the mesh background of the detection, in pixels.",
                ),
                Key(
                    "filter_size",
                    integer(at_least=1),
                    3,
                    "Median filter of that mesh, in boxes.",
                ),
            ),
            "The background used by the detection.",
            refused={
                "method": (
                    "preparation.background.method is not supported: the background "
                    "is estimated empirically from the sky region."
                )
            },
        ),
        Section(
            "noise",
            (
                Key(
                    "sky_correlation",
                    choice(*SKY_CORRELATION_ESTIMATORS),
                    DEFAULT_SKY_CORRELATION,
                    "Estimator of the sky correlation (AD-047, AD-051).",
                ),
            ),
            "The noise model inputs.",
            refused={
                "method": (
                    "preparation.noise.method is not supported: the noise scale is "
                    "measured about a plane fitted to the sky region (AD-041)."
                )
            },
        ),
        Section(
            "masking",
            (
                Key(
                    "negative_threshold_sigma",
                    optional(number(above=0.0)),
                    5.0,
                    "Impossibly negative pixels, in noise scales; null disables "
                    "(AD-043).",
                ),
                Key(
                    "contaminant_rule",
                    choice("local_excess", "all_segments"),
                    "local_excess",
                    "Which detected segments are masked (AD-050).",
                ),
                Key(
                    "contaminant_growth_pixels",
                    integer(at_least=0),
                    Derived(_growth_default, "20 with mask_mode union, else 0"),
                    "Growth of the contaminant masks, in pixels (AD-051).",
                ),
            ),
            "The masks of the contaminants.",
        ),
        Section(
            "detection",
            (
                Key(
                    "n_sigma",
                    number(above=0.0),
                    3.0,
                    "Detection threshold, in noise scales.",
                ),
                Key(
                    "n_levels",
                    integer(at_least=1),
                    32,
                    "Levels of the deblending.",
                ),
                Key(
                    "n_pixels",
                    integer(at_least=1),
                    20,
                    "Smallest segment, in connected pixels.",
                ),
            ),
            "The detection of sources.",
            refused={
                "npixels": (
                    "Unsupported configuration key 'preparation.detection.npixels'. "
                    "Use 'preparation.detection.n_pixels' instead."
                )
            },
        ),
    ),
    "The preparation of the image.",
)

CANDIDATE_SELECTION = Section(
    "candidate_selection",
    (
        Key("auto", boolean, True, "Select the PSF stars automatically."),
        Key(
            "target_n_candidates",
            integer(at_least=1),
            30,
            "Number of stars kept.",
        ),
        Key("min_area", integer(at_least=1), 10, "Smallest star segment, in pixels."),
        Key("max_area", integer(at_least=1), 800, "Largest star segment, in pixels."),
        Key(
            "min_peak",
            number(at_least=0.0),
            20.0,
            "Lowest peak, in image units.",
        ),
        Key(
            "max_peak",
            number(above=0.0),
            80000.0,
            "Highest peak, in image units, below saturation.",
        ),
        Key(
            "min_peak_snr",
            number(above=0.0),
            30.0,
            "Lowest peak signal-to-noise.",
        ),
        Key(
            "max_eccentricity",
            number(at_least=0.0, at_most=1.0),
            0.8,
            "Highest eccentricity of a star.",
        ),
        Key(
            "max_elongation",
            number(at_least=1.0),
            2.5,
            "Highest elongation of a star.",
        ),
        Key(
            "edge_margin",
            integer(at_least=0),
            30,
            "Distance from the image edge, in pixels.",
        ),
        Key(
            "galaxy_exclusion_box",
            exclusion_box,
            None,
            "Box about the galaxy where no star is taken, in pixels.",
        ),
        Key(
            "sigma_factor",
            number(above=0.0),
            4.0,
            "Half-size of a star stamp, in Gaussian widths.",
        ),
        Key(
            "size_margin",
            integer(at_least=0),
            4,
            "Pixels added to that half-size.",
        ),
        Key(
            "min_half_size",
            or_auto(integer(at_least=1)),
            12,
            "Smallest half-size of a star stamp, in pixels, or auto.",
        ),
        Key(
            "max_half_size",
            or_auto(integer(at_least=1)),
            30,
            "Largest half-size of a star stamp, in pixels, or auto.",
        ),
        Key(
            "isolation_margin",
            number(at_least=0.0),
            2.0,
            "Clearance between a star and its neighbours, in pixels.",
        ),
        Key(
            "point_source_screen",
            choice(*POINT_SOURCE_SCREEN_RULES),
            DEFAULT_POINT_SOURCE_SCREEN,
            "Screen of the stars by their sizes (AD-056).",
        ),
    ),
    "The selection of the PSF stars.",
)

CONSTRUCTION = Section(
    "construction",
    (
        Key(
            "centroid_tolerance_pixels",
            number(above=0.0),
            0.05,
            "Largest centroid offset of the kernel, in pixels.",
        ),
        Key(
            "sky_inner_radius",
            number(above=0.0),
            80.0,
            "Inner radius of the sky ring of a star, in pixels.",
        ),
        Key(
            "sky_outer_radius",
            number(above=0.0),
            160.0,
            "Outer radius of the sky ring of a star, in pixels.",
        ),
        Key(
            "mask_growth_pixels",
            integer(at_least=0),
            5,
            "Growth of the masks of a star's neighbours, in pixels.",
        ),
        Key(
            "support_half_size",
            integer(at_least=1),
            64,
            "Half-size of the extended kernel, in pixels (AD-046).",
        ),
    ),
    "The construction of the empirical PSF.",
    refused={
        "auto_size": (
            "Unsupported configuration key 'psf.empirical.construction.auto_size'. "
            "Empirical PSF sizing is controlled by candidate-selection cutout limits."
        ),
        "normalize": (
            "Unsupported configuration key 'psf.empirical.construction.normalize'. "
            "BayMorph always normalizes PSF kernels to unit sum."
        ),
    },
)

PSF = Section(
    "psf",
    (
        Key(
            "type",
            choice("empirical", "gaussian"),
            "empirical",
            "Empirical PSF from the stars, or a Gaussian PSF.",
        ),
        Section(
            "gaussian",
            (
                Key(
                    "fwhm",
                    optional(_POSITIVE),
                    None,
                    "FWHM of a Gaussian PSF, in pixels.",
                ),
            ),
            "The Gaussian PSF.",
        ),
        Section("empirical", (CANDIDATE_SELECTION, CONSTRUCTION), "The empirical PSF."),
    ),
    "The point-spread function.",
)

SECTIONS: tuple[Section, ...] = (OBJECT, DATA, OUTPUTS, PREPARATION, PSF)
"""The sections this reader owns, in order."""

REQUIRED_SECTIONS = ("object", "data", "outputs")
"""Sections every configuration declares."""

OTHER_SECTIONS = ("fit", "diagnostics", "simulation")
"""Sections that other readers check."""

REFUSED_SECTIONS: Mapping[str, str] = MappingProxyType(
    {
        "masking": (
            "Unsupported top-level configuration section 'masking'. The only "
            "masking control is 'preparation.masking'."
        ),
        "observations": (
            "A joint configuration with 'observations' is read by the joint path, "
            "which the new preparation does not cover yet (decision D10)."
        ),
    }
)


# ---------------------------------------------------------------------------
# Reading


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})

    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)

    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}

    if isinstance(value, tuple):
        return [_thaw(item) for item in value]

    return value


@dataclass(frozen=True, eq=False)
class PreparationSettings:
    """The checked and resolved preparation sections of one configuration.

    Parameters
    ----------
    values : Mapping
        Resolved sections, read-only, by section name.
    sources : Mapping of str to str
        Source of each resolved key, by dotted name: ``user`` or ``default``.
    """

    values: Mapping[str, Any]
    sources: Mapping[str, str]

    def get(self, dotted_name: str) -> Any:
        """Return one resolved value by its dotted name."""
        value: Any = self.values

        for part in dotted_name.split("."):
            if not isinstance(value, Mapping) or part not in value:
                raise KeyError(dotted_name)

            value = value[part]

        return _thaw(value)

    def as_dict(self) -> dict[str, Any]:
        """Return a mutable deep copy of the resolved sections."""
        return copy.deepcopy(_thaw(self.values))


def _resolve_section(
    section: Section,
    given: Any,
    where: str,
    sources: dict[str, str],
    derived: list[tuple[str, Key]],
) -> dict[str, Any]:
    """Check the declared entries of one section and fill its defaults."""
    if given is None:
        given = {}

    if not isinstance(given, Mapping):
        raise ValueError(f"Config section '{where}' must be a mapping; got {given!r}.")

    for key in given:
        if key in section.refused:
            raise ValueError(section.refused[key])

    unknown = sorted(str(key) for key in set(given) - set(section.names()))

    if unknown:
        raise ValueError(
            f"Unsupported configuration entries in '{where}': {unknown}. "
            f"Available entries: {sorted(section.names())}."
        )

    resolved: dict[str, Any] = {}

    for entry in section.entries:
        name = f"{where}.{entry.name}"

        if isinstance(entry, Section):
            resolved[entry.name] = _resolve_section(
                entry, given.get(entry.name), name, sources, derived
            )
            continue

        if entry.name in given:
            resolved[entry.name] = entry.check(given[entry.name], name)
            sources[name] = USER
        elif entry.default is REQUIRED:
            raise ValueError(f"Config section '{where}' must contain '{entry.name}'.")
        elif isinstance(entry.default, Derived):
            resolved[entry.name] = None
            derived.append((name, entry))
        else:
            resolved[entry.name] = copy.deepcopy(entry.default)
            sources[name] = DEFAULT

    return resolved


def _set(root: dict[str, Any], dotted_name: str, value: Any) -> None:
    parts = dotted_name.split(".")
    target = root

    for part in parts[:-1]:
        target = target[part]

    target[parts[-1]] = value


def _cross_checks(root: Mapping[str, Any]) -> None:
    """Check the rules that tie keys together."""
    data = root["data"]
    cutout = data["cutout"]

    if data["mask_mode"] in {"exact", "replace"} and data["mask_path"] is None:
        raise ValueError(
            "Config data.mask_path is required when data.mask_mode is "
            f"{data['mask_mode']!r}."
        )

    if cutout["mode"] == "fixed" and cutout["size"] is None:
        raise ValueError(
            "A fixed cutout needs data.cutout.size in pixels; or set "
            "data.cutout.mode: auto."
        )

    if (
        cutout["min_size"] is not None
        and cutout["max_size"] is not None
        and cutout["min_size"] > cutout["max_size"]
    ):
        raise ValueError("data.cutout.min_size must not exceed data.cutout.max_size.")

    masking = root["preparation"]["masking"]
    growth = masking["contaminant_growth_pixels"]

    if growth > 0 and data["mask_mode"] in {"exact", "replace"}:
        raise ValueError(
            "Config preparation.masking.contaminant_growth_pixels grows the detected "
            f"contaminant segments, which data.mask_mode {data['mask_mode']!r} does "
            "not use: set it to 0, or use mask_mode 'union'."
        )

    sky_correlation = root["preparation"]["noise"]["sky_correlation"]

    if sky_correlation == "field_separated" and growth != FIELD_SEPARATED_GROWTH_PIXELS:
        raise ValueError(
            "Config preparation.noise.sky_correlation 'field_separated' was validated "
            "on sky regions whose contaminant masks are grown by "
            f"{FIELD_SEPARATED_GROWTH_PIXELS} px "
            "(preparation.masking.contaminant_growth_pixels)."
        )

    psf = root["psf"]

    if psf["type"] == "gaussian" and psf["gaussian"]["fwhm"] is None:
        raise ValueError("psf.type 'gaussian' needs psf.gaussian.fwhm in pixels.")

    selection = psf["empirical"]["candidate_selection"]

    if not selection["min_peak"] < selection["max_peak"]:
        raise ValueError(
            f"psf.empirical.candidate_selection.min_peak {selection['min_peak']:.6g} "
            f"is not below max_peak {selection['max_peak']:.6g}, so no star can be "
            "selected for the PSF."
        )

    if selection["min_area"] > selection["max_area"]:
        raise ValueError(
            "psf.empirical.candidate_selection.min_area must not exceed max_area."
        )

    half_sizes = (selection["min_half_size"], selection["max_half_size"])

    if "auto" not in half_sizes and half_sizes[0] > half_sizes[1]:
        raise ValueError(
            "psf.empirical.candidate_selection.min_half_size must not exceed "
            "max_half_size."
        )

    construction = psf["empirical"]["construction"]

    if not (
        construction["support_half_size"]
        < construction["sky_inner_radius"]
        < construction["sky_outer_radius"]
    ):
        raise ValueError(
            "psf.empirical.construction needs support_half_size < sky_inner_radius "
            "< sky_outer_radius, so that the sky of each star is measured beyond "
            "its light."
        )


def read_preparation_settings(config: Mapping[str, Any]) -> PreparationSettings:
    """Check and resolve the preparation sections of one configuration.

    Parameters
    ----------
    config : Mapping
        A configuration as loaded from YAML.

    Returns
    -------
    PreparationSettings
        The resolved sections and the source of every value.

    Raises
    ------
    TypeError
        If ``config`` is not a mapping.
    ValueError
        If a section or key is unknown, retired, missing, out of its domain,
        or inconsistent with another key.
    """
    if not isinstance(config, Mapping):
        raise TypeError("A configuration is a mapping of sections.")

    for name in config:
        if name in REFUSED_SECTIONS:
            raise ValueError(REFUSED_SECTIONS[name])

    known = [section.name for section in SECTIONS] + list(OTHER_SECTIONS)
    unknown = sorted(str(name) for name in set(config) - set(known))

    if unknown:
        raise ValueError(
            f"Unsupported configuration sections: {unknown}. Available sections: "
            f"{sorted(known)}."
        )

    missing = [name for name in REQUIRED_SECTIONS if name not in config]

    if missing:
        raise ValueError(f"Missing required config sections: {missing}")

    sources: dict[str, str] = {}
    derived: list[tuple[str, Key]] = []
    root = {
        section.name: _resolve_section(
            section, config.get(section.name), section.name, sources, derived
        )
        for section in SECTIONS
    }

    for name, key in derived:
        _set(root, name, key.default.rule(root))
        sources[name] = DEFAULT

    _cross_checks(root)

    return PreparationSettings(
        values=_freeze(root), sources=MappingProxyType(dict(sorted(sources.items())))
    )


def describe_keys() -> list[dict[str, Any]]:
    """Return the table of keys, one row per key, for documentation.

    Returns
    -------
    list of dict
        Rows with ``key`` (dotted name), ``default`` and ``meaning``.
    """
    rows: list[dict[str, Any]] = []

    def visit(section: Section, prefix: str) -> None:
        for entry in section.entries:
            name = f"{prefix}.{entry.name}"

            if isinstance(entry, Section):
                visit(entry, name)
                continue

            if entry.default is REQUIRED:
                default: Any = "required"
            elif isinstance(entry.default, Derived):
                default = entry.default.description
            else:
                default = entry.default

            rows.append({"key": name, "default": default, "meaning": entry.meaning})

    for section in SECTIONS:
        visit(section, section.name)

    return rows
