"""Effective gain of the source Poisson variance (AD-047).

AD-047 adds to the sky variance ``s**2`` of each pixel the Poisson variance of
the source, ``max(m, 0) / g``, where ``m`` is the source part of the model,
without the residual background, and ``g`` the effective gain: the inverse of
the variance per unit signal, in the units of the image. The gain is never a
parameter of the fit. Each observation declares where it comes from:

- ``declared``: the gain itself;
- ``sky_level``: the level ``B`` of a sky subtracted before the image reached
  BayMorph, which gives ``g = B / s**2``;
- ``measured_sky``: the background level ``B`` measured by the preparation,
  when the sky is still in the image, which gives the same ratio;
- ``absent``: the source variance is not modelled, which the products record.

When the noise of the sky is the Poisson noise of its level, ``B / s`` is the
square root of the photo-electrons per pixel, up to the variance factor of
any resampling, so both sky sources require ``B`` to be at least
``SKY_LEVEL_GUARD`` times ``s``: a smaller ratio means that the level is not a
sky or was already subtracted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

EFFECTIVE_GAIN_SOURCES = ("declared", "sky_level", "measured_sky", "absent")
"""Sources of the effective gain that a configuration may declare."""

SKY_LEVEL_GUARD = 10.0
"""Smallest ratio of a sky level to the sky scale that may give the gain."""


@dataclass(frozen=True)
class EffectiveGain:
    """The resolved effective gain of one observation.

    Parameters
    ----------
    source : str
        Declared source, one of ``EFFECTIVE_GAIN_SOURCES``.
    value : float or None
        Effective gain in the units of the image, or None when the source
        variance is absent.
    sky_level : float or None
        Sky level that gave the gain, for the two sky sources.
    sky_scale : float or None
        Prepared sky scale ``s``, when the observation has one.
    """

    source: str
    value: float | None
    sky_level: float | None
    sky_scale: float | None

    def metadata(self) -> dict[str, Any]:
        """Return the provenance of the gain for products and signatures."""
        return {
            "source": self.source,
            "value": self.value,
            "sky_level": self.sky_level,
            "sky_scale": self.sky_scale,
            "sky_level_over_scale": (
                None
                if self.sky_level is None or self.sky_scale is None
                else self.sky_level / self.sky_scale
            ),
            "sky_level_guard": SKY_LEVEL_GUARD,
        }


def resolve_effective_gain(
    declaration: Mapping[str, Any],
    *,
    sky_scale: float | None,
    measured_sky_level: float | None = None,
) -> EffectiveGain:
    """Resolve the effective gain that one observation declares (AD-047).

    Parameters
    ----------
    declaration : mapping
        The ``fit.likelihood.effective_gain`` block: ``source`` and, for
        ``declared``, ``gain``, or, for ``sky_level``, ``sky_level``, both in
        the units of the image.
    sky_scale : float or None
        Prepared sky scale ``s``; the two sky sources require it.
    measured_sky_level : float, optional
        Background level measured by the preparation, for ``measured_sky``.

    Returns
    -------
    EffectiveGain
        The gain and its provenance.

    Raises
    ------
    ValueError
        If the source is missing or unknown, a required value is missing or
        not a positive finite number, an option does not belong to the
        source, or a sky level fails the guard.
    """
    if not isinstance(declaration, Mapping):
        raise TypeError("fit.likelihood.effective_gain must be a mapping.")

    options = dict(declaration)
    source = options.pop("source", None)

    if source not in EFFECTIVE_GAIN_SOURCES:
        raise ValueError(
            "fit.likelihood.effective_gain.source must be one of "
            f"{', '.join(EFFECTIVE_GAIN_SOURCES)}; AD-047 gives the gain no "
            f"default, got {source!r}."
        )

    scale = (
        None
        if sky_scale is None
        else _positive_finite(sky_scale, "the prepared sky scale")
    )
    sky_level = None
    value = None

    if source == "declared":
        value = _positive_finite(options.pop("gain", None), "effective_gain.gain")
    elif source == "sky_level":
        sky_level = _positive_finite(
            options.pop("sky_level", None),
            "effective_gain.sky_level",
        )
    elif source == "measured_sky":
        if measured_sky_level is None:
            raise ValueError(
                "effective_gain.source 'measured_sky' requires the background "
                "level measured by the preparation."
            )
        sky_level = float(measured_sky_level)

    if options:
        raise ValueError(
            f"Unsupported options of fit.likelihood.effective_gain for source "
            f"{source!r}: {', '.join(sorted(options))}."
        )

    if sky_level is not None:
        if scale is None:
            raise ValueError(
                f"effective_gain.source {source!r} requires one prepared sky "
                "scale, which this observation does not have."
            )

        if not np.isfinite(sky_level) or sky_level < SKY_LEVEL_GUARD * scale:
            raise ValueError(
                f"The sky level {sky_level:.6g} is {sky_level / scale:.3g} times "
                f"the sky scale {scale:.6g}, below the {SKY_LEVEL_GUARD:g} that "
                "AD-047 requires of a sky that gives the gain: the level is not "
                "a sky or was already subtracted. Declare the gain instead."
            )
        value = sky_level / scale**2

    return EffectiveGain(
        source=str(source),
        value=value,
        sky_level=sky_level,
        sky_scale=scale,
    )


def _positive_finite(value: Any, name: str) -> float:
    """Return ``value`` as a float, or raise unless positive and finite."""
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number.")

    number = float(value)

    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number.")

    return number
