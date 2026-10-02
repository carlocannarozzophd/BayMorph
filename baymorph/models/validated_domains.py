"""Validated rendering domains of the morphology families of AD-048.

AD-048 registers a family only within the part of its parameter space where
its rendering met the criteria of AD-046 on the bench recorded in
``docs/research/MORPHOLOGY_FAMILIES_RENDERING_2026-10-02.md``. Configuration
validation rejects a prior whose support leaves that domain.

A parameter without an entry is not restricted: larger radii, broader cores
and lower Sérsic indices are smoother on the scale of a pixel than the
validated cases. The four original models have no entry; their domain is the
one AD-046 states.
"""

from __future__ import annotations

from collections.abc import Mapping

Bounds = tuple[float | None, float | None]

VALIDATED_DOMAINS: dict[str, dict[str, Bounds]] = {
    "boxy_sersic": {
        "re": (1.0, None),
        "n": (None, 6.0),
        "c0": (-1.0, 1.0),
    },
    "core_sersic": {
        "re": (1.0, None),
        "n": (None, 6.0),
        "rb": (0.5, None),
        "gamma": (0.0, 0.9),
        "alpha": (0.5, 2.0),
    },
    "bulge_broken_disk": {
        "wb_disk": (0.5, None),
    },
}


def validated_domain(model_name: str) -> Mapping[str, Bounds]:
    """Return the validated bounds of the parameters of a registered model.

    Each value is ``(lower, upper)``, with ``None`` for an open side.
    """
    return VALIDATED_DOMAINS.get(model_name, {})
