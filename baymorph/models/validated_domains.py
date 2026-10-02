"""Validated rendering domains of the morphology families of AD-048.

AD-048 registers a family only within the part of its parameter space where
its rendering met the criteria of AD-046 on the bench recorded in
``docs/research/MORPHOLOGY_FAMILIES_RENDERING_2026-10-02.md``. Configuration
validation rejects a prior whose support leaves that domain.

A parameter without an entry is not restricted: larger radii, broader cores
and lower Sérsic indices are smoother on the scale of a pixel than the
validated cases, and the radius of a ring does not set its sharpness, which
its width does. The four original models have no entry; their domain is the
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
    "bulge_disk_bar": {
        "a_bar": (5.0, None),
        "alpha_bar": (2.0, None),
        "c0_bar": (0.0, 3.5),
        "q_bar": (0.2, None),
    },
    "bulge_disk_ring": {
        "sigma_ring": (2.0, None),
        "q_ring": (0.5, None),
    },
    "bulge_disk_spiral": {
        "re_disk": (10.0, None),
        "q_disk": (0.4, None),
        "winding_spiral": (-6.0, 6.0),
        "r_in_spiral": (1.0, None),
    },
}


def validated_domain(model_name: str) -> Mapping[str, Bounds]:
    """Return the validated bounds of the parameters of a registered model.

    Each value is ``(lower, upper)``, with ``None`` for an open side.
    """
    return VALIDATED_DOMAINS.get(model_name, {})
