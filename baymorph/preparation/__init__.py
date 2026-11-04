"""Preparation of observations for inference, rebuilt stage by stage.

This sub-package is under construction beside :mod:`baymorph.pipeline`. Each
stage of the preparation is a function that takes declared inputs and returns
one immutable product; :mod:`baymorph.preparation.store` saves, checks and
reuses every product in the same way, and :mod:`baymorph.preparation.identity`
gives each product an identity computed from the content of its inputs, never
from their paths.

Nothing here is used by :class:`baymorph.runner.BayMorphRunner` or by the
command-line interface yet. The public behaviour of the package is unchanged
until the switch to the new preparation.
"""

from __future__ import annotations

__all__: list[str] = []
