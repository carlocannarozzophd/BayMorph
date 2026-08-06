"""Tests for public morphological protocol exports."""

from __future__ import annotations

import baymorph.models as models


def test_morphological_component_is_exported() -> None:
    """The component protocol must be available from baymorph.models."""
    assert (
        models.MorphologicalComponent.__name__
        == "MorphologicalComponent"
    )


def test_decomposable_morphology_is_exported() -> None:
    """The decomposition protocol must be available from baymorph.models."""
    assert (
        models.DecomposableMorphology.__name__
        == "DecomposableMorphology"
    )


def test_protocols_are_declared_in_models_all() -> None:
    """Public protocol exports must remain explicit."""
    assert "MorphologicalComponent" in models.__all__
    assert "DecomposableMorphology" in models.__all__
