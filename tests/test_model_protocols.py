"""Tests for structural morphological-model protocols."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.models.composite import CompositeModel
from baymorph.models.protocols import (
    DecomposableMorphology,
    MorphologicalComponent,
)
from baymorph.models.sersic import SersicModel


def make_sersic() -> SersicModel:
    """Return a valid Sérsic component."""
    return SersicModel(
        flux=100.0,
        re=4.0,
        n=2.0,
        q=0.8,
        theta=0.1,
        x0=10.0,
        y0=10.0,
    )


class RenderOnlyComponent:
    """Minimal structurally compatible component."""

    def render(
        self,
        shape: tuple[int, int],
        *,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        del origin
        return np.ones(shape, dtype=float)


def test_sersic_satisfies_component_protocol() -> None:
    """Sérsic models must remain valid morphological components."""
    model = make_sersic()

    assert isinstance(model, MorphologicalComponent)


def test_sersic_exposes_named_decomposition() -> None:
    """The existing Sérsic model also exposes its named contribution."""
    model = make_sersic()

    assert isinstance(model, DecomposableMorphology)


def test_render_only_object_is_valid_component() -> None:
    """Leaf components must not be required to decompose themselves."""
    component = RenderOnlyComponent()

    assert isinstance(component, MorphologicalComponent)
    assert not isinstance(component, DecomposableMorphology)


def test_composite_satisfies_both_protocols() -> None:
    """Composite models must render totals and named contributions."""
    model = CompositeModel(
        {
            "component_1": make_sersic(),
            "component_2": RenderOnlyComponent(),
        }
    )

    assert isinstance(model, MorphologicalComponent)
    assert isinstance(model, DecomposableMorphology)


def test_composite_accepts_render_only_components() -> None:
    """Additive composition must require only component rendering."""
    model = CompositeModel(
        {
            "first": RenderOnlyComponent(),
            "second": RenderOnlyComponent(),
        },
        sky=2.0,
    )

    components = model.render_components((3, 4))
    total = model.render((3, 4))

    assert set(components) == {
        "first",
        "second",
    }
    assert np.allclose(total, 4.0)


def test_composite_rejects_incomplete_components() -> None:
    """Objects without render() must fail explicitly."""
    with pytest.raises(
        TypeError,
        match=r"does not implement render",
    ):
        CompositeModel(
            {
                "invalid": object(),
            }
        )


def test_composite_rejects_empty_component_name() -> None:
    """Component names must be stable non-empty identifiers."""
    with pytest.raises(
        ValueError,
        match="non-empty strings",
    ):
        CompositeModel(
            {
                "": RenderOnlyComponent(),
            }
        )
