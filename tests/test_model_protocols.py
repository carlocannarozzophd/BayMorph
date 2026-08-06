"""Tests for structural morphological-model protocols."""

from __future__ import annotations

from baymorph.models.composite import CompositeModel
from baymorph.models.protocols import MorphologicalComponent
from baymorph.models.sersic import SersicModel


def make_sersic() -> SersicModel:
    """Return a valid Sérsic component for protocol tests."""
    return SersicModel(
        flux=100.0,
        re=4.0,
        n=2.0,
        q=0.8,
        theta=0.1,
        x0=10.0,
        y0=10.0,
    )


def test_sersic_model_satisfies_morphological_component_protocol() -> None:
    """The existing Sérsic implementation must satisfy the new contract."""
    model = make_sersic()

    assert isinstance(model, MorphologicalComponent)


def test_composite_model_satisfies_morphological_component_protocol() -> None:
    """Additive composite morphologies must satisfy the same contract."""
    model = CompositeModel(
        {
            "component_1": make_sersic(),
            "component_2": make_sersic(),
        }
    )

    assert isinstance(model, MorphologicalComponent)


def test_protocol_does_not_require_nominal_inheritance() -> None:
    """A structurally compatible custom object must satisfy the protocol."""

    class CompatibleComponent:
        def render(
            self,
            shape,
            *,
            origin=(0.0, 0.0),
        ):
            return make_sersic().render(
                shape=shape,
                origin=origin,
            )

        def render_components(
            self,
            shape,
            *,
            origin=(0.0, 0.0),
        ):
            return {
                "compatible": self.render(
                    shape=shape,
                    origin=origin,
                )
            }

    assert isinstance(
        CompatibleComponent(),
        MorphologicalComponent,
    )


def test_incomplete_object_does_not_satisfy_protocol() -> None:
    """Objects missing component rendering must fail the contract."""

    class IncompleteComponent:
        def render(
            self,
            shape,
            *,
            origin=(0.0, 0.0),
        ):
            return make_sersic().render(
                shape=shape,
                origin=origin,
            )

    assert not isinstance(
        IncompleteComponent(),
        MorphologicalComponent,
    )
