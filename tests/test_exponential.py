"""Tests for exponential disc morphology."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.models import (
    DecomposableMorphology,
    ExponentialComponent,
    MorphologicalComponent,
    SersicModel,
)


def make_exponential() -> ExponentialComponent:
    """Return a valid exponential component."""
    return ExponentialComponent(
        flux=100.0,
        re=4.0,
        q=0.7,
        theta=0.2,
        x0=12.3,
        y0=13.7,
    )


def make_equivalent_sersic() -> SersicModel:
    """Return the mathematically equivalent Sérsic model."""
    return SersicModel(
        flux=100.0,
        re=4.0,
        n=1.0,
        q=0.7,
        theta=0.2,
        x0=12.3,
        y0=13.7,
    )


def test_exponential_has_fixed_sersic_index() -> None:
    """The exponential profile must fix the Sérsic index to unity."""
    component = make_exponential()

    assert component.n == pytest.approx(1.0)


def test_exponential_matches_sersic_n_one() -> None:
    """The component must be numerically identical to Sérsic n=1."""
    exponential = make_exponential()
    sersic = make_equivalent_sersic()

    exponential_image = exponential.render(
        shape=(29, 31),
        integration_order=7,
    )
    sersic_image = sersic.render(
        shape=(29, 31),
        integration_order=7,
    )

    np.testing.assert_allclose(
        exponential_image,
        sersic_image,
        rtol=0.0,
        atol=0.0,
    )


def test_exponential_surface_brightness_matches_sersic() -> None:
    """Continuous profile evaluation must retain Sérsic normalization."""
    radii = np.linspace(0.0, 15.0, 100)

    np.testing.assert_allclose(
        make_exponential().surface_brightness(radii),
        make_equivalent_sersic().surface_brightness(radii),
        rtol=0.0,
        atol=0.0,
    )


def test_exponential_uses_effective_radius() -> None:
    """The declared radius must enclose half the continuous total flux."""
    component = make_exponential()

    enclosed_fraction = 1.0 - np.exp(
        -component.bn
    ) * (
        1.0 + component.bn
    )

    assert enclosed_fraction == pytest.approx(
        0.5,
        rel=1.0e-12,
        abs=1.0e-12,
    )


def test_exponential_exposes_named_component() -> None:
    """Component rendering must use an explicit exponential name."""
    component = make_exponential()

    rendered = component.render(
        shape=(25, 27),
    )
    components = component.render_components(
        shape=(25, 27),
    )

    assert set(components) == {
        "exponential",
    }

    np.testing.assert_allclose(
        components["exponential"],
        rendered,
        rtol=0.0,
        atol=0.0,
    )


def test_exponential_satisfies_model_protocols() -> None:
    """The component must satisfy regular morphology contracts."""
    component = make_exponential()

    assert isinstance(
        component,
        MorphologicalComponent,
    )
    assert isinstance(
        component,
        DecomposableMorphology,
    )


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "flux": 0.0,
            "re": 4.0,
            "q": 0.7,
        },
        {
            "flux": 100.0,
            "re": 0.0,
            "q": 0.7,
        },
        {
            "flux": 100.0,
            "re": 4.0,
            "q": 1.2,
        },
    ],
)
def test_exponential_reuses_sersic_validation(
    parameters,
) -> None:
    """Invalid physical parameters must retain Sérsic validation."""
    with pytest.raises(ValueError):
        ExponentialComponent(
            theta=0.0,
            x0=10.0,
            y0=10.0,
            **parameters,
        )
