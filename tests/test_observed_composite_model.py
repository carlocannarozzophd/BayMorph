"""Tests for observed composite source models."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.models import PointSourceModel, SersicModel
from baymorph.observation.composite import ObservedCompositeModel
from baymorph.simulation.psf import GaussianPSF


def make_sersic() -> SersicModel:
    """Return a valid extended Sérsic component."""
    return SersicModel(
        flux=100.0,
        re=4.0,
        n=2.0,
        q=0.8,
        theta=0.1,
        x0=15.0,
        y0=15.0,
    )


def make_point_source() -> PointSourceModel:
    """Return a valid unresolved source."""
    return PointSourceModel(
        flux=25.0,
        x0=15.25,
        y0=14.75,
    )


def test_observed_composite_combines_extended_and_point_source() -> None:
    """Observed components must sum exactly to the total image."""
    model = ObservedCompositeModel(
        extended_model=make_sersic(),
        point_sources={
            "nucleus": make_point_source(),
        },
        psf=GaussianPSF(fwhm=2.5),
        kernel_size=15,
    )

    components = model.render_components(
        shape=(31, 31),
    )
    total = model.render(
        shape=(31, 31),
    )

    assert set(components) == {
        "sersic",
        "nucleus",
    }

    np.testing.assert_allclose(
        total,
        components["sersic"] + components["nucleus"],
        rtol=0.0,
        atol=0.0,
    )


def test_point_source_is_not_convolved_twice() -> None:
    """The unresolved contribution must equal direct PSF rendering."""
    source = make_point_source()

    model = ObservedCompositeModel(
        extended_model=make_sersic(),
        point_sources={
            "nucleus": source,
        },
        psf=GaussianPSF(fwhm=2.5),
        kernel_size=15,
    )

    components = model.render_components(
        shape=(31, 31),
    )

    expected = model.point_source_renderer.render(
        source,
        shape=(31, 31),
    )

    np.testing.assert_allclose(
        components["nucleus"],
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_observed_composite_supports_only_point_sources() -> None:
    """An observed model may contain no extended morphology."""
    model = ObservedCompositeModel(
        point_sources={
            "source": make_point_source(),
        },
        psf=GaussianPSF(fwhm=2.5),
        kernel_size=15,
    )

    components = model.render_components(
        shape=(31, 31),
    )

    assert set(components) == {
        "source",
    }

    np.testing.assert_allclose(
        model.render((31, 31)),
        components["source"],
        rtol=0.0,
        atol=0.0,
    )


def test_observed_composite_supports_only_extended_morphology() -> None:
    """The wrapper must remain valid without unresolved sources."""
    model = ObservedCompositeModel(
        extended_model=make_sersic(),
        psf=GaussianPSF(fwhm=2.5),
        kernel_size=15,
    )

    components = model.render_components(
        shape=(31, 31),
    )

    assert set(components) == {
        "sersic",
    }


def test_observed_composite_rejects_empty_model() -> None:
    """At least one source contribution must be present."""
    with pytest.raises(
        ValueError,
        match="at least one source",
    ):
        ObservedCompositeModel(
            psf=GaussianPSF(fwhm=2.5),
        )


def test_observed_composite_rejects_invalid_point_source() -> None:
    """Singular components must be explicit PointSourceModel objects."""
    with pytest.raises(
        TypeError,
        match="PointSourceModel",
    ):
        ObservedCompositeModel(
            point_sources={
                "invalid": object(),
            },
            psf=GaussianPSF(fwhm=2.5),
        )
