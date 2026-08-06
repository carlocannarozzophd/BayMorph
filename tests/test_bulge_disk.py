"""Tests for intrinsic bulge-disc morphology."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.models import (
    BulgeDiskModel,
    CompositeModel,
    DecomposableMorphology,
    ExponentialComponent,
    MorphologicalComponent,
    SersicModel,
)


def make_bulge() -> SersicModel:
    """Return a valid Sérsic bulge."""
    return SersicModel(
        flux=300.0,
        re=3.0,
        n=3.5,
        q=0.8,
        theta=0.1,
        x0=20.0,
        y0=21.0,
    )


def make_disk() -> ExponentialComponent:
    """Return a valid exponential disc."""
    return ExponentialComponent(
        flux=700.0,
        re=10.0,
        q=0.6,
        theta=0.2,
        x0=20.0,
        y0=21.0,
    )


def test_bulge_disk_preserves_component_types() -> None:
    """The model must retain explicit bulge and disc semantics."""
    bulge = make_bulge()
    disk = make_disk()

    model = BulgeDiskModel(
        bulge=bulge,
        disk=disk,
    )

    assert model.bulge is bulge
    assert model.disk is disk
    assert model.disk.n == pytest.approx(1.0)


def test_bulge_disk_matches_named_composite() -> None:
    """Rendering must equal the delegated additive composition."""
    bulge = make_bulge()
    disk = make_disk()

    model = BulgeDiskModel(
        bulge=bulge,
        disk=disk,
    )

    reference = CompositeModel(
        {
            "bulge": bulge,
            "disk": disk,
        }
    )

    np.testing.assert_allclose(
        model.render(shape=(41, 43)),
        reference.render(shape=(41, 43)),
        rtol=0.0,
        atol=0.0,
    )


def test_bulge_disk_components_sum_to_total() -> None:
    """Named component images must reproduce the total morphology."""
    model = BulgeDiskModel(
        bulge=make_bulge(),
        disk=make_disk(),
    )

    components = model.render_components(
        shape=(41, 43),
    )
    total = model.render(
        shape=(41, 43),
    )

    assert set(components) == {
        "bulge",
        "disk",
    }

    np.testing.assert_allclose(
        total,
        components["bulge"] + components["disk"],
        rtol=0.0,
        atol=0.0,
    )


def test_bulge_disk_respects_render_origin() -> None:
    """Component composition must preserve global coordinate origins."""
    model = BulgeDiskModel(
        bulge=make_bulge(),
        disk=make_disk(),
    )

    components = model.render_components(
        shape=(31, 33),
        origin=(-5.0, -7.0),
    )
    total = model.render(
        shape=(31, 33),
        origin=(-5.0, -7.0),
    )

    np.testing.assert_allclose(
        total,
        components["bulge"] + components["disk"],
        rtol=0.0,
        atol=0.0,
    )


def test_bulge_disk_reports_intrinsic_flux_ratios() -> None:
    """Flux summaries must follow directly from component fluxes."""
    model = BulgeDiskModel(
        bulge=make_bulge(),
        disk=make_disk(),
    )

    assert model.total_flux == pytest.approx(1000.0)
    assert model.bulge_to_total == pytest.approx(0.3)


def test_bulge_disk_satisfies_regular_model_protocols() -> None:
    """Bulge-disc morphology must expose total and component rendering."""
    model = BulgeDiskModel(
        bulge=make_bulge(),
        disk=make_disk(),
    )

    assert isinstance(
        model,
        MorphologicalComponent,
    )
    assert isinstance(
        model,
        DecomposableMorphology,
    )


def test_bulge_disk_rejects_non_exponential_disk() -> None:
    """The physical disc component must have fixed Sérsic index one."""
    invalid_disk = SersicModel(
        flux=700.0,
        re=10.0,
        n=1.5,
        q=0.6,
        theta=0.2,
        x0=20.0,
        y0=21.0,
    )

    with pytest.raises(
        TypeError,
        match="ExponentialComponent disk",
    ):
        BulgeDiskModel(
            bulge=make_bulge(),
            disk=invalid_disk,
        )


def test_bulge_disk_rejects_exponential_bulge() -> None:
    """Bulge and disc roles must not be silently interchangeable."""
    with pytest.raises(
        TypeError,
        match="general SersicModel",
    ):
        BulgeDiskModel(
            bulge=make_disk(),
            disk=make_disk(),
        )
