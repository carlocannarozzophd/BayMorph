"""Tests for intrinsic Sérsic plus point-source morphology."""

from __future__ import annotations

import pytest

from baymorph.models import (
    PointSourceModel,
    SersicModel,
    SersicPointSourceModel,
)


def make_sersic() -> SersicModel:
    """Return a valid Sérsic component."""
    return SersicModel(
        flux=100.0,
        re=4.0,
        n=2.0,
        q=0.8,
        theta=0.1,
        x0=10.25,
        y0=11.75,
    )


def make_point_source() -> PointSourceModel:
    """Return a point source sharing the Sérsic centre."""
    return PointSourceModel(
        flux=25.0,
        x0=10.25,
        y0=11.75,
    )


def test_sersic_point_source_stores_intrinsic_components() -> None:
    """The named morphology must preserve its two intrinsic components."""
    sersic = make_sersic()
    point_source = make_point_source()

    model = SersicPointSourceModel(
        sersic=sersic,
        point_source=point_source,
    )

    assert model.sersic is sersic
    assert model.point_source is point_source


def test_sersic_point_source_exposes_shared_centre() -> None:
    """The baseline model must expose its common centre."""
    model = SersicPointSourceModel(
        sersic=make_sersic(),
        point_source=make_point_source(),
    )

    assert model.x0 == pytest.approx(10.25)
    assert model.y0 == pytest.approx(11.75)


def test_sersic_point_source_reports_total_flux() -> None:
    """Total intrinsic flux must sum extended and unresolved components."""
    model = SersicPointSourceModel(
        sersic=make_sersic(),
        point_source=make_point_source(),
    )

    assert model.total_flux == pytest.approx(125.0)


def test_sersic_point_source_reports_point_source_fraction() -> None:
    """The unresolved fraction must follow directly from component fluxes."""
    model = SersicPointSourceModel(
        sersic=make_sersic(),
        point_source=make_point_source(),
    )

    assert model.point_source_fraction == pytest.approx(
        0.2
    )


def test_sersic_point_source_rejects_different_centres() -> None:
    """The baseline named model must enforce its shared-centre assumption."""
    point_source = PointSourceModel(
        flux=25.0,
        x0=10.5,
        y0=11.75,
    )

    with pytest.raises(
        ValueError,
        match="shared centre",
    ):
        SersicPointSourceModel(
            sersic=make_sersic(),
            point_source=point_source,
        )


def test_sersic_point_source_owns_no_psf() -> None:
    """Observation-specific PSFs must remain outside morphology."""
    model = SersicPointSourceModel(
        sersic=make_sersic(),
        point_source=make_point_source(),
    )

    assert not hasattr(model, "psf")


def test_sersic_point_source_has_no_intrinsic_image_renderer() -> None:
    """A morphology containing a Dirac component cannot render regularly."""
    model = SersicPointSourceModel(
        sersic=make_sersic(),
        point_source=make_point_source(),
    )

    assert not hasattr(model, "render")
