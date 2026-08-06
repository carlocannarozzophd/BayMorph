"""Tests for intrinsic unresolved point sources."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.models import PointSourceModel


def test_point_source_stores_intrinsic_parameters() -> None:
    """The intrinsic model must contain only source properties."""
    model = PointSourceModel(
        flux=100.0,
        x0=10.25,
        y0=11.75,
    )

    assert model.flux == 100.0
    assert model.x0 == 10.25
    assert model.y0 == 11.75


def test_point_source_owns_no_psf() -> None:
    """Observation-specific PSFs must not belong to morphology."""
    model = PointSourceModel(
        flux=100.0,
        x0=10.0,
        y0=11.0,
    )

    assert not hasattr(model, "psf")
    assert not hasattr(model, "kernel_size")


def test_point_source_has_no_regular_intrinsic_renderer() -> None:
    """A Dirac distribution must not masquerade as a pixelized image."""
    model = PointSourceModel(
        flux=100.0,
        x0=10.0,
        y0=11.0,
    )

    assert not hasattr(model, "render")


@pytest.mark.parametrize(
    ("flux", "x0", "y0"),
    [
        (0.0, 5.0, 5.0),
        (-1.0, 5.0, 5.0),
        (np.nan, 5.0, 5.0),
        (1.0, np.nan, 5.0),
        (1.0, 5.0, np.inf),
    ],
)
def test_point_source_rejects_invalid_parameters(
    flux: float,
    x0: float,
    y0: float,
) -> None:
    """Invalid intrinsic parameters must fail explicitly."""
    with pytest.raises(ValueError):
        PointSourceModel(
            flux=flux,
            x0=x0,
            y0=y0,
        )
