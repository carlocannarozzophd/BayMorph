import numpy as np
import pytest

from baymorph import (
    PlateauResult,
    SkyRegion,
    build_sky_region,
)
from baymorph.pipeline.preparation_steps import TargetGeometry


def make_plateau(radius: float = 20.0) -> PlateauResult:
    return PlateauResult(
        radius=radius,
        index=int(radius),
        background_level=0.0,
        noise_sigma=1.0,
        window_start=int(radius),
        window_stop=int(radius) + 5,
    )


def test_build_sky_region():
    geometry = TargetGeometry(
        x0=25.0,
        y0=25.0,
        q=1.0,
        theta=0.0,
    )

    result = build_sky_region(
        shape=(51, 51),
        geometry=geometry,
        plateau=make_plateau(),
        margin=5.0,
    )

    assert isinstance(result, SkyRegion)
    assert result.inner_radius == pytest.approx(25.0)
    assert result.n_pixels > 0
    assert np.all(
        result.radius_map[result.mask] >= result.inner_radius
    )
    assert 0.0 < result.coverage <= 1.0


def test_build_sky_region_applies_mask():
    geometry = TargetGeometry(
        x0=25.0,
        y0=25.0,
        q=1.0,
        theta=0.0,
    )

    excluded = np.zeros((51, 51), dtype=bool)
    excluded[:, :25] = True

    result = build_sky_region(
        shape=(51, 51),
        geometry=geometry,
        plateau=make_plateau(),
        margin=5.0,
        mask=excluded,
    )

    assert not np.any(result.mask & excluded)
    assert result.coverage < 1.0


def test_build_sky_region_rejects_negative_margin():
    geometry = TargetGeometry(
        x0=10.0,
        y0=10.0,
        q=1.0,
        theta=0.0,
    )

    with pytest.raises(
        ValueError,
        match="margin",
    ):
        build_sky_region(
            shape=(21, 21),
            geometry=geometry,
            plateau=make_plateau(radius=5.0),
            margin=-1.0,
        )


def test_build_sky_region_rejects_invalid_outer_radius():
    geometry = TargetGeometry(
        x0=10.0,
        y0=10.0,
        q=1.0,
        theta=0.0,
    )

    with pytest.raises(
        ValueError,
        match="outer_radius",
    ):
        build_sky_region(
            shape=(21, 21),
            geometry=geometry,
            plateau=make_plateau(radius=5.0),
            margin=5.0,
            outer_radius=9.0,
        )
