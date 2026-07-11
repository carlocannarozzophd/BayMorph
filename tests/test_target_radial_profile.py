import numpy as np
import pytest

from baymorph.pipeline.preparation_steps import (
    TargetGeometry,
    build_target_radial_profile,
)


def test_build_target_radial_profile_constant_image():
    image = np.full((41, 41), 5.0)

    geometry = TargetGeometry(
        x0=20.0,
        y0=20.0,
        q=1.0,
        theta=0.0,
    )

    result = build_target_radial_profile(
        image,
        geometry,
    )

    assert result.radius.ndim == 1
    assert np.allclose(result.profile, 5.0)
    assert np.all(result.n_pixels > 0)


def test_build_target_radial_profile_uses_geometry():
    yy, xx = np.indices((61, 61), dtype=float)

    image = np.sqrt(
        (xx - 30.0) ** 2
        + ((yy - 30.0) / 0.5) ** 2
    )

    geometry = TargetGeometry(
        x0=30.0,
        y0=30.0,
        q=0.5,
        theta=0.0,
    )

    result = build_target_radial_profile(
        image,
        geometry,
        statistic="median",
    )

    finite = np.isfinite(result.profile)

    assert np.allclose(
        result.profile[finite],
        result.radius[finite] + 0.5,
        atol=0.6,
    )


def test_build_target_radial_profile_applies_mask():
    image = np.ones((31, 31))

    geometry = TargetGeometry(
        x0=15.0,
        y0=15.0,
        q=1.0,
        theta=0.0,
    )

    mask = np.zeros_like(image, dtype=bool)
    mask[:, :15] = True

    unmasked = build_target_radial_profile(
        image,
        geometry,
    )
    masked = build_target_radial_profile(
        image,
        geometry,
        mask=mask,
    )

    assert np.all(
        masked.n_pixels <= unmasked.n_pixels
    )
    assert np.any(
        masked.n_pixels < unmasked.n_pixels
    )


def test_build_target_radial_profile_forwards_min_pixels():
    image = np.ones((21, 21))

    geometry = TargetGeometry(
        x0=10.0,
        y0=10.0,
        q=1.0,
        theta=0.0,
    )

    result = build_target_radial_profile(
        image,
        geometry,
        min_pixels=10_000,
    )

    assert np.all(np.isnan(result.profile))


def test_build_target_radial_profile_forwards_statistic():
    image = np.ones((21, 21))

    geometry = TargetGeometry(
        x0=10.0,
        y0=10.0,
        q=1.0,
        theta=0.0,
    )

    with pytest.raises(
        ValueError,
        match="Unknown statistic",
    ):
        build_target_radial_profile(
            image,
            geometry,
            statistic="banana",
        )