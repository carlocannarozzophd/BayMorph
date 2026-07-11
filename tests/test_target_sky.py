import numpy as np
import pytest

from baymorph.pipeline.preparation_steps import estimate_target_sky


def test_estimate_target_sky():
    yy, xx = np.indices((101, 101), dtype=float)

    radius = np.sqrt(
        (xx - 50.0) ** 2
        + ((yy - 50.0) / 0.7) ** 2
    )

    image = np.exp(-radius / 10.0)
    image[radius >= 40.0] = 0.01

    target_segment = radius <= 12.0

    geometry, radial_profile, plateau = estimate_target_sky(
        image,
        target_segment,
        min_radius=20.0,
        window_size=5,
        sigma_threshold=1.0,
    )

    assert geometry.x0 == pytest.approx(50.0, abs=0.1)
    assert geometry.y0 == pytest.approx(50.0, abs=0.1)
    assert geometry.q == pytest.approx(0.7, abs=0.05)

    assert radial_profile.radius.ndim == 1
    assert radial_profile.profile.ndim == 1
    assert 35.0 <= plateau.radius <= 45.0
    assert plateau.background_level == pytest.approx(
        0.01,
        abs=1e-6,
    )


def test_estimate_target_sky_applies_contaminant_mask():
    image = np.ones((51, 51))

    yy, xx = np.indices(image.shape)
    target_segment = (
        (xx - 25.0) ** 2
        + (yy - 25.0) ** 2
        <= 6.0**2
    )

    contaminant_mask = np.zeros_like(image, dtype=bool)
    contaminant_mask[:, :10] = True

    _, profile, _ = estimate_target_sky(
        image,
        target_segment,
        mask=contaminant_mask,
    )

    assert np.any(
        profile.n_pixels < profile.area
    )