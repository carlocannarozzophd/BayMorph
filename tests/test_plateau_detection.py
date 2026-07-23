import numpy as np
import pytest

from baymorph.core import (
    detect_profile_plateau,
    RadialProfile,
)



def make_radial_profile(
    values: np.ndarray,
    *,
    n_pixels: int = 100,
    area: int = 100,
) -> RadialProfile:
    values = np.asarray(values, dtype=float)
    size = values.size

    return RadialProfile(
        radius=np.arange(size, dtype=float),
        profile=values,
        n_pixels=np.full(size, n_pixels, dtype=int),
        scatter=np.zeros(size, dtype=float),
        area=np.full(size, area, dtype=int),
    )


def test_detect_plateau_in_flat_profile():
    radial_profile = make_radial_profile(
        np.ones(100),
    )

    result = detect_profile_plateau(
        radial_profile,
        window_size=5,
    )

    assert result.index == 0
    assert result.radius == pytest.approx(0.0)
    assert result.background_level == pytest.approx(1.0)
    assert result.window_start == 0
    assert result.window_stop == 5


def test_detect_plateau_after_declining_profile():
    declining = np.exp(
        -np.arange(40, dtype=float) / 8.0
    )
    plateau = np.full(60, 0.02)

    radial_profile = make_radial_profile(
        np.concatenate(
            [
                declining,
                plateau,
            ]
        )
    )

    result = detect_profile_plateau(
        radial_profile,
        min_radius=20.0,
        window_size=5,
        sigma_threshold=1.0,
    )

    assert 38 <= result.index <= 40
    assert result.background_level == pytest.approx(0.02)


def test_detect_noisy_plateau():
    rng = np.random.default_rng(42)

    declining = np.exp(
        -np.arange(40, dtype=float) / 8.0
    )
    plateau = (
        np.full(60, 0.02)
        + rng.normal(
            loc=0.0,
            scale=0.003,
            size=60,
        )
    )

    radial_profile = make_radial_profile(
        np.concatenate(
            [
                declining,
                plateau,
            ]
        )
    )

    result = detect_profile_plateau(
        radial_profile,
        min_radius=20.0,
        window_size=5,
        sigma_threshold=2.5,
    )

    assert 35 <= result.index <= 45
    assert result.background_level == pytest.approx(
        0.02,
        abs=0.01,
    )
    assert result.noise_sigma > 0.0


def test_plateau_detection_fails_without_stable_region():
    values = np.linspace(
        1.0,
        0.0,
        100,
    )

    radial_profile = make_radial_profile(values)

    with pytest.raises(
        ValueError,
        match="No stable radial-profile plateau",
    ):
        detect_profile_plateau(
            radial_profile,
            window_size=5,
            sigma_threshold=0.1,
        )


def test_plateau_detection_rejects_low_coverage():
    radial_profile = make_radial_profile(
        np.ones(100),
        n_pixels=10,
        area=100,
    )

    with pytest.raises(
        ValueError,
        match="Not enough valid radial bins",
    ):
        detect_profile_plateau(
            radial_profile,
            min_coverage=0.5,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"window_size": 1},
            "'window_size' must be at least 2.",
        ),
        (
            {"sigma_threshold": 0.0},
            "'sigma_threshold' must be positive.",
        ),
        (
            {"outer_fraction": 0.0},
            "'outer_fraction' must satisfy",
        ),
        (
            {"outer_fraction": 1.1},
            "'outer_fraction' must satisfy",
        ),
        (
            {"min_coverage": -0.1},
            "'min_coverage' must satisfy",
        ),
        (
            {"min_coverage": 1.1},
            "'min_coverage' must satisfy",
        ),
    ],
)
def test_plateau_detection_rejects_invalid_parameters(
    kwargs,
    message,
):
    radial_profile = make_radial_profile(
        np.ones(20),
    )

    with pytest.raises(
        ValueError,
        match=message,
    ):
        detect_profile_plateau(
            radial_profile,
            **kwargs,
        )


def test_plateau_detection_ignores_non_finite_bins():
    values = np.ones(30)
    values[:5] = np.nan

    radial_profile = make_radial_profile(values)

    result = detect_profile_plateau(
        radial_profile,
        window_size=5,
    )

    assert result.index == 5
    assert result.radius == pytest.approx(5.0)