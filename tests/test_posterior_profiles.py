import numpy as np
import pytest

from baymorph import (
    compute_posterior_profile_bands,
)


def test_compute_posterior_profile_bands_returns_expected_structure():
    posterior_samples = np.array(
        [
            [
                1000.0,
                5.0,
                4.0,
                500.0,
                10.0,
                1.0,
                0.8,
                0.0,
                25.0,
                25.0,
                0.0,
            ]
        ]
    )

    result = compute_posterior_profile_bands(
        posterior_samples=posterior_samples,
        shape=(51, 51),
        radii=np.arange(0, 20),
    )

    assert set(result.keys()) == {
        "radii",
        "radius",
        "inner",
        "outer",
        "total",
    }

    for component in ("inner", "outer", "total"):
        assert set(result[component].keys()) == {
            "p16",
            "p50",
            "p84",
        }


def test_compute_posterior_profile_bands_profile_lengths():
    posterior_samples = np.array(
        [
            [
                1000.0,
                5.0,
                4.0,
                500.0,
                10.0,
                1.0,
                0.8,
                0.0,
                25.0,
                25.0,
                0.0,
            ]
        ]
    )

    radii = np.arange(0, 20)

    result = compute_posterior_profile_bands(
        posterior_samples=posterior_samples,
        shape=(51, 51),
        radii=radii,
    )

    expected_length = len(radii) - 1

    for component in ("inner", "outer", "total"):
        assert result[component]["p16"].shape == (
            expected_length,
        )
        assert result[component]["p50"].shape == (
            expected_length,
        )
        assert result[component]["p84"].shape == (
            expected_length,
        )


def test_compute_posterior_profile_bands_rejects_wrong_parameter_count():
    posterior_samples = np.ones((3, 10))

    with pytest.raises(ValueError):
        compute_posterior_profile_bands(
            posterior_samples=posterior_samples,
            shape=(51, 51),
            radii=np.arange(0, 20),
        )


def test_compute_posterior_profile_bands_rejects_empty_samples():
    posterior_samples = np.empty((0, 11))

    with pytest.raises(ValueError):
        compute_posterior_profile_bands(
            posterior_samples=posterior_samples,
            shape=(51, 51),
            radii=np.arange(0, 20),
        )


def test_compute_posterior_profile_bands_ignores_sky_for_morphology_profiles():
    base_sample = np.array(
        [
            1000.0,
            5.0,
            4.0,
            500.0,
            10.0,
            1.0,
            0.8,
            0.0,
            25.0,
            25.0,
            0.0,
        ]
    )

    sample_with_sky = base_sample.copy()
    sample_with_sky[-1] = 100.0

    radii = np.arange(0, 20)

    result_no_sky = compute_posterior_profile_bands(
        posterior_samples=base_sample[None, :],
        shape=(51, 51),
        radii=radii,
    )

    result_with_sky = compute_posterior_profile_bands(
        posterior_samples=sample_with_sky[None, :],
        shape=(51, 51),
        radii=radii,
    )

    for component in ("inner", "outer", "total"):
        for percentile in ("p16", "p50", "p84"):
            np.testing.assert_allclose(
                result_with_sky[component][percentile],
                result_no_sky[component][percentile],
            )


def test_render_components_helper_is_not_public_api():
    import baymorph

    assert not hasattr(
        baymorph,
        "render_components_from_values",
    )
    assert not hasattr(
        baymorph,
        "_render_components_from_values",
    )


def test_total_profile_is_not_smaller_than_component_profiles():
    posterior_samples = np.array(
        [
            [
                1000.0,
                5.0,
                4.0,
                500.0,
                10.0,
                1.0,
                0.8,
                0.0,
                25.0,
                25.0,
                0.0,
            ]
        ]
    )

    result = compute_posterior_profile_bands(
        posterior_samples=posterior_samples,
        shape=(51, 51),
        radii=np.arange(0, 20),
    )

    np.testing.assert_array_less(
        result["inner"]["p50"],
        result["total"]["p50"] + 1e-12,
    )
    np.testing.assert_array_less(
        result["outer"]["p50"],
        result["total"]["p50"] + 1e-12,
    )


def test_compute_posterior_profile_bands_rejects_non_finite_radii():
    posterior_samples = np.ones((1, 11))

    with pytest.raises(ValueError):
        compute_posterior_profile_bands(
            posterior_samples=posterior_samples,
            shape=(51, 51),
            radii=np.array([0.0, 1.0, np.nan, 3.0]),
        )


def test_compute_posterior_profile_bands_rejects_non_increasing_radii():
    posterior_samples = np.ones((1, 11))

    with pytest.raises(ValueError):
        compute_posterior_profile_bands(
            posterior_samples=posterior_samples,
            shape=(51, 51),
            radii=np.array([0.0, 1.0, 1.0, 3.0]),
        )


def test_compute_posterior_profile_bands_returns_radii():
    posterior_samples = np.array(
        [
            [
                1000.0,
                5.0,
                4.0,
                500.0,
                10.0,
                1.0,
                0.8,
                0.0,
                25.0,
                25.0,
                0.0,
            ]
        ]
    )

    radii = np.arange(0, 20)

    result = compute_posterior_profile_bands(
        posterior_samples=posterior_samples,
        shape=(51, 51),
        radii=radii,
    )

    assert "radii" in result

    np.testing.assert_allclose(
        result["radii"],
        radii,
    )


def test_compute_posterior_profile_bands_returns_radius_bin_centers():
    posterior_samples = np.array(
        [
            [
                1000.0,
                5.0,
                4.0,
                500.0,
                10.0,
                1.0,
                0.8,
                0.0,
                25.0,
                25.0,
                0.0,
            ]
        ]
    )

    radii = np.array([0.0, 1.0, 3.0, 6.0])

    result = compute_posterior_profile_bands(
        posterior_samples=posterior_samples,
        shape=(51, 51),
        radii=radii,
    )

    np.testing.assert_allclose(
        result["radius"],
        np.array([0.5, 2.0, 4.5]),
    )


def test_compute_posterior_profile_bands_accepts_fixed_profile_center():
    posterior_samples = np.array(
        [
            [
                1000.0,
                5.0,
                4.0,
                500.0,
                10.0,
                1.0,
                0.8,
                0.0,
                20.0,
                20.0,
                0.0,
            ]
        ]
    )

    result = compute_posterior_profile_bands(
        posterior_samples=posterior_samples,
        shape=(51, 51),
        radii=np.arange(0, 20),
        profile_center=(25.0, 25.0),
    )

    assert "total" in result