import numpy as np
import pytest

from baymorph.analysis import (
    PosteriorProfileResult,
    ProfileBand,
    compute_posterior_profile_bands,
)


class DummyModel:
    def __init__(self, amplitude_a, amplitude_b, names=("component_a", "component_b")):
        self.amplitude_a = float(amplitude_a)
        self.amplitude_b = float(amplitude_b)
        self.names = names

    def render_components(self, shape):
        return {
            self.names[0]: np.full(shape, self.amplitude_a, dtype=float),
            self.names[1]: np.full(shape, self.amplitude_b, dtype=float),
        }

    def render(self, shape):
        components = self.render_components(shape)
        image = np.zeros(shape, dtype=float)

        for component in components.values():
            image += component

        return image


def dummy_model_factory(sample):
    return DummyModel(
        amplitude_a=sample[0],
        amplitude_b=sample[1],
    )


def test_compute_posterior_profile_bands_returns_result_object():
    samples = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    result = compute_posterior_profile_bands(
        posterior=samples,
        model_factory=dummy_model_factory,
        image_shape=(21, 21),
        radii=np.arange(0.0, 6.0),
        center_x=10.0,
        center_y=10.0,
    )

    assert isinstance(result, PosteriorProfileResult)
    assert isinstance(result.total, ProfileBand)
    assert isinstance(result.components["component_a"], ProfileBand)
    assert isinstance(result.components["component_b"], ProfileBand)


def test_compute_posterior_profile_bands_returns_radius_bin_centers():
    samples = np.array(
        [
            [1.0, 2.0],
        ]
    )

    radii = np.array([0.0, 1.0, 3.0, 6.0])

    result = compute_posterior_profile_bands(
        posterior=samples,
        model_factory=dummy_model_factory,
        image_shape=(21, 21),
        radii=radii,
        center_x=10.0,
        center_y=10.0,
    )

    np.testing.assert_allclose(
        result.radius,
        np.array([0.5, 2.0, 4.5]),
    )


def test_compute_posterior_profile_bands_profile_lengths():
    samples = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    radii = np.arange(0.0, 8.0)

    result = compute_posterior_profile_bands(
        posterior=samples,
        model_factory=dummy_model_factory,
        image_shape=(21, 21),
        radii=radii,
        center_x=10.0,
        center_y=10.0,
    )

    expected_length = len(radii) - 1

    assert result.total.p16.shape == (expected_length,)
    assert result.total.p50.shape == (expected_length,)
    assert result.total.p84.shape == (expected_length,)

    for component in result.components.values():
        assert component.p16.shape == (expected_length,)
        assert component.p50.shape == (expected_length,)
        assert component.p84.shape == (expected_length,)


def test_compute_posterior_profile_bands_propagates_component_names():
    samples = np.array(
        [
            [1.0, 2.0],
        ]
    )

    result = compute_posterior_profile_bands(
        posterior=samples,
        model_factory=dummy_model_factory,
        image_shape=(21, 21),
        radii=np.arange(0.0, 6.0),
        center_x=10.0,
        center_y=10.0,
    )

    assert set(result.components) == {
        "component_a",
        "component_b",
    }


def test_compute_posterior_profile_bands_percentiles_are_ordered():
    samples = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    result = compute_posterior_profile_bands(
        posterior=samples,
        model_factory=dummy_model_factory,
        image_shape=(21, 21),
        radii=np.arange(0.0, 6.0),
        center_x=10.0,
        center_y=10.0,
    )

    np.testing.assert_array_less(
        result.total.p16,
        result.total.p50 + 1e-12,
    )
    np.testing.assert_array_less(
        result.total.p50,
        result.total.p84 + 1e-12,
    )


def test_total_profile_is_not_smaller_than_component_profiles():
    samples = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    result = compute_posterior_profile_bands(
        posterior=samples,
        model_factory=dummy_model_factory,
        image_shape=(21, 21),
        radii=np.arange(0.0, 6.0),
        center_x=10.0,
        center_y=10.0,
    )

    for component in result.components.values():
        np.testing.assert_array_less(
            component.p50,
            result.total.p50 + 1e-12,
        )


def test_compute_posterior_profile_bands_accepts_mask():
    samples = np.array(
        [
            [1.0, 2.0],
        ]
    )

    mask = np.zeros((21, 21), dtype=bool)
    mask[0, 0] = True

    result = compute_posterior_profile_bands(
        posterior=samples,
        model_factory=dummy_model_factory,
        image_shape=(21, 21),
        radii=np.arange(0.0, 6.0),
        center_x=10.0,
        center_y=10.0,
        mask=mask,
    )

    assert isinstance(result, PosteriorProfileResult)


def test_compute_posterior_profile_bands_rejects_empty_samples():
    samples = np.empty((0, 2))

    with pytest.raises(ValueError):
        compute_posterior_profile_bands(
            posterior=samples,
            model_factory=dummy_model_factory,
            image_shape=(21, 21),
            radii=np.arange(0.0, 6.0),
            center_x=10.0,
            center_y=10.0,
        )


def test_compute_posterior_profile_bands_rejects_non_2d_samples():
    samples = np.ones(2)

    with pytest.raises(ValueError):
        compute_posterior_profile_bands(
            posterior=samples,
            model_factory=dummy_model_factory,
            image_shape=(21, 21),
            radii=np.arange(0.0, 6.0),
            center_x=10.0,
            center_y=10.0,
        )


def test_compute_posterior_profile_bands_rejects_changing_component_names():
    samples = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    def changing_factory(sample):
        if sample[0] < 2.0:
            return DummyModel(sample[0], sample[1], names=("a", "b"))

        return DummyModel(sample[0], sample[1], names=("a", "c"))

    with pytest.raises(ValueError):
        compute_posterior_profile_bands(
            posterior=samples,
            model_factory=changing_factory,
            image_shape=(21, 21),
            radii=np.arange(0.0, 6.0),
            center_x=10.0,
            center_y=10.0,
            max_samples=None,
        )