"""Tests for deterministic observation-model composition."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.observation import ObservationModel


class AddEffect:
    """Add a constant value to an image."""

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def apply(self, image: np.ndarray) -> np.ndarray:
        return image + self.value


class MultiplyEffect:
    """Multiply an image by a constant value."""

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def apply(self, image: np.ndarray) -> np.ndarray:
        return image * self.value


def test_empty_observation_model_returns_independent_copy() -> None:
    """An empty observation model must act as the identity."""
    image = np.arange(12, dtype=float).reshape(3, 4)

    observed = ObservationModel().observe(image)

    assert np.array_equal(observed, image)
    assert observed is not image


def test_observation_effects_are_applied_in_order() -> None:
    """Effect ordering must be explicit and deterministic."""
    image = np.ones((3, 4), dtype=float)

    model = ObservationModel(
        [
            AddEffect(2.0),
            MultiplyEffect(3.0),
        ]
    )

    observed = model.observe(image)

    assert np.allclose(observed, 9.0)


def test_reversing_effect_order_changes_result() -> None:
    """Observation-effect composition must not be treated as commutative."""
    image = np.ones((3, 4), dtype=float)

    first = ObservationModel(
        [
            AddEffect(2.0),
            MultiplyEffect(3.0),
        ]
    ).observe(image)

    second = ObservationModel(
        [
            MultiplyEffect(3.0),
            AddEffect(2.0),
        ]
    ).observe(image)

    assert not np.array_equal(first, second)


def test_observation_model_rejects_non_effect() -> None:
    """Every configured effect must implement apply()."""
    with pytest.raises(
        TypeError,
        match=r"must implement apply",
    ):
        ObservationModel([object()])


@pytest.mark.parametrize(
    "image",
    [
        np.zeros(5),
        np.zeros((2, 3, 4)),
        np.full((3, 3), np.nan),
        np.full((3, 3), np.inf),
    ],
)
def test_observation_model_rejects_invalid_input(image) -> None:
    """Expected detector inputs must be finite two-dimensional arrays."""
    with pytest.raises(ValueError):
        ObservationModel().observe(image)


def test_observation_model_rejects_invalid_effect_output() -> None:
    """Effects must preserve the deterministic image contract."""

    class InvalidEffect:
        def apply(self, image: np.ndarray) -> np.ndarray:
            del image
            return np.full((2, 2), np.nan)

    model = ObservationModel([InvalidEffect()])

    with pytest.raises(
        ValueError,
        match=r"only finite",
    ):
        model.observe(
            np.ones((2, 2), dtype=float)
        )
