"""Tests for the observation-effect protocol."""

from __future__ import annotations

import numpy as np

from baymorph.observation import ObservationEffect


class IdentityEffect:
    """Deterministic effect returning an independent image copy."""

    def apply(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(
            image,
            dtype=float,
        ).copy()


class IncompleteEffect:
    """Object that does not implement the observation-effect contract."""


def test_structurally_compatible_effect_satisfies_protocol() -> None:
    """Nominal inheritance must not be required."""
    effect = IdentityEffect()

    assert isinstance(
        effect,
        ObservationEffect,
    )


def test_incomplete_effect_does_not_satisfy_protocol() -> None:
    """Objects without apply() must fail the structural contract."""
    effect = IncompleteEffect()

    assert not isinstance(
        effect,
        ObservationEffect,
    )


def test_observation_effect_preserves_input_shape() -> None:
    """A valid deterministic effect can preserve detector geometry."""
    effect = IdentityEffect()
    image = np.arange(
        35,
        dtype=float,
    ).reshape(5, 7)

    result = effect.apply(image)

    assert result.shape == image.shape
    assert np.array_equal(
        result,
        image,
    )
    assert result is not image
