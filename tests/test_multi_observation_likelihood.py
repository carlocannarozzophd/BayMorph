"""Tests for the joint multi-band image likelihood."""

import numpy as np
import pytest

from baymorph.likelihood import GaussianImageLikelihood
from baymorph.likelihood.multi_observation import MultiObservationLikelihood


def make_likelihoods() -> dict[str, GaussianImageLikelihood]:
    return {
        "g": GaussianImageLikelihood(
            data=np.zeros((3, 3)),
            noise=np.ones((3, 3)),
        ),
        "r": GaussianImageLikelihood(
            data=np.ones((2, 4)),
            noise=np.ones((2, 4)),
        ),
    }


def test_multi_observation_likelihood_exposes_bands_and_shapes():
    likelihood = MultiObservationLikelihood(
        make_likelihoods()
    )

    assert likelihood.observations == (
        "g",
        "r",
    )
    assert likelihood.image_shapes == {
        "g": (3, 3),
        "r": (2, 4),
    }


def test_multi_observation_likelihood_sums_image_factors():
    likelihoods = make_likelihoods()

    joint = MultiObservationLikelihood(
        likelihoods
    )

    images = {
        "g": np.zeros((3, 3)),
        "r": np.ones((2, 4)),
    }

    expected = (
        likelihoods["g"](images["g"])
        + likelihoods["r"](images["r"])
    )

    assert joint(images) == pytest.approx(expected)


def test_multi_observation_likelihood_requires_exact_bands():
    likelihood = MultiObservationLikelihood(
        make_likelihoods()
    )

    with pytest.raises(
        ValueError,
        match="exactly the configured bands",
    ):
        likelihood(
            {
                "g": np.zeros((3, 3)),
            }
        )
