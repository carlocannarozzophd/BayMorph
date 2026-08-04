"""Tests for the joint multi-band image likelihood."""

import numpy as np
import pytest

from baymorph.likelihood import GaussianImageLikelihood
from baymorph.likelihood.multiband import MultiBandLikelihood


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


def test_multiband_likelihood_exposes_bands_and_shapes():
    likelihood = MultiBandLikelihood(
        make_likelihoods()
    )

    assert likelihood.bands == (
        "g",
        "r",
    )
    assert likelihood.image_shapes == {
        "g": (3, 3),
        "r": (2, 4),
    }


def test_multiband_likelihood_sums_image_factors():
    likelihoods = make_likelihoods()

    joint = MultiBandLikelihood(
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


def test_multiband_likelihood_requires_exact_bands():
    likelihood = MultiBandLikelihood(
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
