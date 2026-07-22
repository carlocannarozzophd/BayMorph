"""Tests for BayMorph image likelihoods."""

import numpy as np
import pytest

from baymorph import GaussianImageLikelihood


def test_gaussian_image_likelihood_is_finite():
    """The Gaussian image likelihood should return a finite value."""
    data = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    model = np.array(
        [
            [1.1, 1.9],
            [3.1, 3.9],
        ]
    )
    noise = np.ones_like(data)

    likelihood = GaussianImageLikelihood(
        data=data,
        noise=noise,
    )

    loglike = likelihood(model)

    assert np.isfinite(loglike)


def test_gaussian_image_likelihood_prefers_better_model():
    """A model closer to the data should have higher likelihood."""
    data = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    noise = np.ones_like(data)

    good_model = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    bad_model = np.zeros_like(data)

    likelihood = GaussianImageLikelihood(
        data=data,
        noise=noise,
    )

    good_loglike = likelihood(good_model)
    bad_loglike = likelihood(bad_model)

    assert good_loglike > bad_loglike


def test_gaussian_image_likelihood_with_mask():
    """Masked pixels should not contribute to the likelihood."""
    data = np.array(
        [
            [1.0, 1000.0],
            [3.0, 4.0],
        ]
    )
    model = np.array(
        [
            [1.0, -1000.0],
            [3.0, 4.0],
        ]
    )
    noise = np.ones_like(data)

    mask = np.array(
        [
            [False, True],
            [False, False],
        ]
    )

    likelihood = GaussianImageLikelihood(
        data=data,
        noise=noise,
        mask=mask,
    )

    loglike = likelihood(model)

    expected = -0.5 * 3.0 * np.log(
        2.0 * np.pi
    )

    assert np.isclose(
        loglike,
        expected,
    )


def test_gaussian_image_likelihood_rejects_bad_noise():
    """Noise values must be strictly positive."""
    data = np.ones(
        (2, 2),
        dtype=float,
    )

    noise = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        GaussianImageLikelihood(
            data=data,
            noise=noise,
        )


def test_gaussian_image_likelihood_rejects_wrong_model_shape():
    """The model image must match the observed image shape."""
    data = np.ones(
        (2, 2),
        dtype=float,
    )
    noise = np.ones_like(data)

    likelihood = GaussianImageLikelihood(
        data=data,
        noise=noise,
    )

    with pytest.raises(
        ValueError,
        match="same shape",
    ):
        likelihood(
            np.ones(
                (3, 3),
                dtype=float,
            )
        )