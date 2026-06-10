import numpy as np
import pytest

from baymorph import gaussian_log_likelihood


def test_gaussian_log_likelihood_is_finite():
    data = np.ones((10, 10))
    model = np.ones((10, 10))
    noise = np.ones((10, 10))

    loglike = gaussian_log_likelihood(
        data,
        model,
        noise,
    )

    assert np.isfinite(loglike)


def test_gaussian_log_likelihood_best_model_higher():
    data = np.ones((10, 10))
    good_model = np.ones((10, 10))
    bad_model = np.zeros((10, 10))
    noise = np.ones((10, 10))

    good_loglike = gaussian_log_likelihood(
        data,
        good_model,
        noise,
    )

    bad_loglike = gaussian_log_likelihood(
        data,
        bad_model,
        noise,
    )

    assert good_loglike > bad_loglike


def test_gaussian_log_likelihood_with_mask():
    data = np.ones((10, 10))
    model = np.ones((10, 10))
    noise = np.ones((10, 10))

    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0] = True

    loglike = gaussian_log_likelihood(
        data,
        model,
        noise,
        mask=mask,
    )

    assert np.isfinite(loglike)


def test_gaussian_log_likelihood_rejects_bad_noise():
    data = np.ones((10, 10))
    model = np.ones((10, 10))
    noise = np.zeros((10, 10))

    with pytest.raises(ValueError):
        gaussian_log_likelihood(
            data,
            model,
            noise,
        )