from types import SimpleNamespace

import numpy as np

from baymorph.pipeline.radial_profile import _posterior_samples


def test_posterior_samples_prefers_equal_weight_samples():
    posterior = SimpleNamespace(
        posterior_samples=np.array([[1.0, 2.0]]),
        samples=np.array([[9.0, 9.0]]),
    )

    result = _posterior_samples(posterior)

    assert np.array_equal(
        result,
        posterior.posterior_samples,
    )


def test_posterior_samples_falls_back_to_raw_samples():
    posterior = SimpleNamespace(
        samples=np.array([[3.0, 4.0]]),
    )

    result = _posterior_samples(posterior)

    assert np.array_equal(
        result,
        posterior.samples,
    )