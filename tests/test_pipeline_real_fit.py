import numpy as np
import pytest

from baymorph.pipeline.real_fit import (
    _compute_best_model,
    _compute_fit_statistics,
)


class DummyResult:
    best_parameters = {
        "a": 2.0,
        "b": 3.0,
    }


class DummyModel:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def render(self, shape):
        return np.full(shape, self.a + self.b)


def test_compute_best_model_uses_best_parameters():
    def model_factory(a, b):
        return DummyModel(a, b)

    image = _compute_best_model(
        result=DummyResult(),
        parameter_names=["a", "b"],
        model_factory=model_factory,
        image_shape=(4, 5),
    )

    assert image.shape == (4, 5)
    assert np.all(image == 5.0)


def test_compute_fit_statistics_uses_unmasked_valid_pixels():
    data = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    model = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )

    noise = np.ones_like(data)

    mask = np.array(
        [
            [False, False],
            [True, False],
        ]
    )

    stats = _compute_fit_statistics(
        data=data,
        model_image=model,
        noise=noise,
        mask=mask,
        n_parameters=2,
    )

    assert stats["chi2"] == pytest.approx(10.0)
    assert stats["ndof"] == 1
    assert stats["reduced_chi2"] == pytest.approx(10.0)
    assert np.allclose(
        stats["residual"],
        data - model,
    )
    assert np.isnan(
        stats["norm_residual"][1, 0]
    )

    assert np.allclose(
        stats["norm_residual"][~mask],
        (data - model)[~mask],
)