import numpy as np
import pytest

from baymorph.pipeline.real_fit import (
    RealFitProducts,
    _build_parameter_set,
    _make_model_factory,
    _compute_best_model,
    _compute_fit_statistics,
    _compute_bayesian_model_comparison,
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


class DeltaPSF:
    def kernel(self):
        psf = np.zeros((9, 9))
        psf[4, 4] = 1.0
        return psf


class DummyPosterior:
    def __init__(self, logz):
        self.logz = np.array([logz])


def test_compute_best_model_uses_best_parameters():
    def model_factory(a, b):
        return DummyModel(a, b)

    image, best_parameters = _compute_best_model(
        result=DummyResult(),
        parameter_names=["a", "b"],
        model_factory=model_factory,
        image_shape=(4, 5),
    )

    assert image.shape == (4, 5)
    assert best_parameters == {
        "a": 2.0,
        "b": 3.0,
    }
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


def test_build_parameter_set_single_sersic():
    parameter_names, parameters, _, _ = (
        _build_parameter_set(
            "single_sersic",
            data_fit=np.ones((10, 10)),
            noise_fit=np.ones((10, 10)),
            mask_fit=np.zeros((10, 10), dtype=bool),
            target_x=5,
            target_y=5,
        )
    )

    assert len(parameter_names) == 8
    assert parameter_names[0] == "flux"
    assert parameters.names[0] == "flux"


def test_build_parameter_set_double_sersic():
    parameter_names, parameters, _, _ = (
        _build_parameter_set(
            "double_sersic",
            data_fit=np.ones((10, 10)),
            noise_fit=np.ones((10, 10)),
            mask_fit=np.zeros((10, 10), dtype=bool),
            target_x=5,
            target_y=5,
        )
    )

    assert len(parameter_names) == 11
    assert parameter_names[0] == "flux_bulge"
    assert parameters.names[0] == "flux_bulge"


def test_make_model_factory_single_sersic():
    factory = _make_model_factory(
        "single_sersic",
        DeltaPSF(),
    )

    model = factory(
        100.0,
        5.0,
        2.0,
        0.8,
        0.0,
        15.0,
        15.0,
        1.0,
    )

    image = model.render((31, 31))

    assert image.shape == (31, 31)
    assert np.all(np.isfinite(image))


def test_make_model_factory_double_sersic():
    factory = _make_model_factory(
        "double_sersic",
        DeltaPSF(),
    )

    model = factory(
        100.0,
        3.0,
        2.0,
        50.0,
        8.0,
        1.0,
        0.8,
        0.0,
        15.0,
        15.0,
        1.0,
    )

    image = model.render((31, 31))

    assert image.shape == (31, 31)
    assert np.all(np.isfinite(image))


def test_compute_bayesian_model_comparison():
    single = RealFitProducts(
        posterior=DummyPosterior(10.0),
        best_parameters={},
        best_model_image=np.zeros((2, 2)),
        residual=np.zeros((2, 2)),
        norm_residual=np.zeros((2, 2)),
        chi2=1.0,
        ndof=1,
        reduced_chi2=1.0,
    )

    double = RealFitProducts(
        posterior=DummyPosterior(13.0),
        best_parameters={},
        best_model_image=np.zeros((2, 2)),
        residual=np.zeros((2, 2)),
        norm_residual=np.zeros((2, 2)),
        chi2=1.0,
        ndof=1,
        reduced_chi2=1.0,
    )

    comparison = _compute_bayesian_model_comparison(
        {
            "single_sersic": single,
            "double_sersic": double,
        }
    )

    assert comparison["logz"]["single_sersic"] == pytest.approx(10.0)
    assert comparison["logz"]["double_sersic"] == pytest.approx(13.0)
    assert comparison["delta_logz"]["double_sersic_minus_single_sersic"] == pytest.approx(3.0)
    assert comparison["bayes_factor"]["double_sersic_over_single_sersic"] == pytest.approx(
        np.exp(3.0)
    )