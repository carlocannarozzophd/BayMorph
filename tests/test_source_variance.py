"""Tests of the source Poisson variance of AD-047."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from baymorph.core.effective_gain import (
    SKY_LEVEL_GUARD,
    resolve_effective_gain,
)
from baymorph.fitting.image_problem import ImageInferenceProblem
from baymorph.fitting.parameters import Parameter, ParameterSet
from baymorph.likelihood import GaussianImageLikelihood, make_likelihood
from baymorph.likelihood.multi_observation import MultiObservationLikelihood
from baymorph.likelihood.vecchia_gaussian import (
    VecchiaCorrelatedGaussianImageLikelihood,
    build_vecchia_factorization,
)
from baymorph.observation.background import (
    BackgroundAppliedModel,
    ConstantBackground,
)

SKY_SCALE = 2.0

KERNEL = np.array(
    [
        [0.02, 0.10, 0.02],
        [0.15, 1.00, 0.15],
        [0.02, 0.10, 0.02],
    ]
)


def _scene(shape: tuple[int, int] = (6, 7), seed: int = 0):
    """Return data, a model with a residual background, and its source."""
    rng = np.random.default_rng(seed)
    source = rng.uniform(0.0, 40.0, size=shape)
    source[0, 0] = -2.0
    model = source + 0.3
    data = model + 3.0 * rng.normal(size=shape)

    return data, model, source


def _likelihood(data, **options) -> VecchiaCorrelatedGaussianImageLikelihood:
    return VecchiaCorrelatedGaussianImageLikelihood(
        data,
        SKY_SCALE,
        KERNEL,
        max_neighbors=8,
        **options,
    )


@pytest.mark.parametrize(
    ("declaration", "gain", "level"),
    [
        ({"source": "declared", "gain": 44.4}, 44.4, None),
        ({"source": "sky_level", "sky_level": 326.87}, 326.87 / SKY_SCALE**2, 326.87),
        ({"source": "measured_sky"}, 120.0 / SKY_SCALE**2, 120.0),
        ({"source": "absent"}, None, None),
    ],
)
def test_every_source_of_the_gain_resolves(declaration, gain, level):
    resolved = resolve_effective_gain(
        declaration,
        sky_scale=SKY_SCALE,
        measured_sky_level=120.0,
    )

    assert resolved.source == declaration["source"]
    assert resolved.sky_level == level

    if gain is None:
        assert resolved.value is None
    else:
        assert resolved.value == pytest.approx(gain, rel=1e-15)

    assert resolved.metadata()["sky_level_guard"] == SKY_LEVEL_GUARD


def test_the_guard_admits_a_sky_of_exactly_ten_scales():
    resolved = resolve_effective_gain(
        {"source": "sky_level", "sky_level": SKY_LEVEL_GUARD * SKY_SCALE},
        sky_scale=SKY_SCALE,
    )

    assert resolved.metadata()["sky_level_over_scale"] == SKY_LEVEL_GUARD


@pytest.mark.parametrize(
    "declaration",
    [
        {"source": "sky_level", "sky_level": 19.9},
        {"source": "measured_sky"},
    ],
)
def test_a_sky_below_the_guard_cannot_give_the_gain(declaration):
    with pytest.raises(ValueError, match="below the 10"):
        resolve_effective_gain(
            declaration,
            sky_scale=SKY_SCALE,
            measured_sky_level=0.3,
        )


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        ({}, "no default"),
        ({"source": "poisson"}, "no default"),
        ({"source": "declared"}, "gain must be a positive"),
        ({"source": "declared", "gain": -3.0}, "gain must be a positive"),
        ({"source": "declared", "gain": True}, "gain must be a positive"),
        ({"source": "declared", "gain": 4.0, "sky_level": 300.0}, "Unsupported"),
        ({"source": "sky_level"}, "sky_level must be a positive"),
        ({"source": "absent", "gain": 4.0}, "Unsupported"),
    ],
)
def test_invalid_declarations_are_rejected(declaration, message):
    with pytest.raises(ValueError, match=message):
        resolve_effective_gain(
            declaration,
            sky_scale=SKY_SCALE,
            measured_sky_level=120.0,
        )


def test_a_sky_source_needs_a_measured_level_and_one_sky_scale():
    with pytest.raises(ValueError, match="measured by the preparation"):
        resolve_effective_gain({"source": "measured_sky"}, sky_scale=SKY_SCALE)

    with pytest.raises(ValueError, match="one prepared sky scale"):
        resolve_effective_gain(
            {"source": "measured_sky"},
            sky_scale=None,
            measured_sky_level=120.0,
        )

    declared = resolve_effective_gain(
        {"source": "declared", "gain": 3.0},
        sky_scale=None,
    )

    assert declared.value == 3.0


def test_source_variance_likelihood_is_a_normalized_gaussian():
    data, model, source = _scene()
    gain = 4.0
    likelihood = _likelihood(data, effective_gain=gain)
    factorization = likelihood.factorization
    size = data.size
    weights = csr_matrix(
        (
            factorization.weights,
            factorization.neighbor_indices,
            factorization.neighbor_indptr,
        ),
        shape=(size, size),
    ).toarray()
    transform = np.eye(size) - weights
    precision = (
        transform.T
        @ np.diag(1.0 / factorization.conditional_variances)
        @ transform
    )
    scale = np.sqrt(SKY_SCALE**2 + np.clip(source.ravel(), 0.0, None) / gain)
    covariance = np.diag(scale) @ np.linalg.inv(precision) @ np.diag(scale)
    residual = (data - model).ravel()
    sign, logdet = np.linalg.slogdet(2.0 * np.pi * covariance)
    expected = -0.5 * residual @ np.linalg.solve(covariance, residual) - 0.5 * logdet

    assert sign > 0.0
    assert likelihood(model, source=source) == pytest.approx(expected, rel=1e-10)
    np.testing.assert_allclose(likelihood.marginal_scale(source), scale, rtol=1e-15)


def test_the_source_is_required_once_a_gain_is_set():
    data, model, _ = _scene()
    likelihood = _likelihood(data, effective_gain=4.0)

    assert likelihood.requires_source
    assert likelihood.approximation_metadata["effective_gain"] == 4.0

    with pytest.raises(ValueError, match="source part"):
        likelihood(model)

    with pytest.raises(ValueError, match="normalization cannot be disabled"):
        _likelihood(data, effective_gain=4.0, include_normalization=False)

    with pytest.raises(ValueError, match="positive finite"):
        _likelihood(data, effective_gain=0.0)


def test_without_a_gain_the_source_is_ignored():
    data, model, source = _scene()
    fixed = _likelihood(data)
    gaussian = GaussianImageLikelihood(data=data, noise=SKY_SCALE)

    assert not fixed.requires_source
    assert fixed(model, source=source) == fixed(model)
    assert gaussian(model, source=source) == gaussian(model)


class _Ramp:
    """A source image proportional to the sum of the pixel indices."""

    def __init__(self, amplitude: float) -> None:
        self.amplitude = float(amplitude)

    def render(self, shape):
        return self.amplitude * np.add.outer(
            np.arange(shape[0]),
            np.arange(shape[1]),
        ).astype(float)


def test_the_image_problem_passes_the_source_without_the_background():
    data, _, _ = _scene()
    likelihood = _likelihood(data, effective_gain=4.0)
    problem = ImageInferenceProblem(
        model_factory=lambda amplitude, level: BackgroundAppliedModel(
            model=_Ramp(amplitude),
            background=ConstantBackground(level=level),
        ),
        parameters=ParameterSet(
            [
                Parameter("amplitude", 0.0, 5.0),
                Parameter("level", -10.0, 10.0),
            ]
        ),
        likelihood=likelihood,
    )
    source = _Ramp(2.0).render(data.shape)
    expected = likelihood(source + 7.0, source=source)

    assert problem.log_likelihood(np.array([2.0, 7.0])) == expected
    assert expected != likelihood(source + 7.0, source=source + 7.0)


def test_the_joint_likelihood_hands_each_observation_its_source():
    data, model, source = _scene()
    with_gain = _likelihood(data, effective_gain=4.0)
    plain = GaussianImageLikelihood(data=data, noise=SKY_SCALE)
    joint = MultiObservationLikelihood({"g": plain, "r": with_gain})

    value = joint(
        {"g": model, "r": model},
        source_images={"g": source, "r": source},
    )

    assert joint.requires_source
    assert value == pytest.approx(
        plain(model) + with_gain(model, source=source),
        rel=1e-14,
    )

    with pytest.raises(ValueError, match="source part"):
        joint({"g": model, "r": model})


def test_the_factory_takes_the_declared_and_the_resolved_gain_together():
    data, _, _ = _scene()
    factorization = build_vecchia_factorization(
        np.zeros(data.shape, dtype=bool),
        KERNEL,
        max_neighbors=8,
    )
    config = {
        "type": "correlated_gaussian",
        "include_normalization": True,
        "factorization": {"type": "nearest_neighbor", "max_neighbors": 8},
        "effective_gain": {"source": "declared", "gain": 4.0},
    }
    arguments = {
        "data": data,
        "noise": SKY_SCALE,
        "correlation_kernel": KERNEL,
        "vecchia_factorization": factorization,
    }
    gain = resolve_effective_gain(config["effective_gain"], sky_scale=SKY_SCALE)
    absent = resolve_effective_gain({"source": "absent"}, sky_scale=SKY_SCALE)

    likelihood = make_likelihood(config, effective_gain=gain, **arguments)

    assert likelihood.effective_gain == 4.0

    with pytest.raises(ValueError, match="supplied together"):
        make_likelihood(config, **arguments)

    with pytest.raises(ValueError, match="does not match"):
        make_likelihood(config, effective_gain=absent, **arguments)

    with pytest.raises(ValueError, match="belongs to the correlated"):
        make_likelihood(
            {"type": "gaussian", "effective_gain": {"source": "absent"}},
            data=data,
            noise=SKY_SCALE,
            effective_gain=absent,
        )

    exact = {key: value for key, value in config.items() if key != "factorization"}

    with pytest.raises(ValueError, match="nearest_neighbor"):
        make_likelihood(
            exact,
            data=data,
            noise=SKY_SCALE,
            correlation_kernel=KERNEL,
            effective_gain=gain,
        )
