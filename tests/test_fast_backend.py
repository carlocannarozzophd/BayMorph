"""Tests of the fast backend (AD-057) against known posteriors."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from baymorph.fitting import make_sampler
from baymorph.fitting.fast import (
    FAST_BACKEND_NAME,
    FLAG_SEVERAL_MODES,
    FastBackend,
    resolve_fast_settings,
)
from baymorph.fitting.problem import InferenceProblem
from baymorph.stats import weighted_quantile


class BoxProblem(InferenceProblem):
    """Uniform priors on a box and an analytic log-likelihood."""

    def __init__(self, lower, upper, log_likelihood, cyclic=()):
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self._log_likelihood = log_likelihood
        self._cyclic = tuple(cyclic)

    @property
    def cyclic_coordinates(self) -> tuple[int, ...]:
        return self._cyclic

    @property
    def ndim(self) -> int:
        return int(self.lower.size)

    @property
    def parameter_names(self) -> list[str]:
        return [f"p{index}" for index in range(self.ndim)]

    def prior_transform(self, cube):
        return self.lower + (self.upper - self.lower) * np.asarray(cube, dtype=float)

    def log_likelihood(self, theta) -> float:
        return float(self._log_likelihood(np.asarray(theta, dtype=float)))


def _fit(problem, **settings):
    config = {
        "name": FAST_BACKEND_NAME,
        "execution": {"processes": 1, "progress": False},
        **settings,
    }

    return make_sampler(config, problem=problem).fit()


def _quantiles(result, index, levels=(0.16, 0.5, 0.84)):
    return [
        weighted_quantile(result.samples[:, index], level, result.weights)
        for level in levels
    ]


MEANS = np.array([3.0, 5.0, 7.0])
SIGMAS = np.array([0.1, 0.2, 0.05])


@pytest.fixture(scope="module")
def gaussian_result():
    problem = BoxProblem(
        [0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0],
        lambda theta: -0.5 * np.sum(((theta - MEANS) / SIGMAS) ** 2),
    )

    return _fit(problem)


def test_gaussian_posterior_is_recovered(gaussian_result):
    diagnostics = gaussian_result.diagnostics
    effective = diagnostics["effective_sample_size"]

    assert diagnostics["pareto_k"] < 0.5
    assert diagnostics["flags"] == []
    assert diagnostics["reliable"] is True

    for index, (mean, sigma) in enumerate(zip(MEANS, SIGMAS, strict=True)):
        lower, median, upper = _quantiles(gaussian_result, index)
        median_error = 1.2533 * sigma / np.sqrt(effective)
        width_error = sigma / np.sqrt(2.0 * effective)

        assert abs(median - mean) < 3.0 * median_error
        assert abs(0.5 * (upper - lower) - sigma) < 3.0 * width_error + 0.01 * sigma


def test_result_carries_no_evidence(gaussian_result):
    assert gaussian_result.evidence is None
    assert gaussian_result.logz is None
    assert gaussian_result.diagnostics["evidence"] is None
    assert gaussian_result.provenance["backend"] == FAST_BACKEND_NAME


def test_samples_are_weighted_with_exact_log_likelihoods(gaussian_result):
    samples = gaussian_result.samples
    expected = -0.5 * np.sum(((samples - MEANS) / SIGMAS) ** 2, axis=1)

    assert np.allclose(gaussian_result.log_likelihood, expected)
    assert np.sum(gaussian_result.weights) == pytest.approx(1.0)


def test_correlation_is_recovered():
    covariance = np.array([[1.0, 0.4], [0.4, 0.25]])
    precision = np.linalg.inv(covariance)
    mean = np.array([1.0, -2.0])
    problem = BoxProblem(
        [-10.0, -10.0],
        [10.0, 10.0],
        lambda theta: -0.5 * (theta - mean) @ precision @ (theta - mean),
    )
    result = _fit(problem)
    weights = result.weights
    centre = weights @ result.samples
    centred = result.samples - centre
    estimate = (weights[:, None] * centred).T @ centred
    correlation = estimate[0, 1] / np.sqrt(estimate[0, 0] * estimate[1, 1])

    assert correlation == pytest.approx(0.8, abs=0.05)


def test_posterior_pressed_on_a_bound():
    # L = exp(a u) on a uniform prior: 1 - theta is exponential with rate a,
    # so the quantile p of theta is 1 + ln(p) / a.
    rate = 200.0
    problem = BoxProblem([0.0], [1.0], lambda theta: rate * theta[0])
    result = _fit(problem)
    effective = result.diagnostics["effective_sample_size"]

    assert result.diagnostics["pareto_k"] < 0.7

    for level, estimate in zip((0.16, 0.5, 0.84), _quantiles(result, 0), strict=True):
        exact = 1.0 + np.log(level) / rate
        error = np.sqrt(level * (1.0 - level) / effective) / (rate * level)

        assert abs(estimate - exact) < 3.0 * error


def test_two_modes_are_found_and_flagged():
    def log_likelihood(theta):
        return np.logaddexp(
            stats.norm.logpdf(theta[0], 2.0, 0.1),
            stats.norm.logpdf(theta[0], 8.0, 0.1),
        ) + stats.norm.logpdf(theta[1], 5.0, 0.3)

    problem = BoxProblem([0.0, 0.0], [10.0, 10.0], log_likelihood)
    result = _fit(problem)
    mass = np.sum(result.weights[result.samples[:, 0] < 5.0])

    assert len(result.diagnostics["modes"]) == 2
    assert FLAG_SEVERAL_MODES in result.diagnostics["flags"]
    assert mass == pytest.approx(0.5, abs=0.05)
    # Even when the masses come out right, several modes make the fit
    # unreliable: the next test shows how wrong they can be.
    assert result.diagnostics["reliable"] is False


# Four modes in four dimensions: x and y are each bimodal, through pixels
# x**2 and y**2, with a linear pixel that makes their two modes unequal, and
# z0 and z1 are Gaussian. Pixels [x**2, x, y**2, y, z0, z1], data and noise
# below, uniform priors; the normalization constant is omitted.
FOUR_MODE_LOWER = [-4.0, -3.5, -3.0, -30.0]
FOUR_MODE_UPPER = [4.0, 3.5, 3.0, 30.0]
FOUR_MODE_DATA = np.array([4.0, 2.0, 2.25, 1.5, 0.0, 0.0])
FOUR_MODE_NOISE = np.array([1.0, 2.5, 0.6, 2.0, 0.3, 3.0])


def _four_mode_log_likelihood(theta):
    x, y, z0, z1 = theta
    pixels = np.array([x * x, x, y * y, y, z0, z1])

    return -0.5 * np.sum(((pixels - FOUR_MODE_DATA) / FOUR_MODE_NOISE) ** 2)


def test_four_mode_target_is_flagged_unreliable():
    # With these ten seeds the starts find two or three of the four modes,
    # and in every run the mass of at least one quadrant is off by 0.055 to
    # 0.58 (exact masses by quadrature), while the Pareto shape and the
    # effective sample size stay within their limits. Every run must be
    # marked unreliable.
    problem = BoxProblem(FOUR_MODE_LOWER, FOUR_MODE_UPPER, _four_mode_log_likelihood)

    for seed in range(3000, 3010):
        diagnostics = _fit(problem, seed=seed).diagnostics

        assert len(diagnostics["modes"]) > 1, seed
        assert FLAG_SEVERAL_MODES in diagnostics["flags"], seed
        assert diagnostics["reliable"] is False, seed


def test_same_seed_gives_identical_results():
    problem = BoxProblem(
        [0.0, 0.0],
        [10.0, 10.0],
        lambda theta: -0.5 * np.sum(((theta - 4.0) / 0.3) ** 2),
    )
    first = _fit(problem, seed=7, draws=300, scan=64)
    second = _fit(problem, seed=7, draws=300, scan=64)
    other = _fit(problem, seed=8, draws=300, scan=64)

    assert np.array_equal(first.samples, second.samples)
    assert np.array_equal(first.weights, second.weights)
    assert not np.array_equal(first.samples, other.samples)


def test_parallel_evaluation_matches_serial():
    problem = BoxProblem(
        [0.0, 0.0],
        [10.0, 10.0],
        lambda theta: -0.5 * np.sum(((theta - 4.0) / 0.3) ** 2),
    )
    serial = _fit(problem, seed=3, draws=200, scan=32)
    parallel = make_sampler(
        {
            "name": FAST_BACKEND_NAME,
            "seed": 3,
            "draws": 200,
            "scan": 32,
            "execution": {"processes": 2, "progress": False},
        },
        problem=problem,
    ).fit()

    assert np.array_equal(serial.samples, parallel.samples)
    assert np.array_equal(serial.weights, parallel.weights)


def test_registry_creates_the_fast_backend():
    problem = BoxProblem([0.0], [1.0], lambda theta: 0.0)
    backend = make_sampler({"name": "fast"}, problem=problem)

    assert isinstance(backend, FastBackend)
    assert backend.name == FAST_BACKEND_NAME


@pytest.mark.parametrize(
    ("config", "error"),
    [
        ({"walkers": 10}, ValueError),
        ({"draws": 50}, ValueError),
        ({"starts": 0}, ValueError),
        ({"starts": 8, "scan": 4}, ValueError),
        ({"seed": -1}, ValueError),
        ({"seed": 1.5}, TypeError),
        ({"draws": True}, TypeError),
        ({"name": "nautilus"}, ValueError),
        ({"execution": {"threads": 2}}, ValueError),
        ({"execution": "auto"}, TypeError),
    ],
)
def test_invalid_settings_are_refused(config, error):
    with pytest.raises(error):
        resolve_fast_settings(config)


def test_settings_defaults_are_recorded():
    settings = resolve_fast_settings({})

    assert settings.as_dict() == {
        "name": "fast",
        "seed": 42,
        "starts": 4,
        "scan": 256,
        "draws": 2000,
        "max_iterations": 500,
        "execution": {"processes": "auto", "progress": True},
    }


def test_nan_log_likelihood_stops_the_fit():
    problem = BoxProblem([0.0], [1.0], lambda theta: np.nan)

    with pytest.raises(RuntimeError, match="NaN"):
        _fit(problem, draws=100, scan=8)


def _wrap_problem():
    # A position angle at the cut of its period: a pi-periodic likelihood
    # with its mode at 0.495 pi on a uniform prior over one period, and seven
    # nuisance parameters with off-centre modes.
    mode = 0.495 * np.pi
    concentration = 1.0 / (2.0 * (np.pi * 0.006) ** 2)

    def periodic(angle):
        return 0.5 * concentration * (np.cos(2.0 * (angle - mode)) - 1.0)

    centres = np.array([0.31, 0.72, 0.45, 0.63, 0.27, 0.58, 0.81])
    problem = BoxProblem(
        [-0.5 * np.pi] + [0.0] * 7,
        [0.5 * np.pi] + [1.0] * 7,
        lambda theta: periodic(theta[0])
        - 0.5 * np.sum((theta[1:] - centres) ** 2) / 0.005**2,
        cyclic=(0,),
    )
    grid = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 2_000_001)
    density = np.exp(periodic(grid))
    beyond = float(np.sum(density[grid < 0.0]) / np.sum(density))

    return problem, beyond


def test_position_angle_at_the_cut_of_its_period():
    problem, beyond = _wrap_problem()

    assert beyond == pytest.approx(0.202, abs=0.001)

    for seed in range(100, 130):
        result = _fit(problem, seed=seed)
        mass = float(np.sum(result.weights[result.samples[:, 0] < 0.0]))

        assert abs(mass - beyond) <= 0.03, seed
        assert result.diagnostics["reliable"] is True, seed
        assert np.all(result.samples[:, 0] >= -0.5 * np.pi)
        assert np.all(result.samples[:, 0] <= 0.5 * np.pi)
        assert result.diagnostics["cyclic_parameters"] == ["p0"]


def test_cyclic_coordinates_follow_the_declared_period():
    from baymorph.fitting.parameters import Parameter, ParameterSet
    from baymorph.fitting.priors import UniformPrior

    full = Parameter(
        "theta", prior=UniformPrior(-0.5 * np.pi, 0.5 * np.pi), period=np.pi
    )
    narrow = Parameter("theta", prior=UniformPrior(-1.0, 1.0), period=np.pi)
    undeclared = Parameter("theta", prior=UniformPrior(-0.5 * np.pi, 0.5 * np.pi))

    assert full.is_cyclic
    assert not narrow.is_cyclic
    assert not undeclared.is_cyclic
    assert ParameterSet([Parameter("x", 0.0, 1.0), full]).cyclic_coordinates == (1,)

    with pytest.raises(ValueError, match="period"):
        Parameter("theta", 0.0, 1.0, period=-1.0)


def test_models_declare_the_period_of_their_position_angles():
    from baymorph.models.geometry import parameter_periods

    assert parameter_periods("single_sersic") == {"theta": pytest.approx(np.pi)}
    assert set(parameter_periods("bulge_disk")) == {"theta_bulge", "theta_disk"}
    assert parameter_periods("boxy_sersic") == {}
