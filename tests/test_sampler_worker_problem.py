"""The inference problem reaches each worker once, not once per task chunk."""

from __future__ import annotations

import os

import dill
import numpy as np
import pytest

from baymorph.fitting import Parameter, ParameterSet
from baymorph.fitting.execution import sampler_pool
from baymorph.fitting.image_problem import ImageInferenceProblem
from baymorph.fitting.nautilus import (
    NautilusFitter,
    _initialize_nautilus_worker,
    _worker_loglikelihood,
)
from baymorph.likelihood import GaussianImageLikelihood
from baymorph.pipeline.real_fit import _make_model_factory
from baymorph.simulation.psf import GaussianPSF
import baymorph.fitting.nautilus as nautilus_backend


SHAPE = (21, 21)


def _problem() -> ImageInferenceProblem:
    """Build one small production-shaped inference problem."""
    parameters = ParameterSet(
        [
            Parameter("flux", 100.0, 1000.0),
            Parameter("re", 1.0, 10.0),
            Parameter("n", 0.5, 5.0),
            Parameter("q", 0.4, 1.0),
            Parameter("theta", -1.0, 1.0),
            Parameter("x0", 8.0, 12.0),
            Parameter("y0", 8.0, 12.0),
            Parameter("background_level", -5.0, 5.0),
            Parameter("background_gradient_x", -0.1, 0.1),
            Parameter("background_gradient_y", -0.1, 0.1),
        ]
    )

    return ImageInferenceProblem(
        model_factory=_make_model_factory(
            "single_sersic",
            GaussianPSF(fwhm=2.5),
            background_cfg={"model": "plane"},
        ),
        parameters=parameters,
        likelihood=GaussianImageLikelihood(
            data=np.zeros(SHAPE, dtype=float),
            noise=np.ones(SHAPE, dtype=float),
        ),
    )


def _points(problem: ImageInferenceProblem) -> list[np.ndarray]:
    """Return physical parameter vectors drawn through the prior."""
    rng = np.random.default_rng(20260920)

    cubes = [np.full(problem.ndim, 0.5, dtype=float)]
    cubes.extend(rng.random(problem.ndim) for _ in range(4))

    return [problem.prior_transform(cube) for cube in cubes]


def _probe(theta):
    """Report, from inside a worker, that the problem is installed."""
    return (
        os.getpid(),
        nautilus_backend._NAUTILUS_WORKER_PROBLEM is not None,
        _worker_loglikelihood(theta),
    )


@pytest.fixture(autouse=True)
def _restore_worker_problem():
    """Keep the module-level worker problem out of other tests."""
    previous = nautilus_backend._NAUTILUS_WORKER_PROBLEM

    yield

    nautilus_backend._NAUTILUS_WORKER_PROBLEM = previous


def test_worker_callable_fails_closed_without_initialization() -> None:
    """An uninitialized worker must raise, never evaluate something else."""
    nautilus_backend._NAUTILUS_WORKER_PROBLEM = None

    with pytest.raises(RuntimeError, match="not initialized"):
        _worker_loglikelihood(np.zeros(10, dtype=float))


def test_worker_callable_matches_the_bound_method() -> None:
    """The two paths must evaluate the same mathematical function."""
    problem = _problem()
    fitter = NautilusFitter(problem=problem)

    _initialize_nautilus_worker(problem)

    for theta in _points(problem):
        assert _worker_loglikelihood(theta) == fitter.loglikelihood(theta)


def test_worker_callable_does_not_carry_the_problem() -> None:
    """What Nautilus ships per task chunk must not contain the problem.

    ``Pool.map`` serializes the mapped callable once per chunk. A bound method
    drags the whole problem, with its data, noise, mask and PSF, into every
    chunk for the entire run; a module-level function does not.
    """
    problem = _problem()
    fitter = NautilusFitter(problem=problem)

    bound_payload = len(dill.dumps(fitter.loglikelihood))
    worker_payload = len(dill.dumps(_worker_loglikelihood))

    assert worker_payload < 4096
    assert worker_payload < bound_payload / 100


def test_pool_workers_receive_the_problem_once() -> None:
    """Every worker evaluates with an installed problem, matching serial."""
    problem = _problem()
    fitter = NautilusFitter(problem=problem)
    points = _points(problem)

    with sampler_pool(
        2,
        initializer=_initialize_nautilus_worker,
        initargs=(problem,),
    ) as pool:
        assert pool is not None

        results = list(pool.map(_probe, points))

    for theta, (_, installed, value) in zip(points, results, strict=True):
        assert installed
        assert value == fitter.loglikelihood(theta)


def test_serial_execution_keeps_the_bound_method() -> None:
    """With one process there is no pool and no module-level installation."""
    problem = _problem()
    fitter = NautilusFitter(problem=problem)

    nautilus_backend._NAUTILUS_WORKER_PROBLEM = None

    with sampler_pool(
        1,
        initializer=_initialize_nautilus_worker,
        initargs=(problem,),
    ) as pool:
        assert pool is None

    assert nautilus_backend._NAUTILUS_WORKER_PROBLEM is None

    for theta in _points(problem):
        assert np.isfinite(fitter.loglikelihood(theta))
