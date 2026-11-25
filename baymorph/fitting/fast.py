"""Fast approximate posterior inference, without evidence (AD-057).

The fast backend approximates the posterior of one :class:`InferenceProblem`
in minutes instead of the hours a nested sampler may need, and computes no
Bayesian evidence.

Every BayMorph prior is defined by its unit-cube transform ``theta = T(u)``,
under which ``u`` is uniform. The posterior of ``u`` is therefore
``L(T(u))`` on the cube, whatever the prior. The backend works in the
unconstrained coordinates ``z = logit(u)``, where

``log p(z) = log L(T(sigmoid(z))) + sum_i log[sigmoid(z_i) (1 - sigmoid(z_i))]``

up to a constant. The prior is preserved exactly and a posterior pressed on a
bound keeps a finite mode in ``z``. The steps are:

1. a scrambled Sobol scan of the cube chooses the starting points;
2. BFGS with central-difference gradients finds the local maxima of
   ``log p(z)``, in coordinates scaled by a probe of the curvature;
3. maxima closer than three standard deviations are merged;
4. the Hessian at each distinct mode gives a Student-t component of an
   importance proposal, beside a defensive component equal to the prior;
5. draws from the proposal are weighted by the exact posterior and the
   weights are Pareto-smoothed; one moment-matched adaptation follows when
   the Pareto shape exceeds 0.5.

The products are weighted draws with their exact log-likelihoods, and
diagnostics whose flags state when the approximation is unreliable.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit, gammaln, logit, logsumexp
from scipy.stats import qmc

from baymorph.fitting.backend import InferenceBackend
from baymorph.fitting.execution import (
    SamplerExecution,
    resolve_sampler_execution,
    sampler_pool,
)
from baymorph.fitting.problem import InferenceProblem
from baymorph.results import InferenceResult, PosteriorSamples
from baymorph.stats.importance import pareto_smoothed_log_weights

FloatArray = NDArray[np.float64]

FAST_BACKEND_NAME = "fast"
FAST_METHOD = "logit_cube_laplace_psis_v2"

FAST_DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "starts": 4,
    "scan": 256,
    "draws": 2000,
    "max_iterations": 500,
    "execution": {"processes": "auto", "progress": True},
}

# Fixed numerical choices of the method (AD-057). They are not
# configuration: changing one changes the method.
STUDENT_DEGREES_OF_FREEDOM = 5.0
DEFENSIVE_FRACTION = 0.02
PARETO_K_ADAPT = 0.5
PARETO_K_UNRELIABLE = 0.7
MINIMUM_EFFECTIVE_SAMPLE_SIZE = 400.0
MINIMUM_DRAWS = 100
MODE_MAHALANOBIS_DISTANCE = 3.0
MODE_LOG_MASS_WINDOW = 20.0
GRADIENT_STEP = 1e-2
GRADIENT_TOLERANCE = 1e-2
CONVERGED_GRADIENT = 0.1
HESSIAN_STEP = 0.1
EIGENVALUE_FLOOR = 1e-8
RESCALE_FACTOR = 3.0
PROBE_ROUNDS = 8
PROBE_LARGEST_STEP = 3.0
PROBE_SMALLEST_STEP = 1e-9

_FAST_KEYS = {"name", "seed", "starts", "scan", "draws", "max_iterations", "execution"}

FLAG_PARETO = "pareto_k_above_0.7"
FLAG_EFFECTIVE_SAMPLE_SIZE = "effective_sample_size_below_400"
FLAG_SEVERAL_MODES = "several_modes"
FLAG_HESSIAN = "hessian_not_positive_definite"
FLAG_OPTIMIZER = "optimizer_not_converged"

# A posterior with several modes is unreliable too: the starts may miss a
# mode, and the masses the mixture gives to the modes it finds can be far
# from the true ones while the Pareto shape and the effective sample size
# stay within their limits.
UNRELIABLE_FLAGS = (FLAG_PARETO, FLAG_EFFECTIVE_SAMPLE_SIZE, FLAG_SEVERAL_MODES)


@dataclass(frozen=True)
class FastSettings:
    """Validated settings of the fast backend."""

    seed: int
    starts: int
    scan: int
    draws: int
    max_iterations: int
    execution: SamplerExecution
    requested_execution: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return the settings as a configuration mapping."""
        return {
            "name": FAST_BACKEND_NAME,
            "seed": self.seed,
            "starts": self.starts,
            "scan": self.scan,
            "draws": self.draws,
            "max_iterations": self.max_iterations,
            "execution": dict(self.requested_execution),
        }


def _positive_integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"fit.fast.{name} must be an integer.")

    if int(value) < minimum:
        raise ValueError(f"fit.fast.{name} must be at least {minimum}.")

    return int(value)


def resolve_fast_settings(config: Mapping[str, Any] | None = None) -> FastSettings:
    """Validate the fast-backend configuration and fill its defaults.

    Parameters
    ----------
    config : mapping, optional
        Entries of ``fit.fast``. ``name``, when present, must be ``"fast"``.

    Returns
    -------
    FastSettings
        The validated settings.

    Raises
    ------
    ValueError
        If an entry is unknown or out of its domain.
    TypeError
        If an entry has the wrong type.
    """
    raw = {} if config is None else dict(config)
    unknown = sorted(set(raw) - _FAST_KEYS)

    if unknown:
        raise ValueError(
            f"Unknown fit.fast option(s): {', '.join(unknown)}. "
            f"Available options: {', '.join(sorted(_FAST_KEYS - {'name'}))}."
        )

    name = str(raw.get("name", FAST_BACKEND_NAME)).strip().lower()

    if name != FAST_BACKEND_NAME:
        raise ValueError(f"The fast backend cannot be named {name!r}.")

    merged = {**FAST_DEFAULTS, **{k: v for k, v in raw.items() if k != "name"}}
    execution = merged.get("execution") or {}

    if not isinstance(execution, Mapping):
        raise TypeError("fit.fast.execution must be a mapping.")

    execution = {**FAST_DEFAULTS["execution"], **dict(execution)}

    seed = _positive_integer(merged["seed"], "seed", minimum=0)
    starts = _positive_integer(merged["starts"], "starts")
    scan = _positive_integer(merged["scan"], "scan")
    draws = _positive_integer(merged["draws"], "draws", minimum=MINIMUM_DRAWS)
    max_iterations = _positive_integer(merged["max_iterations"], "max_iterations")

    if scan < starts:
        raise ValueError("fit.fast.scan must be at least fit.fast.starts.")

    return FastSettings(
        seed=seed,
        starts=starts,
        scan=scan,
        draws=draws,
        max_iterations=max_iterations,
        execution=resolve_sampler_execution(execution),
        requested_execution=execution,
    )


# ---------------------------------------------------------------------------
# Evaluation of the posterior in logit-cube coordinates
# ---------------------------------------------------------------------------

_FAST_WORKER_PROBLEM: InferenceProblem | None = None


def _initialize_fast_worker(problem: InferenceProblem) -> None:
    """Install one inference problem in a worker process."""
    global _FAST_WORKER_PROBLEM

    _FAST_WORKER_PROBLEM = problem


def _cube_evaluation(
    problem: InferenceProblem,
    cube: FloatArray,
) -> tuple[FloatArray, float]:
    """Return the physical parameters and the log-likelihood at one cube point."""
    theta = np.asarray(problem.prior_transform(np.asarray(cube, dtype=float)))
    value = float(problem.log_likelihood(theta))

    if np.isnan(value) or np.isposinf(value):
        raise RuntimeError(
            "The log-likelihood returned NaN or +inf at parameters "
            f"{theta.tolist()}; the fast backend cannot continue."
        )

    return theta, value


def _worker_cube_evaluation(cube: FloatArray) -> tuple[FloatArray, float]:
    if _FAST_WORKER_PROBLEM is None:
        raise RuntimeError("The fast worker inference problem was not initialized.")

    return _cube_evaluation(_FAST_WORKER_PROBLEM, cube)


def log_jacobian(z: FloatArray) -> FloatArray:
    """Return ``sum_i log[sigmoid(z_i) (1 - sigmoid(z_i))]`` for each row."""
    z = np.atleast_2d(np.asarray(z, dtype=float))

    return -np.sum(np.logaddexp(0.0, -z) + np.logaddexp(0.0, z), axis=1)


class _Evaluator:
    """Evaluate ``log p(z)`` in batches, serially or on a process pool."""

    def __init__(self, problem: InferenceProblem, pool: Any, processes: int) -> None:
        self.problem = problem
        self.pool = pool
        self.processes = processes
        self.count = 0
        self.best_log_likelihood = -np.inf
        self.best_theta: FloatArray | None = None
        self.cyclic = np.zeros(problem.ndim, dtype=bool)
        self.cyclic[list(problem.cyclic_coordinates)] = True
        # Cyclic shift of the cube, in cube units, of the cyclic coordinates.
        self.shift = np.zeros(problem.ndim)

    def cube(self, z: FloatArray) -> FloatArray:
        """Return the cube coordinates of ``z``, shifted on cyclic coordinates."""
        cube = expit(np.atleast_2d(np.asarray(z, dtype=float)))

        if np.any(self.cyclic):
            cube[:, self.cyclic] = np.mod(
                cube[:, self.cyclic] + self.shift[self.cyclic], 1.0
            )

        return cube

    def recentre(self, z: FloatArray) -> FloatArray:
        """Shift the cyclic coordinates so that ``z`` sits at the cube centre.

        The prior of a cyclic coordinate is uniform over one period of a
        likelihood periodic in it, so the shift leaves the posterior
        unchanged; it moves the cut of the period away from the mode. Returns
        ``z`` in the new coordinates.
        """
        z = np.asarray(z, dtype=float).copy()

        if not np.any(self.cyclic):
            return z

        actual = self.cube(z[None, :])[0]
        self.shift[self.cyclic] = actual[self.cyclic] - 0.5
        z[self.cyclic] = 0.0

        return z

    def to_current(self, cube: FloatArray) -> FloatArray:
        """Return the ``z`` of unshifted cube points in the current coordinates."""
        cube = np.atleast_2d(np.asarray(cube, dtype=float)).copy()

        if np.any(self.cyclic):
            cube[:, self.cyclic] = np.mod(
                cube[:, self.cyclic] - self.shift[self.cyclic], 1.0
            )

        return logit(np.clip(cube, 1e-12, 1.0 - 1e-12))

    def __call__(self, z: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return ``log p(z)``, ``log L`` and the physical parameters."""
        z = np.atleast_2d(np.asarray(z, dtype=float))
        count, ndim = z.shape
        raw = expit(z)
        inside = np.all((raw > 0.0) & (raw < 1.0) & np.isfinite(z), axis=1)
        cube = self.cube(z)
        log_likelihood = np.full(count, -np.inf)
        theta = np.full((count, ndim), np.nan)
        indices = np.flatnonzero(inside)

        if indices.size:
            tasks = [cube[index] for index in indices]

            if self.pool is None:
                results = [_cube_evaluation(self.problem, task) for task in tasks]
            else:
                chunk = max(1, len(tasks) // (4 * self.processes))
                results = self.pool.map(_worker_cube_evaluation, tasks, chunksize=chunk)

            self.count += len(tasks)

            for index, (point, value) in zip(indices, results, strict=True):
                theta[index] = point
                log_likelihood[index] = value

                if value > self.best_log_likelihood:
                    self.best_log_likelihood = value
                    self.best_theta = np.asarray(point, dtype=float).copy()

        log_posterior = np.where(
            np.isfinite(log_likelihood),
            log_likelihood + log_jacobian(z),
            -np.inf,
        )

        return log_posterior, log_likelihood, theta


# ---------------------------------------------------------------------------
# Search for the modes
# ---------------------------------------------------------------------------


def _curvature_scales(evaluate: _Evaluator, centre: FloatArray) -> FloatArray:
    """Return a diagonal estimate of the posterior standard deviations in ``z``.

    Each coordinate's step is adjusted until the second difference of
    ``log p`` is of order one, so that the step is of the order of one local
    standard deviation. A coordinate that stays flat or convex up to the
    largest step gets that step.
    """
    ndim = centre.size
    centre_value = evaluate(centre[None, :])[0][0]
    steps = np.full(ndim, 0.1)
    scales = np.full(ndim, np.nan)
    identity = np.eye(ndim)

    for _ in range(PROBE_ROUNDS):
        pending = np.flatnonzero(~np.isfinite(scales))

        if pending.size == 0:
            break

        offsets = identity[pending] * steps[pending, None]
        points = np.concatenate([centre + offsets, centre - offsets])
        values = evaluate(points)[0]
        upper = values[: pending.size]
        lower = values[pending.size :]
        second = upper - 2.0 * centre_value + lower

        for position, index in enumerate(pending):
            curvature = -second[position]

            if not np.isfinite(curvature):
                # A step that leaves the support is too long.
                if steps[index] <= PROBE_SMALLEST_STEP:
                    scales[index] = PROBE_SMALLEST_STEP
                else:
                    steps[index] = max(PROBE_SMALLEST_STEP, 0.1 * steps[index])
                continue

            if curvature <= 0.0:
                # Flat or convex: look further out.
                if steps[index] >= PROBE_LARGEST_STEP:
                    scales[index] = PROBE_LARGEST_STEP
                else:
                    steps[index] = min(PROBE_LARGEST_STEP, 10.0 * steps[index])
                continue

            estimate = steps[index] / np.sqrt(curvature)

            if 0.25 <= curvature <= 4.0:
                scales[index] = estimate
            else:
                steps[index] = float(
                    np.clip(estimate, PROBE_SMALLEST_STEP, PROBE_LARGEST_STEP)
                )

                if steps[index] in (PROBE_SMALLEST_STEP, PROBE_LARGEST_STEP):
                    scales[index] = steps[index]

    unresolved = ~np.isfinite(scales)
    scales[unresolved] = steps[unresolved]

    return scales


@dataclass
class _Search:
    """One local maximization of ``log p(z)``."""

    z: FloatArray
    log_posterior: float
    scales: FloatArray
    converged: bool
    iterations: int
    message: str
    inverse_hessian: FloatArray | None = None
    merged: bool = False


class _EnteredKnownMode(Exception):
    """A local search entered the basin of a mode already found."""


def _maximize(
    evaluate: _Evaluator,
    start: FloatArray,
    scales: FloatArray,
    *,
    max_iterations: int,
    known: list[_Mode] | None = None,
) -> _Search:
    """Maximize ``log p(z)`` by BFGS in coordinates ``z = start + scales * y``.

    A search that comes within ``MODE_MAHALANOBIS_DISTANCE`` of a mode in
    ``known`` stops there: it would converge to that mode, which is merged
    with it anyway.
    """
    ndim = start.size
    cache: dict[bytes, float] = {}
    identity = np.eye(ndim)
    precisions = [np.linalg.inv(mode.covariance) for mode in (known or [])]
    centres = [mode.z for mode in (known or [])]

    def value_at(y: FloatArray) -> float:
        key = np.asarray(y, dtype=float).tobytes()

        if key not in cache:
            cache[key] = float(evaluate((start + scales * y)[None, :])[0][0])

        return cache[key]

    def negative(y: FloatArray) -> float:
        value = value_at(y)

        return -value if np.isfinite(value) else np.inf

    def gradient(y: FloatArray) -> FloatArray:
        centre = value_at(y)
        offsets = GRADIENT_STEP * identity
        points = np.concatenate([y + offsets, y - offsets])
        values = evaluate(start + scales * points)[0]
        upper, lower = values[:ndim], values[ndim:]
        result = np.zeros(ndim)

        for index in range(ndim):
            if np.isfinite(upper[index]) and np.isfinite(lower[index]):
                result[index] = (upper[index] - lower[index]) / (2.0 * GRADIENT_STEP)
            elif np.isfinite(upper[index]) and np.isfinite(centre):
                result[index] = (upper[index] - centre) / GRADIENT_STEP
            elif np.isfinite(lower[index]) and np.isfinite(centre):
                result[index] = (centre - lower[index]) / GRADIENT_STEP

        return -result

    def callback(intermediate_result) -> None:
        z = start + scales * intermediate_result.x

        for centre, precision in zip(centres, precisions, strict=True):
            difference = z - centre

            if difference @ precision @ difference < MODE_MAHALANOBIS_DISTANCE**2:
                raise _EnteredKnownMode()

    try:
        outcome = minimize(
            negative,
            np.zeros(ndim),
            jac=gradient,
            method="BFGS",
            callback=callback if centres else None,
            options={"gtol": GRADIENT_TOLERANCE, "maxiter": max_iterations},
        )
    except _EnteredKnownMode:
        return _Search(
            z=start,
            log_posterior=-np.inf,
            scales=scales,
            converged=True,
            iterations=0,
            message="entered the basin of a mode already found",
            merged=True,
        )

    final_gradient = gradient(outcome.x)
    converged = bool(outcome.success) or bool(
        np.max(np.abs(final_gradient)) <= CONVERGED_GRADIENT
    )
    inverse_hessian = np.asarray(getattr(outcome, "hess_inv", np.eye(ndim)))

    return _Search(
        z=start + scales * outcome.x,
        log_posterior=-float(outcome.fun),
        scales=scales,
        converged=converged,
        iterations=int(outcome.nit),
        message=str(outcome.message),
        inverse_hessian=inverse_hessian,
    )


def _hessian(
    evaluate: _Evaluator,
    centre: FloatArray,
    steps: FloatArray,
) -> FloatArray:
    """Return the Hessian of ``-log p(z)`` by central differences."""
    ndim = centre.size
    points = [centre]
    offsets = np.diag(steps)

    for index in range(ndim):
        points.append(centre + offsets[index])
        points.append(centre - offsets[index])

    pairs = [(i, j) for i in range(ndim) for j in range(i + 1, ndim)]

    for i, j in pairs:
        for sign_i in (1.0, -1.0):
            for sign_j in (1.0, -1.0):
                points.append(centre + sign_i * offsets[i] + sign_j * offsets[j])

    values = evaluate(np.asarray(points))[0]

    if not np.all(np.isfinite(values)):
        raise _HessianOutsideSupport()

    centre_value = values[0]
    hessian = np.empty((ndim, ndim))

    for index in range(ndim):
        upper = values[1 + 2 * index]
        lower = values[2 + 2 * index]
        hessian[index, index] = (
            -(upper - 2.0 * centre_value + lower) / steps[index] ** 2
        )

    cursor = 1 + 2 * ndim

    for i, j in pairs:
        pp, pm, mp, mm = values[cursor : cursor + 4]
        cursor += 4
        hessian[i, j] = hessian[j, i] = -(pp - pm - mp + mm) / (
            4.0 * steps[i] * steps[j]
        )

    return hessian


class _HessianOutsideSupport(RuntimeError):
    """A finite-difference point of the Hessian left the posterior support."""


@dataclass
class _Mode:
    """A distinct mode with its Laplace geometry."""

    z: FloatArray
    log_posterior: float
    covariance: FloatArray
    positive_definite: bool
    converged: bool
    iterations: int
    message: str
    hessian_step_change: float = 0.0

    @property
    def log_mass(self) -> float:
        sign, log_determinant = np.linalg.slogdet(2.0 * np.pi * self.covariance)

        return float(self.log_posterior + 0.5 * log_determinant)


def _laplace_covariance(hessian: FloatArray) -> tuple[FloatArray, bool]:
    """Return the inverse of a Hessian, with its eigenvalues floored."""
    symmetric = 0.5 * (hessian + hessian.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    largest = float(np.max(np.abs(eigenvalues)))
    positive_definite = bool(np.all(eigenvalues > 0.0))
    floor = EIGENVALUE_FLOOR * (largest if largest > 0.0 else 1.0)
    eigenvalues = np.maximum(eigenvalues, floor)
    covariance = (eigenvectors / eigenvalues) @ eigenvectors.T

    return 0.5 * (covariance + covariance.T), positive_definite


def _geometry(evaluate: _Evaluator, search: _Search) -> tuple[FloatArray, bool, float]:
    """Return the Laplace covariance at a mode, from two Hessian passes.

    The third value is the largest relative change of the standard deviations
    between the two passes, whose steps differ: a check of the step size.
    """
    if search.inverse_hessian is not None and search.inverse_hessian.shape == (
        search.z.size,
        search.z.size,
    ):
        rough = np.sqrt(np.clip(np.diag(search.inverse_hessian), 1e-12, 1e12))
    else:
        rough = np.ones(search.z.size)

    steps = HESSIAN_STEP * search.scales * rough
    changes: list[FloatArray] = []

    for _ in range(2):
        try:
            hessian = _hessian(evaluate, search.z, steps)
        except _HessianOutsideSupport:
            steps = 0.1 * steps
            hessian = _hessian(evaluate, search.z, steps)

        covariance, positive_definite = _laplace_covariance(hessian)
        deviations = np.sqrt(np.diag(covariance))
        changes.append(deviations)
        steps = HESSIAN_STEP * deviations

    change = float(np.max(np.abs(changes[1] / changes[0] - 1.0)))

    return covariance, positive_definite, change


def _distinct(
    candidate: FloatArray,
    modes: list[_Mode],
) -> bool:
    for mode in modes:
        difference = candidate - mode.z
        distance = float(difference @ np.linalg.solve(mode.covariance, difference))

        if distance < MODE_MAHALANOBIS_DISTANCE**2:
            return False

    return True


# ---------------------------------------------------------------------------
# Importance proposal
# ---------------------------------------------------------------------------


class _Proposal:
    """Mixture of Student-t components and the prior, in ``z`` coordinates."""

    def __init__(
        self,
        locations: FloatArray,
        covariances: FloatArray,
        component_weights: FloatArray,
    ) -> None:
        self.locations = np.atleast_2d(np.asarray(locations, dtype=float))
        self.scales = np.asarray(covariances, dtype=float).reshape(
            (self.locations.shape[0],) + (self.locations.shape[1],) * 2
        )
        weights = np.asarray(component_weights, dtype=float)
        self.component_weights = weights / np.sum(weights)
        self.cholesky = np.stack([np.linalg.cholesky(scale) for scale in self.scales])
        self.log_determinants = 2.0 * np.sum(
            np.log(np.diagonal(self.cholesky, axis1=1, axis2=2)), axis=1
        )

    @property
    def ndim(self) -> int:
        return int(self.locations.shape[1])

    def sample(self, rng: np.random.Generator, count: int) -> FloatArray:
        """Draw ``count`` points."""
        nu = STUDENT_DEGREES_OF_FREEDOM
        defensive = rng.random(count) < DEFENSIVE_FRACTION
        components = rng.choice(
            self.component_weights.size, size=count, p=self.component_weights
        )
        normal = rng.standard_normal((count, self.ndim))
        chi = rng.chisquare(nu, size=count) / nu
        uniform = rng.random((count, self.ndim))
        tiny = np.finfo(float).tiny
        prior_draws = logit(np.clip(uniform, tiny, 1.0 - np.finfo(float).epsneg))
        student = (
            self.locations[components]
            + np.einsum("kij,kj->ki", self.cholesky[components], normal)
            / np.sqrt(chi)[:, None]
        )

        return np.where(defensive[:, None], prior_draws, student)

    def log_density(self, z: FloatArray) -> FloatArray:
        """Return the log-density of the proposal."""
        z = np.atleast_2d(np.asarray(z, dtype=float))

        return np.logaddexp(
            np.log1p(-DEFENSIVE_FRACTION) + logsumexp(self.component_terms(z), axis=0),
            np.log(DEFENSIVE_FRACTION) + log_jacobian(z),
        )

    def component_terms(self, z: FloatArray) -> FloatArray:
        """Return ``log w_k + log t_k(z)`` for every Student-t component."""
        z = np.atleast_2d(np.asarray(z, dtype=float))
        nu = STUDENT_DEGREES_OF_FREEDOM
        ndim = self.ndim
        constant = (
            gammaln(0.5 * (nu + ndim))
            - gammaln(0.5 * nu)
            - 0.5 * ndim * np.log(nu * np.pi)
        )
        terms = []

        for location, cholesky, log_determinant, weight in zip(
            self.locations,
            self.cholesky,
            self.log_determinants,
            self.component_weights,
            strict=True,
        ):
            whitened = np.linalg.solve(cholesky, (z - location).T).T
            distance = np.sum(whitened**2, axis=1)
            terms.append(
                np.log(weight)
                + constant
                - 0.5 * log_determinant
                - 0.5 * (nu + ndim) * np.log1p(distance / nu)
            )

        return np.stack(terms)


def _adapted_proposal(batch: "_Batch", proposal: _Proposal) -> _Proposal | None:
    """Return the proposal moment-matched to the weighted draws, or None.

    Every Student-t component takes the weighted mean and covariance of the
    draws in proportion to its responsibility for them. A component with
    fewer than five effective draws per dimension keeps its location, scale
    and weight, and a covariance that is not positive definite cancels the
    adaptation.
    """
    weights = np.exp(batch.log_weights)
    kept = weights > 0.0
    z = batch.z[kept]
    weights = weights[kept]
    terms = proposal.component_terms(z)
    responsibilities = np.exp(terms - logsumexp(terms, axis=0))
    ndim = proposal.ndim
    locations, scales, masses = [], [], []

    for index in range(responsibilities.shape[0]):
        component = weights * responsibilities[index]
        total = float(np.sum(component))
        effective = total**2 / float(np.sum(component**2)) if total > 0 else 0.0

        if effective < 5.0 * ndim:
            locations.append(proposal.locations[index])
            scales.append(proposal.scales[index])
            masses.append(max(total, 1e-12))
            continue

        mean = component @ z / total
        centred = z - mean
        covariance = (component[:, None] * centred).T @ centred / total
        covariance = 0.5 * (covariance + covariance.T)

        if not np.all(np.linalg.eigvalsh(covariance) > 0.0):
            return None

        locations.append(mean)
        scales.append(covariance)
        masses.append(total)

    return _Proposal(np.stack(locations), np.stack(scales), np.asarray(masses))


@dataclass
class _Batch:
    """One batch of weighted importance draws."""

    z: FloatArray
    theta: FloatArray
    log_likelihood: FloatArray
    log_weights: FloatArray
    pareto_k: float
    effective_sample_size: float
    raw_log_weights: FloatArray


def _weighted_batch(
    evaluate: _Evaluator,
    proposal: _Proposal,
    rng: np.random.Generator,
    count: int,
) -> _Batch:
    z = proposal.sample(rng, count)
    log_posterior, log_likelihood, theta = evaluate(z)
    log_weights = log_posterior - proposal.log_density(z)
    smoothed = pareto_smoothed_log_weights(log_weights)

    return _Batch(
        z=z,
        theta=theta,
        log_likelihood=log_likelihood,
        log_weights=smoothed.log_weights,
        pareto_k=smoothed.pareto_k,
        effective_sample_size=smoothed.effective_sample_size,
        raw_log_weights=log_weights,
    )


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------


def _named(problem: InferenceProblem, theta: FloatArray | None) -> dict[str, float]:
    if theta is None:
        return {}

    return {
        name: float(value)
        for name, value in zip(problem.parameter_names, theta, strict=True)
    }


def run_fast_inference(
    problem: InferenceProblem,
    settings: FastSettings,
    *,
    progress_label: str | None = None,
) -> InferenceResult:
    """Approximate the posterior of one problem with the fast method.

    Parameters
    ----------
    problem : InferenceProblem
        Complete backend-neutral problem.
    settings : FastSettings
        Validated settings.
    progress_label : str, optional
        Label printed with the progress messages.

    Returns
    -------
    InferenceResult
        Weighted draws without evidence, with diagnostics and flags.
    """
    if not isinstance(problem, InferenceProblem):
        raise TypeError("problem must be an InferenceProblem instance.")

    ndim = problem.ndim
    rng = np.random.default_rng(settings.seed)
    talk = progress_label is not None and settings.execution.progress
    started = time.perf_counter()
    counts: dict[str, int] = {}

    if talk:
        print()
        print("────────────────────────────────────────────────────")
        print(f"Running fast fit: {progress_label}")
        print(f"ndim={ndim}")
        print("────────────────────────────────────────────────────")

    with sampler_pool(
        settings.execution.processes,
        initializer=_initialize_fast_worker,
        initargs=(problem,),
    ) as pool:
        evaluate = _Evaluator(problem, pool, settings.execution.processes)

        # 1. Scan of the cube.
        exponent = int(np.ceil(np.log2(settings.scan)))
        sobol = qmc.Sobol(ndim, scramble=True, seed=rng)
        cube = sobol.random_base2(exponent)[: settings.scan]
        cube = np.concatenate([np.full((1, ndim), 0.5), cube])
        cube = np.clip(cube, 1e-12, 1.0 - 1e-12)
        scan_z = logit(cube)
        scan_values = evaluate(scan_z)[0]
        counts["scan"] = evaluate.count

        if not np.any(np.isfinite(scan_values)):
            raise RuntimeError(
                "Every point of the scan has zero posterior density; the "
                "fast backend cannot find a starting point."
            )

        order = np.argsort(-np.where(np.isfinite(scan_values), scan_values, -np.inf))
        start_cubes = [cube[index] for index in order[: settings.starts]]

        # 2-4. Local maxima, merged into distinct modes with their geometry.
        modes: list[_Mode] = []
        merged = 0
        recentred = False

        for start_cube in start_cubes:
            start = evaluate.to_current(start_cube)[0]
            scales = _curvature_scales(evaluate, start)
            search = _maximize(
                evaluate,
                start,
                scales,
                max_iterations=settings.max_iterations,
                known=modes,
            )

            if not search.merged:
                new_scales = _curvature_scales(evaluate, search.z)
                ratio = new_scales / search.scales

                if np.any((ratio > RESCALE_FACTOR) | (ratio < 1.0 / RESCALE_FACTOR)):
                    search = _maximize(
                        evaluate,
                        search.z,
                        new_scales,
                        max_iterations=settings.max_iterations,
                        known=modes,
                    )

            if search.merged or (modes and not _distinct(search.z, modes)):
                merged += 1
                continue

            if not np.isfinite(search.log_posterior):
                continue

            if not recentred and np.any(evaluate.cyclic):
                # Move the cut of every cyclic coordinate half a period away
                # from the first mode, then polish the mode in the new
                # coordinates, where the Jacobian of the logit has changed.
                centre = evaluate.recentre(search.z)
                recentred = True
                search = _maximize(
                    evaluate,
                    centre,
                    _curvature_scales(evaluate, centre),
                    max_iterations=settings.max_iterations,
                )

                if not np.isfinite(search.log_posterior):
                    continue

            covariance, positive_definite, change = _geometry(evaluate, search)
            modes.append(
                _Mode(
                    z=search.z,
                    log_posterior=search.log_posterior,
                    covariance=covariance,
                    positive_definite=positive_definite,
                    converged=search.converged,
                    iterations=search.iterations,
                    message=search.message,
                    hessian_step_change=change,
                )
            )

        counts["optimization"] = evaluate.count - counts["scan"]

        if not modes:
            raise RuntimeError("The fast backend found no finite posterior mode.")

        # A mode whose Laplace mass is negligible beside the best is dropped.
        best_log_mass = max(mode.log_mass for mode in modes)
        modes = sorted(
            (
                mode
                for mode in modes
                if best_log_mass - mode.log_mass <= MODE_LOG_MASS_WINDOW
            ),
            key=lambda mode: -mode.log_mass,
        )

        # 5-6. Importance draws.
        log_masses = np.array([mode.log_mass for mode in modes])
        proposal = _Proposal(
            np.stack([mode.z for mode in modes]),
            np.stack([mode.covariance for mode in modes]),
            np.exp(log_masses - np.max(log_masses)),
        )
        batch = _weighted_batch(evaluate, proposal, rng, settings.draws)
        first_pareto_k = batch.pareto_k
        adapted = False

        # 7. One moment-matched adaptation of every component.
        if batch.pareto_k > PARETO_K_ADAPT:
            adapted_proposal = _adapted_proposal(batch, proposal)

            if adapted_proposal is not None:
                proposal = adapted_proposal
                batch = _weighted_batch(evaluate, proposal, rng, settings.draws)
                adapted = True

        counts["draws"] = evaluate.count - counts["scan"] - counts["optimization"]
        best_theta = evaluate.best_theta
        best_log_likelihood = evaluate.best_log_likelihood
        total = evaluate.count

    # 8. Products.
    kept = np.isfinite(batch.log_weights) & np.all(np.isfinite(batch.theta), axis=1)
    weights = np.exp(batch.log_weights[kept])

    if not np.any(weights > 0.0):
        raise RuntimeError("Every fast importance draw has zero weight.")

    flags: list[str] = []

    if not batch.pareto_k <= PARETO_K_UNRELIABLE:
        flags.append(FLAG_PARETO)

    if batch.effective_sample_size < MINIMUM_EFFECTIVE_SAMPLE_SIZE:
        flags.append(FLAG_EFFECTIVE_SAMPLE_SIZE)

    if len(modes) > 1:
        flags.append(FLAG_SEVERAL_MODES)

    if not all(mode.positive_definite for mode in modes):
        flags.append(FLAG_HESSIAN)

    if not all(mode.converged for mode in modes):
        flags.append(FLAG_OPTIMIZER)

    reliable = not any(flag in UNRELIABLE_FLAGS for flag in flags)
    seconds = time.perf_counter() - started
    configuration = settings.as_dict()
    configuration["execution"] = settings.execution.as_dict()

    mode_records = []

    for mode in modes:
        cube = evaluate.cube(mode.z[None, :])[0]
        theta = np.asarray(problem.prior_transform(np.clip(cube, 0.0, 1.0)))
        mode_records.append(
            {
                "parameters": _named(problem, theta),
                "log_posterior_logit_cube": float(mode.log_posterior),
                "standard_deviations_logit_cube": np.sqrt(
                    np.diag(mode.covariance)
                ).tolist(),
                "log_laplace_mass_logit_cube": float(mode.log_mass),
                "hessian_step_change": float(mode.hessian_step_change),
                "positive_definite": bool(mode.positive_definite),
                "converged": bool(mode.converged),
                "iterations": int(mode.iterations),
                "message": mode.message,
            }
        )

    diagnostics = {
        "backend": FAST_BACKEND_NAME,
        "method": FAST_METHOD,
        "evidence": None,
        "calls": int(total),
        "evaluations": {key: int(value) for key, value in counts.items()},
        "draws": int(settings.draws),
        "draws_with_weight": int(np.count_nonzero(weights > 0.0)),
        "pareto_k": float(batch.pareto_k),
        "pareto_k_before_adaptation": float(first_pareto_k),
        "adapted": bool(adapted),
        "effective_sample_size": float(batch.effective_sample_size),
        "ess": float(batch.effective_sample_size),
        "modes": mode_records,
        "starts_merged": int(merged),
        "cyclic_parameters": [
            problem.parameter_names[index] for index in problem.cyclic_coordinates
        ],
        "cyclic_shift_cube": [
            float(evaluate.shift[index]) for index in problem.cyclic_coordinates
        ],
        "raw_log_weights": [
            float(value) if np.isfinite(value) else None
            for value in batch.raw_log_weights
        ],
        "maximum_log_likelihood_found": float(best_log_likelihood),
        "maximum_likelihood_parameters_found": _named(problem, best_theta),
        "flags": flags,
        "reliable": bool(reliable),
        "seconds": float(seconds),
        "seed": int(settings.seed),
    }

    if talk:
        print("────────────────────────────────────────────────────")
        print(f"Completed fast fit: {progress_label}")
        print(f"pareto_k={batch.pareto_k:.3g}  ESS={batch.effective_sample_size:.4g}")
        print(f"ncall={total}  modes={len(modes)}  seconds={seconds:.1f}")
        print(
            "reliable=yes"
            if reliable
            else "reliable=NO: run the evidence mode for this target"
        )

        if flags:
            print(f"flags={', '.join(flags)}")

        print("────────────────────────────────────────────────────")

    posterior = PosteriorSamples(
        samples=batch.theta[kept],
        weights=weights,
        parameter_names=problem.parameter_names,
        log_likelihood=batch.log_likelihood[kept],
    )

    return InferenceResult(
        posterior=posterior,
        evidence=None,
        diagnostics=diagnostics,
        provenance={
            "backend": FAST_BACKEND_NAME,
            "method": FAST_METHOD,
            "sampler": configuration,
        },
    )


class FastBackend(InferenceBackend):
    """Backend of the fast mode: an approximate posterior without evidence.

    Parameters
    ----------
    problem : InferenceProblem
        Complete backend-neutral problem.
    config : mapping, optional
        Entries of ``fit.fast``: ``seed``, ``starts``, ``scan``, ``draws``,
        ``max_iterations`` and ``execution``.

    Notes
    -----
    The backend writes no evidence, so its results cannot enter a model
    comparison. See AD-057 and :func:`run_fast_inference`.
    """

    def __init__(
        self,
        *,
        problem: InferenceProblem,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        settings = resolve_fast_settings(config)
        super().__init__(problem=problem, config=settings.as_dict())
        self.settings = settings

    @property
    def name(self) -> str:
        """Return the stable backend identifier."""
        return FAST_BACKEND_NAME

    def fit(self, *, progress_label: str | None = None) -> InferenceResult:
        """Approximate the posterior and return weighted draws without evidence."""
        return run_fast_inference(
            self.problem,
            self.settings,
            progress_label=progress_label,
        )


__all__ = [
    "FAST_BACKEND_NAME",
    "FAST_DEFAULTS",
    "FAST_METHOD",
    "FastBackend",
    "FastSettings",
    "UNRELIABLE_FLAGS",
    "resolve_fast_settings",
    "run_fast_inference",
]
