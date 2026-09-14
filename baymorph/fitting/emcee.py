"""Posterior-only emcee fitting backend for BayMorph."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from baymorph.fitting.execution import SamplerExecution, sampler_pool
from baymorph.fitting.emcee_checkpoint import (
    EmceeCheckpointState,
    create_emcee_checkpoint,
    emcee_checkpoint_metadata,
    ensure_fresh_checkpoint_available,
    load_emcee_checkpoint,
    pop_emcee_checkpoint_request,
    record_emcee_checkpoint_progress,
)
from baymorph.fitting.problem import InferenceProblem
from baymorph.results import InferenceResult, PosteriorSamples


_EMCEE_WORKER_PROBLEM: InferenceProblem | None = None


def _initialize_emcee_worker(problem: InferenceProblem) -> None:
    """Install one inference problem in a persistent worker process."""
    global _EMCEE_WORKER_PROBLEM

    _EMCEE_WORKER_PROBLEM = problem


def _evaluate_log_probability(
    problem: InferenceProblem,
    cube,
) -> float:
    """Evaluate the posterior density in unit-cube coordinates."""
    coordinates = np.asarray(
        cube,
        dtype=float,
    )

    expected_shape = (problem.ndim,)

    if coordinates.shape != expected_shape:
        raise ValueError(
            "emcee coordinates must have shape "
            f"{expected_shape}, received {coordinates.shape}."
        )

    if (
        not np.all(np.isfinite(coordinates))
        or np.any(coordinates < 0.0)
        or np.any(coordinates > 1.0)
    ):
        return float("-inf")

    theta = np.asarray(
        problem.prior_transform(coordinates),
        dtype=float,
    )

    if theta.shape != expected_shape:
        raise ValueError(
            "InferenceProblem.prior_transform must return shape "
            f"{expected_shape}, received {theta.shape}."
        )

    if not np.all(np.isfinite(theta)):
        return float("-inf")

    value = float(
        problem.log_likelihood(theta)
    )

    if np.isnan(value) or np.isposinf(value):
        raise ValueError(
            "InferenceProblem.log_likelihood returned NaN or positive infinity."
        )

    return value


def _worker_log_probability(cube) -> float:
    """Evaluate one unit-cube point inside an emcee worker."""
    if _EMCEE_WORKER_PROBLEM is None:
        raise RuntimeError(
            "The emcee worker inference problem was not initialized."
        )

    return _evaluate_log_probability(
        _EMCEE_WORKER_PROBLEM,
        cube,
    )


def _validate_resumed_log_probability(
    *,
    state,
    log_probability,
    pool,
) -> None:
    """Verify stored walker densities before advancing a resumed chain."""
    coordinates = np.asarray(
        state.coords,
        dtype=float,
    )
    stored = np.asarray(
        state.log_prob,
        dtype=float,
    )

    if coordinates.ndim != 2:
        raise ValueError(
            "emcee checkpoint coordinates are invalid."
        )

    expected_shape = (
        coordinates.shape[
            0
        ],
    )

    if stored.shape != expected_shape:
        raise ValueError(
            "emcee checkpoint log-probability shape is invalid."
        )

    if pool is None:
        recomputed = np.asarray(
            [
                log_probability(
                    coordinate
                )
                for coordinate in coordinates
            ],
            dtype=float,
        )
    else:
        recomputed = np.asarray(
            pool.map(
                log_probability,
                coordinates,
            ),
            dtype=float,
        )

    if recomputed.shape != expected_shape:
        raise ValueError(
            "Recomputed emcee log probabilities have an "
            "invalid shape."
        )

    if (
        not np.all(
            np.isfinite(
                stored
            )
        )
        or not np.all(
            np.isfinite(
                recomputed
            )
        )
    ):
        raise ValueError(
            "emcee checkpoint log probabilities must be finite."
        )

    difference = np.abs(
        recomputed
        - stored
    )
    tolerance = (
        1.0e-10
        + 1.0e-12
        * np.maximum(
            np.abs(
                recomputed
            ),
            np.abs(
                stored
            ),
        )
    )

    if np.any(
        difference > tolerance
    ):
        raise ValueError(
            "Stored emcee log probabilities do not match "
            "the current inference problem."
        )


def _integer_option(
    value,
    *,
    name: str,
    minimum: int,
) -> int:
    """Validate one integer emcee option."""
    if isinstance(value, bool) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    resolved = int(value)

    if resolved < minimum:
        raise ValueError(
            f"{name} must be at least {minimum}."
        )

    return resolved


def _float_option(
    value,
    *,
    name: str,
    minimum: float,
    allow_equal: bool,
) -> float:
    """Validate one finite floating-point emcee option."""
    if isinstance(value, bool):
        raise TypeError(
            f"{name} must be a real number."
        )

    resolved = float(value)

    if not np.isfinite(resolved):
        raise ValueError(
            f"{name} must be finite."
        )

    invalid = (
        resolved < minimum
        if allow_equal
        else resolved <= minimum
    )

    if invalid:
        comparison = "at least" if allow_equal else "greater than"
        raise ValueError(
            f"{name} must be {comparison} {minimum}."
        )

    return resolved


def _resolve_nwalkers(
    value,
    *,
    ndim: int,
) -> int:
    """Resolve an explicit or automatic ensemble size."""
    if isinstance(value, str):
        if value.strip().lower() != "auto":
            raise ValueError(
                "emcee nwalkers must be 'auto' or a positive even integer."
            )

        resolved = max(
            32,
            4 * ndim,
        )

        if resolved % 2:
            resolved += 1
    else:
        resolved = _integer_option(
            value,
            name="emcee nwalkers",
            minimum=2,
        )

    if resolved % 2:
        raise ValueError(
            "emcee nwalkers must be even."
        )

    minimum_walkers = 2 * ndim

    if resolved < minimum_walkers:
        raise ValueError(
            "emcee nwalkers must be at least twice the inference "
            f"dimension ({minimum_walkers})."
        )

    return resolved


class EmceeFitter:
    """Fit a backend-neutral problem with posterior-only ensemble MCMC."""

    def __init__(
        self,
        *,
        problem: InferenceProblem,
    ) -> None:
        if not isinstance(
            problem,
            InferenceProblem,
        ):
            raise TypeError(
                "problem must be an InferenceProblem instance."
            )

        self.problem = problem

    @property
    def ndim(self) -> int:
        """Return the number of sampled parameters."""
        return self.problem.ndim

    @property
    def parameter_names(self) -> list[str]:
        """Return parameter names in sampling order."""
        return self.problem.parameter_names

    def prior_transform(
        self,
        cube,
    ) -> np.ndarray:
        """Transform one unit-cube point to physical parameters."""
        return np.asarray(
            self.problem.prior_transform(
                np.asarray(
                    cube,
                    dtype=float,
                )
            ),
            dtype=float,
        )

    def log_probability(
        self,
        cube,
    ) -> float:
        """Evaluate the posterior in unit-cube coordinates."""
        return _evaluate_log_probability(
            self.problem,
            cube,
        )

    def _initial_coordinates(
        self,
        *,
        nwalkers: int,
        random_state,
        max_attempts: int,
    ) -> tuple[np.ndarray, int]:
        """Draw finite prior coordinates for every walker."""
        lower = np.nextafter(
            0.0,
            1.0,
        )
        upper = np.nextafter(
            1.0,
            0.0,
        )

        coordinates = np.empty(
            (
                nwalkers,
                self.ndim,
            ),
            dtype=float,
        )
        calls = 0

        for walker in range(nwalkers):
            for _ in range(max_attempts):
                candidate = random_state.uniform(
                    lower,
                    upper,
                    size=self.ndim,
                )
                calls += 1

                if np.isfinite(
                    self.log_probability(candidate)
                ):
                    coordinates[walker] = candidate
                    break
            else:
                raise RuntimeError(
                    "Unable to initialize every emcee walker at a finite "
                    "posterior point."
                )

        return coordinates, calls

    def run(
        self,
        *,
        init_kwargs: Mapping[str, Any] | None = None,
        run_kwargs: Mapping[str, Any] | None = None,
        execution: SamplerExecution | None = None,
        progress_label: str | None = None,
    ) -> InferenceResult:
        """Run emcee and return an unweighted posterior without evidence."""
        init_config = dict(
            {}
            if init_kwargs is None
            else init_kwargs
        )
        run_config = dict(
            {}
            if run_kwargs is None
            else run_kwargs
        )

        checkpoint_request = pop_emcee_checkpoint_request(
            init_config
        )

        if execution is not None and not isinstance(
            execution,
            SamplerExecution,
        ):
            raise TypeError(
                "execution must be a SamplerExecution instance or None."
            )

        nwalkers = _resolve_nwalkers(
            init_config.pop(
                "nwalkers",
                "auto",
            ),
            ndim=self.ndim,
        )

        seed_value = init_config.pop(
            "seed",
            None,
        )
        seed = (
            None
            if seed_value is None
            else _integer_option(
                seed_value,
                name="emcee seed",
                minimum=0,
            )
        )

        max_initialization_attempts = _integer_option(
            init_config.pop(
                "max_initialization_attempts",
                1000,
            ),
            name="emcee max_initialization_attempts",
            minimum=1,
        )

        reserved_init = {
            "backend",
            "kwargs",
            "log_prob_fn",
            "ndim",
            "parameter_names",
            "pool",
            "vectorize",
        }
        init_collisions = sorted(
            reserved_init
            & set(init_config)
        )

        if init_collisions:
            names = ", ".join(init_collisions)
            raise ValueError(
                "emcee constructor argument(s) owned by BayMorph "
                f"cannot be configured explicitly: {names}."
            )

        burn_in = _integer_option(
            run_config.pop(
                "burn_in",
                500,
            ),
            name="emcee burn_in",
            minimum=0,
        )
        max_steps = _integer_option(
            run_config.pop(
                "max_steps",
                10000,
            ),
            name="emcee max_steps",
            minimum=1,
        )
        check_interval = _integer_option(
            run_config.pop(
                "check_interval",
                100,
            ),
            name="emcee check_interval",
            minimum=1,
        )
        min_autocorr_times = _float_option(
            run_config.pop(
                "min_autocorr_times",
                50.0,
            ),
            name="emcee min_autocorr_times",
            minimum=0.0,
            allow_equal=False,
        )
        tau_tolerance = _float_option(
            run_config.pop(
                "tau_tolerance",
                0.05,
            ),
            name="emcee tau_tolerance",
            minimum=0.0,
            allow_equal=True,
        )

        reserved_run = {
            "initial_state",
            "iterations",
            "progress",
            "store",
            "thin",
            "thin_by",
        }
        run_collisions = sorted(
            reserved_run
            & set(run_config)
        )

        if run_collisions:
            names = ", ".join(run_collisions)
            raise ValueError(
                "emcee run argument(s) owned by BayMorph cannot be "
                f"configured explicitly: {names}."
            )

        try:
            import emcee as emcee_library
        except ImportError as exc:
            raise ImportError(
                "emcee support requires emcee>=3.1.6,<4. "
                "Install BayMorph with its declared dependencies."
            ) from exc

        checkpoint_metadata = None
        checkpoint: EmceeCheckpointState | None = None

        if checkpoint_request is not None:
            if seed is None:
                raise ValueError(
                    "Checkpointed emcee runs require an explicit seed."
                )

            checkpoint_metadata = emcee_checkpoint_metadata(
                request=checkpoint_request,
                emcee_version=getattr(
                    emcee_library,
                    "__version__",
                    "unknown",
                ),
                problem_class=(
                    f"{type(self.problem).__module__}."
                    f"{type(self.problem).__qualname__}"
                ),
                ndim=self.ndim,
                parameter_names=self.parameter_names,
                nwalkers=nwalkers,
                burn_in=burn_in,
                seed=seed,
                max_initialization_attempts=(
                    max_initialization_attempts
                ),
                init_config=init_config,
                run_config=run_config,
                check_interval=check_interval,
                min_autocorr_times=min_autocorr_times,
                tau_tolerance=tau_tolerance,
            )

            if checkpoint_request.mode == "fresh":
                ensure_fresh_checkpoint_available(
                    checkpoint_request
                )
            else:
                checkpoint = load_emcee_checkpoint(
                    request=checkpoint_request,
                    emcee_library=emcee_library,
                    metadata=checkpoint_metadata,
                    nwalkers=nwalkers,
                    ndim=self.ndim,
                )

        initial_state = None

        if checkpoint is None:
            random_state = np.random.RandomState(
                seed
            )
            initial_coordinates, initialization_calls = (
                self._initial_coordinates(
                    nwalkers=nwalkers,
                    random_state=random_state,
                    max_attempts=max_initialization_attempts,
                )
            )
            initial_state = emcee_library.State(
                initial_coordinates,
                random_state=random_state.get_state(),
            )
        else:
            initialization_calls = (
                checkpoint.initialization_calls
            )

        processes = (
            1
            if execution is None
            else execution.processes
        )
        progress = (
            False
            if execution is None
            else execution.progress
        )

        if progress_label is not None:
            print()
            print(
                "────────────────────────────────────────────────────"
            )
            print(
                f"Running emcee posterior fit: {progress_label}"
            )
            print(
                f"ndim={self.ndim}, walkers={nwalkers}, "
                f"processes={processes}"
            )
            print(
                "Evidence: unavailable (posterior-only backend)"
            )
            print(
                "────────────────────────────────────────────────────"
            )

        previous_tau = (
            None
            if checkpoint is None
            else checkpoint.previous_tau
        )
        current_tau = previous_tau
        converged = (
            False
            if checkpoint is None
            else checkpoint.converged
        )

        with sampler_pool(
            processes,
            initializer=_initialize_emcee_worker,
            initargs=(self.problem,),
        ) as pool:
            log_probability = (
                self.log_probability
                if pool is None
                else _worker_log_probability
            )

            def make_ensemble_sampler(
                *,
                backend=None,
            ):
                return emcee_library.EnsembleSampler(
                    nwalkers,
                    self.ndim,
                    log_probability,
                    pool=pool,
                    backend=backend,
                    **init_config,
                )

            if checkpoint is not None:
                sampler = make_ensemble_sampler(
                    backend=checkpoint.backend
                )
                state = sampler.get_last_sample()

                _validate_resumed_log_probability(
                    state=state,
                    log_probability=log_probability,
                    pool=pool,
                )
            else:
                warmup_sampler = make_ensemble_sampler()

                if burn_in:
                    state = warmup_sampler.run_mcmc(
                        initial_state,
                        burn_in,
                        progress=progress,
                        **run_config,
                    )
                else:
                    state = initial_state

                if checkpoint_request is None:
                    if burn_in:
                        warmup_sampler.reset()

                    sampler = warmup_sampler
                else:
                    checkpoint = create_emcee_checkpoint(
                        request=checkpoint_request,
                        emcee_library=emcee_library,
                        nwalkers=nwalkers,
                        ndim=self.ndim,
                        metadata=checkpoint_metadata,
                        initialization_calls=(
                            initialization_calls
                        ),
                    )
                    sampler = make_ensemble_sampler(
                        backend=checkpoint.backend
                    )

            preexisting_steps = int(
                sampler.iteration
            )

            if preexisting_steps > max_steps:
                raise ValueError(
                    "emcee checkpoint already contains more "
                    "production steps "
                    f"({preexisting_steps}) than max_steps "
                    f"({max_steps})."
                )

            remaining_steps = (
                0
                if converged
                else max_steps - preexisting_steps
            )

            if remaining_steps:
                for _ in sampler.sample(
                    state,
                    iterations=remaining_steps,
                    progress=progress,
                    **run_config,
                ):
                    if (
                        sampler.iteration
                        % check_interval
                    ):
                        continue

                    tau = np.asarray(
                        sampler.get_autocorr_time(
                            tol=0,
                        ),
                        dtype=float,
                    )

                    if (
                        tau.shape != (self.ndim,)
                        or not np.all(
                            np.isfinite(
                                tau
                            )
                        )
                        or np.any(
                            tau <= 0.0
                        )
                    ):
                        continue

                    current_tau = tau
                    enough_samples = bool(
                        np.all(
                            sampler.iteration
                            >= min_autocorr_times
                            * tau
                        )
                    )
                    stable_tau = (
                        previous_tau is not None
                        and bool(
                            np.all(
                                np.abs(
                                    previous_tau
                                    - tau
                                )
                                / tau
                                <= tau_tolerance
                            )
                        )
                    )
                    converged_now = bool(
                        enough_samples
                        and stable_tau
                    )

                    record_emcee_checkpoint_progress(
                        checkpoint,
                        tau=tau,
                        converged=converged_now,
                    )

                    if converged_now:
                        converged = True
                        break

                    previous_tau = tau

        completed_steps = int(
            sampler.iteration
        )
        added_steps = (
            completed_steps
            - preexisting_steps
        )
        termination_reason = (
            "converged"
            if converged
            else "max_steps"
        )

        record_emcee_checkpoint_progress(
            checkpoint,
            tau=current_tau,
            converged=converged,
        )
        cube_samples = np.asarray(
            sampler.get_chain(
                flat=True,
            ),
            dtype=float,
        )
        log_likelihood = np.asarray(
            sampler.get_log_prob(
                flat=True,
            ),
            dtype=float,
        )
        samples = np.asarray(
            [
                self.prior_transform(cube)
                for cube in cube_samples
            ],
            dtype=float,
        )

        n_samples = int(
            samples.shape[0]
        )
        weights = np.full(
            n_samples,
            1.0 / n_samples,
            dtype=float,
        )

        if current_tau is None:
            tau_values = None
            ess_values = None
            minimum_mcmc_ess = None
        else:
            tau_values = current_tau.tolist()
            ess_array = (
                sampler.iteration
                * nwalkers
                / current_tau
            )
            ess_values = ess_array.tolist()
            minimum_mcmc_ess = float(
                np.min(ess_array)
            )

        acceptance = np.asarray(
            sampler.acceptance_fraction,
            dtype=float,
        )

        diagnostics = {
            "backend": "emcee",
            "success": converged,
            "converged": converged,
            "walkers": nwalkers,
            "burn_in": burn_in,
            "steps": int(sampler.iteration),
            "stored_samples": n_samples,
            "initialization_calls": initialization_calls,
            "autocorrelation_time": tau_values,
            "mcmc_effective_sample_size": minimum_mcmc_ess,
            "mcmc_effective_sample_size_by_parameter": ess_values,
            "acceptance_fraction_mean": float(
                np.mean(acceptance)
            ),
            "acceptance_fraction_min": float(
                np.min(acceptance)
            ),
            "acceptance_fraction_max": float(
                np.max(acceptance)
            ),
            "checkpoint_path": (
                None
                if checkpoint_request is None
                else str(
                    checkpoint_request.path
                )
            ),
            "checkpoint_signature": (
                None
                if checkpoint_request is None
                else checkpoint_request.signature
            ),
            "checkpoint_resumed": bool(
                checkpoint_request is not None
                and checkpoint_request.mode == "resume"
            ),
            "preexisting_steps": preexisting_steps,
            "added_steps": added_steps,
            "termination_reason": termination_reason,
            "evidence_available": False,
        }

        checkpoint_init_config = {}

        if checkpoint_request is not None:
            checkpoint_init_config = {
                "checkpoint_path": str(
                    checkpoint_request.path
                ),
                "checkpoint_mode": (
                    checkpoint_request.mode
                ),
                "checkpoint_signature": (
                    checkpoint_request.signature
                ),
            }

        sampler_configuration = {
            "name": "emcee",
            "init": {
                "nwalkers": nwalkers,
                "seed": seed,
                "max_initialization_attempts": (
                    max_initialization_attempts
                ),
                **checkpoint_init_config,
                **init_config,
            },
            "run": {
                "burn_in": burn_in,
                "max_steps": max_steps,
                "check_interval": check_interval,
                "min_autocorr_times": min_autocorr_times,
                "tau_tolerance": tau_tolerance,
                **run_config,
            },
        }

        if execution is not None:
            sampler_configuration[
                "execution"
            ] = execution.as_dict()

        if progress_label is not None:
            print(
                "────────────────────────────────────────────────────"
            )
            print(
                f"Completed emcee fit: {progress_label}"
            )
            print(
                f"steps={sampler.iteration}, "
                f"converged={converged}"
            )
            print(
                "logZ=None"
            )
            print(
                "────────────────────────────────────────────────────"
            )

        posterior = PosteriorSamples(
            samples=samples,
            weights=weights,
            parameter_names=self.parameter_names,
            log_likelihood=log_likelihood,
        )

        return InferenceResult(
            posterior=posterior,
            evidence=None,
            diagnostics=diagnostics,
            provenance={
                "backend": "emcee",
                "inference_kind": "posterior_only",
                "library_version": getattr(
                    emcee_library,
                    "__version__",
                    None,
                ),
                "sampler": sampler_configuration,
            },
        )


__all__ = [
    "EmceeFitter",
]
