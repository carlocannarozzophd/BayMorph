"""Nautilus fitting backend for BayMorph."""

from __future__ import annotations

import numpy as np

from baymorph.fitting.execution import (
    SamplerExecution,
    sampler_pool,
)
from baymorph.fitting.problem import InferenceProblem
from baymorph.results import (
    EvidenceEstimate,
    InferenceResult,
    PosteriorSamples,
)


class NautilusIncompleteRunError(RuntimeError):
    """Report a Nautilus run that has not reached its requested target."""

    status = "INCOMPLETE"
    reason_code = "NAUTILUS_INCOMPLETE"

    def __init__(
        self,
        *,
        checkpoint_path: str | None,
    ) -> None:
        self.checkpoint_path = checkpoint_path

        if checkpoint_path is None:
            checkpoint_message = (
                "No native checkpoint path was configured; this run "
                "cannot be resumed safely."
            )
        else:
            checkpoint_message = (
                f"Native checkpoint path: {checkpoint_path!r}. Resume it "
                "only after external manifest validation."
            )

        super().__init__(
            "Nautilus run status is INCOMPLETE: Sampler.run() returned "
            "False before the requested convergence target was reached. "
            "No posterior or evidence result was created. "
            f"{checkpoint_message}"
        )


class NautilusFitter:
    """Fit a backend-neutral inference problem with Nautilus."""

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

    def loglikelihood(
        self,
        theta,
    ) -> float:
        """Evaluate the backend-neutral inference problem."""
        return float(
            self.problem.log_likelihood(
                np.asarray(
                    theta,
                    dtype=float,
                )
            )
        )

    def run(
        self,
        *,
        init_kwargs=None,
        run_kwargs=None,
        execution: SamplerExecution | None = None,
        progress_label: str | None = None,
    ) -> InferenceResult:
        """Run Nautilus using native constructor and run keyword arguments."""
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

        reserved_init = {
            "prior",
            "likelihood",
            "n_dim",
            "pass_dict",
            "vectorized",
            "prior_args",
            "prior_kwargs",
            "likelihood_args",
            "likelihood_kwargs",
        }

        collisions = sorted(
            reserved_init
            & set(init_config)
        )

        if collisions:
            names = ", ".join(
                collisions
            )
            raise ValueError(
                "Nautilus constructor argument(s) owned by BayMorph "
                f"cannot be configured explicitly: {names}."
            )

        if execution is not None:
            if "pool" in init_config:
                raise ValueError(
                    "Nautilus constructor option 'pool' is owned by "
                    "fit.sampler.execution.processes."
                )

            if "verbose" in run_config:
                raise ValueError(
                    "Nautilus run option 'verbose' is owned by "
                    "fit.sampler.execution.progress."
                )

        run_config.setdefault(
            "discard_exploration",
            True,
        )

        try:
            from nautilus import Sampler
        except ImportError as exc:
            raise ImportError(
                "Nautilus support requires the optional nautilus-sampler "
                "dependency. Install it with "
                '`pip install "baymorph[nautilus]"`.'
            ) from exc

        if progress_label is not None:
            print()
            print(
                "────────────────────────────────────────────────────"
            )
            print(
                f"Running Nautilus fit: {progress_label}"
            )
            print(
                f"ndim={self.ndim}"
            )
            print(
                "────────────────────────────────────────────────────"
            )

        runtime_init_config = dict(
            init_config
        )
        runtime_run_config = dict(
            run_config
        )

        processes = (
            1
            if execution is None
            else execution.processes
        )

        if execution is not None:
            runtime_run_config[
                "verbose"
            ] = execution.progress

        with sampler_pool(
            processes
        ) as pool:
            if pool is not None:
                runtime_init_config[
                    "pool"
                ] = pool

            sampler = Sampler(
                self.prior_transform,
                self.loglikelihood,
                n_dim=self.ndim,
                pass_dict=False,
                vectorized=False,
                **runtime_init_config,
            )

            success = bool(
                sampler.run(
                    **runtime_run_config
                )
            )

        if not success:
            configured_checkpoint = init_config.get(
                "filepath"
            )

            raise NautilusIncompleteRunError(
                checkpoint_path=(
                    None
                    if configured_checkpoint is None
                    else str(
                        configured_checkpoint
                    )
                )
            )

        samples, log_weights, log_likelihood = sampler.posterior(
            return_as_dict=False,
            equal_weight=False,
        )

        samples = np.asarray(
            samples,
            dtype=float,
        )
        log_weights = np.asarray(
            log_weights,
            dtype=float,
        )
        log_likelihood = np.asarray(
            log_likelihood,
            dtype=float,
        )

        n_samples = (
            int(
                samples.shape[0]
            )
            if samples.ndim == 2
            else 0
        )

        if (
            samples.ndim != 2
            or n_samples == 0
            or samples.shape[1] != self.ndim
        ):
            raise RuntimeError(
                "Nautilus produced an invalid or empty posterior sample "
                "array."
            )

        if log_weights.shape != (
            n_samples,
        ):
            raise RuntimeError(
                "Nautilus produced posterior log weights with an invalid "
                "shape."
            )

        if log_likelihood.shape != (
            n_samples,
        ):
            raise RuntimeError(
                "Nautilus produced posterior log likelihoods with an "
                "invalid shape."
            )

        if not np.all(
            np.isfinite(
                samples
            )
        ):
            raise RuntimeError(
                "Nautilus produced non-finite posterior samples."
            )

        invalid_log_weights = (
            np.isnan(log_weights)
            | np.isposinf(log_weights)
        )

        if np.any(invalid_log_weights):
            raise RuntimeError(
                "Nautilus produced non-finite posterior log weights: NaN "
                "and positive infinity are invalid."
            )

        finite_log_weights = np.isfinite(log_weights)

        if not np.any(finite_log_weights):
            raise RuntimeError(
                "Nautilus produced non-finite posterior log weights: at "
                "least one finite log weight is required."
            )

        invalid_log_likelihood = (
            np.isnan(log_likelihood)
            | np.isposinf(log_likelihood)
        )

        if np.any(invalid_log_likelihood):
            raise RuntimeError(
                "Nautilus produced non-finite posterior log likelihoods: "
                "NaN and positive infinity are invalid."
            )

        if np.any(
            np.isneginf(log_likelihood)
            & finite_log_weights
        ):
            raise RuntimeError(
                "Nautilus assigned finite posterior log weight to a sample "
                "with negative-infinite log likelihood."
            )

        maximum_log_weight = float(
            np.max(log_weights[finite_log_weights])
        )
        weights = np.exp(
            log_weights - maximum_log_weight
        )

        logz = sampler.log_z

        if (
            logz is None
            or not np.isfinite(
                float(
                    logz
                )
            )
        ):
            raise RuntimeError(
                "Nautilus completed without producing a finite evidence "
                "estimate."
            )

        effective_sample_size = float(
            sampler.n_eff
        )

        if (
            not np.isfinite(
                effective_sample_size
            )
            or effective_sample_size <= 0.0
        ):
            raise RuntimeError(
                "Nautilus completed without producing a finite positive "
                "effective sample size."
            )

        diagnostics = {
            "backend": "nautilus",
            "success": success,
            "calls": int(
                sampler.n_like
            ),
            "ess": float(
                effective_sample_size
            ),
            "efficiency": float(
                sampler.eta
            ),
            "f_live": (
                None
                if sampler.f_live is None
                else float(
                    sampler.f_live
                )
            ),
            "discard_exploration": bool(
                sampler.discard_exploration
            ),
        }

        if progress_label is not None:
            print(
                "────────────────────────────────────────────────────"
            )
            print(
                f"Completed Nautilus fit: {progress_label}"
            )
            print(
                f"logZ={float(logz):.6g}"
            )
            print(
                f"ncall={int(sampler.n_like)}"
            )
            print(
                f"N_eff={float(sampler.n_eff):.6g}"
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

        sampler_configuration = {
            "name": "nautilus",
            "init": init_config,
            "run": run_config,
        }

        if execution is not None:
            sampler_configuration[
                "execution"
            ] = execution.as_dict()

        evidence_statistical_uncertainty = (
            1.0
            / np.sqrt(
                float(
                    effective_sample_size
                )
            )
        )

        evidence = EvidenceEstimate(
            logz=float(
                logz
            ),
            uncertainty=float(
                evidence_statistical_uncertainty
            ),
            method="nautilus",
            uncertainty_kind="statistical_ess_approximation",
            diagnostics={
                "discard_exploration": bool(
                    sampler.discard_exploration
                ),
            },
            provenance={
                "backend": "nautilus",
                "sampler": sampler_configuration,
            },
        )

        return InferenceResult(
            posterior=posterior,
            evidence=evidence,
            diagnostics=diagnostics,
            provenance={
                "backend": "nautilus",
                "sampler": sampler_configuration,
            },
        )
