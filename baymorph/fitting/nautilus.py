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

        weights = np.exp(
            log_weights
        )

        logz = sampler.log_z

        if logz is None:
            raise RuntimeError(
                "Nautilus completed without producing an evidence estimate."
            )

        diagnostics = {
            "backend": "nautilus",
            "success": success,
            "calls": int(
                sampler.n_like
            ),
            "ess": float(
                sampler.n_eff
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
                    sampler.n_eff
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
