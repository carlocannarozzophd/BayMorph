"""Dynesty fitting backend for BayMorph."""

from __future__ import annotations

import dynesty
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


class DynestyFitter:
    """Fit a backend-neutral inference problem with Dynesty.

    Parameters
    ----------
    problem : InferenceProblem
        Complete backend-neutral inference problem.

    Notes
    -----
    BayMorph owns the prior transform, likelihood, and scientific problem
    definition through ``InferenceProblem``. Dynesty-specific constructor
    and runtime options are supplied to :meth:`run`.
    """

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
        """Number of fitted parameters."""
        return self.problem.ndim

    @property
    def parameter_names(self) -> list[str]:
        """Parameter names in sampling order."""
        return self.problem.parameter_names

    @property
    def image_shape(self) -> tuple[int, ...]:
        """Shape expected by a single-image inference problem."""
        image_shape = getattr(
            self.problem,
            "image_shape",
            None,
        )

        if image_shape is None:
            raise AttributeError(
                "This inference problem does not define one image shape."
            )

        return image_shape

    def prior_transform(
        self,
        u,
    ) -> np.ndarray:
        """Transform a unit-cube sample to physical parameters.

        Parameters
        ----------
        u : array-like
            Unit-cube sample with one value per fitted parameter.

        Returns
        -------
        ndarray
            Physical parameter vector in ``ParameterSet`` order.

        Raises
        ------
        ValueError
            If the sample does not contain one value per parameter or contains
            values outside the unit interval.
        """
        unit_values = np.asarray(
            u,
            dtype=float,
        )

        if unit_values.shape != (self.ndim,):
            raise ValueError(
                "Unit-cube sample must contain one value per parameter."
            )

        if not np.all(np.isfinite(unit_values)):
            raise ValueError(
                "Unit-cube sample values must be finite."
            )

        if np.any(
            (unit_values < 0.0)
            | (unit_values > 1.0)
        ):
            raise ValueError(
                "Unit-cube sample values must lie between zero and one."
            )

        transformed = np.asarray(
            self.problem.prior_transform(
                unit_values
            ),
            dtype=float,
        )

        if transformed.shape != (self.ndim,):
            raise ValueError(
                "ParameterSet.transform() must return one value per "
                "parameter."
            )

        if not np.all(np.isfinite(transformed)):
            raise ValueError(
                "Transformed parameter values must be finite."
            )

        return transformed

    def render_model(
        self,
        theta,
    ) -> np.ndarray:
        """Render a model when supported by the inference problem."""
        render_model = getattr(
            self.problem,
            "render_model",
            None,
        )

        if not callable(render_model):
            raise AttributeError(
                "This inference problem does not expose one model image."
            )

        return np.asarray(
            render_model(theta),
            dtype=float,
        )

    def loglikelihood(
        self,
        theta,
    ) -> float:
        """Evaluate the backend-neutral inference problem."""
        return float(
            self.problem.log_likelihood(theta)
        )

    def run(
        self,
        *,
        mode: str = "static",
        init_kwargs=None,
        run_kwargs=None,
        execution: SamplerExecution | None = None,
        progress_label: str | None = None,
    ) -> InferenceResult:
        """Run Dynesty using native constructor and run keyword arguments.

        Parameters
        ----------
        mode : {"static", "dynamic"}, default="static"
            Dynesty nested-sampling mode. ``"static"`` uses
            :class:`dynesty.NestedSampler`; ``"dynamic"`` uses
            :class:`dynesty.DynamicNestedSampler`.
        init_kwargs : mapping, optional
            Native keyword arguments forwarded unchanged to the selected
            Dynesty sampler constructor.
        run_kwargs : mapping, optional
            Native keyword arguments forwarded unchanged to
            ``sampler.run_nested``.
        progress_label : str, optional
            Human-readable BayMorph label displayed around the fit.

        Returns
        -------
        InferenceResult
            Canonical backend-neutral BayMorph inference result.

        Notes
        -----
        BayMorph owns the likelihood, prior transform, and problem
        dimensionality. These constructor arguments therefore cannot be
        supplied through ``init_kwargs``.
        """
        mode = str(
            mode
        ).strip().lower()

        if mode not in {
            "static",
            "dynamic",
        }:
            raise ValueError(
                "Dynesty mode must be 'static' or 'dynamic'."
            )

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
            "loglikelihood",
            "prior_transform",
            "ndim",
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
                "Dynesty constructor argument(s) owned by BayMorph "
                f"cannot be configured explicitly: {names}."
            )

        if execution is not None:
            execution_init_collisions = sorted(
                {
                    "pool",
                    "queue_size",
                }
                & set(init_config)
            )

            if execution_init_collisions:
                names = ", ".join(
                    execution_init_collisions
                )
                raise ValueError(
                    "Dynesty execution argument(s) owned by BayMorph "
                    f"cannot be configured explicitly: {names}."
                )

            if "print_progress" in run_config:
                raise ValueError(
                    "Dynesty run option 'print_progress' is owned by "
                    "fit.sampler.execution.progress."
                )

        if progress_label is not None:
            print()
            print(
                "────────────────────────────────────────────────────"
            )
            print(
                f"Running Dynesty {mode} fit: {progress_label}"
            )
            print(
                f"ndim={self.ndim}"
            )
            print(
                "────────────────────────────────────────────────────"
            )

        sampler_class = (
            dynesty.NestedSampler
            if mode == "static"
            else dynesty.DynamicNestedSampler
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
                "print_progress"
            ] = execution.progress

        with sampler_pool(
            processes
        ) as pool:
            if pool is not None:
                runtime_init_config[
                    "pool"
                ] = pool
                runtime_init_config[
                    "queue_size"
                ] = processes

            sampler = sampler_class(
                self.loglikelihood,
                self.prior_transform,
                self.ndim,
                **runtime_init_config,
            )

            sampler.run_nested(
                **runtime_run_config
            )

            results = sampler.results

        if progress_label is not None:
            total_calls = (
                int(
                    np.sum(
                        results.ncall
                    )
                )
                if hasattr(
                    results,
                    "ncall",
                )
                else "unknown"
            )

            print(
                "────────────────────────────────────────────────────"
            )
            print(
                f"Completed Dynesty {mode} fit: {progress_label}"
            )
            print(
                f"logZ={float(results.logz[-1]):.6g}"
            )
            print(
                f"logZerr={float(results.logzerr[-1]):.6g}"
            )
            print(
                f"ncall={total_calls}"
            )
            print(
                "────────────────────────────────────────────────────"
            )

        final_logz = float(
            results.logz[-1]
        )

        posterior_weights = np.exp(
            np.asarray(
                results.logwt,
                dtype=float,
            )
            - final_logz
        )

        posterior = PosteriorSamples(
            samples=np.asarray(
                results.samples,
                dtype=float,
            ),
            weights=posterior_weights,
            parameter_names=self.parameter_names,
            log_likelihood=np.asarray(
                results.logl,
                dtype=float,
            ),
        )

        total_calls = (
            int(
                np.sum(
                    results.ncall
                )
            )
            if hasattr(
                results,
                "ncall",
            )
            else None
        )

        sampler_configuration = {
            "name": "dynesty",
            "mode": mode,
            "init": init_config,
            "run": run_config,
        }

        if execution is not None:
            sampler_configuration[
                "execution"
            ] = execution.as_dict()

        evidence = EvidenceEstimate(
            logz=final_logz,
            uncertainty=float(
                results.logzerr[-1]
            ),
            method="dynesty",
            uncertainty_kind="nested_sampling_internal",
            diagnostics={
                "mode": mode,
            },
            provenance={
                "backend": "dynesty",
                "sampler": sampler_configuration,
            },
        )

        return InferenceResult(
            posterior=posterior,
            evidence=evidence,
            diagnostics={
                "backend": "dynesty",
                "mode": mode,
                "calls": total_calls,
            },
            provenance={
                "backend": "dynesty",
                "sampler": sampler_configuration,
            },
        )
