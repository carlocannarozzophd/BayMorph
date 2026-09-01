"""Dynesty fitting backend for BayMorph."""

from __future__ import annotations

import sys

import dynesty
import numpy as np

try:
    from IPython.display import HTML, DisplayHandle
except ImportError:
    DisplayHandle = None
    HTML = None

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


class DynestyMultilineProgress:
    """Live Dynesty progress display for terminals and Jupyter notebooks."""

    def __init__(
        self,
        label: str | None = None,
    ) -> None:
        """Initialize the progress display.

        Parameters
        ----------
        label : str, optional
            Human-readable label identifying the fit being monitored.
        """
        self.label = label or "dynesty"
        self._isatty = sys.stdout.isatty()
        self._display_handle = None
        self._last_logl = np.nan
        self._last_delta_logz = np.nan
        self._n_lines = 8
        self.update_every = 100

        if (
            not self._isatty
            and DisplayHandle is not None
            and HTML is not None
        ):
            self._display_handle = DisplayHandle()
            self._display_handle.display(
                HTML("")
            )

    @staticmethod
    def _last(
        value,
    ) -> float:
        """Return the final value from a scalar or array-like object."""
        array = np.atleast_1d(value)

        if array.size == 0:
            return np.nan

        return float(array[-1])

    @staticmethod
    def _field(
        results,
        *names,
    ):
        """Return the first available field from Dynesty results."""
        for name in names:
            if hasattr(results, name):
                return getattr(results, name)

        return np.nan

    @staticmethod
    def _format_or_pending(
        value,
        fmt: str,
    ) -> str:
        """Format a finite value or return a pending-status string."""
        if np.isfinite(value):
            return format(value, fmt)

        return "pending"

    def __call__(
        self,
        results,
        niter,
        ncall,
        dlogz,
        *args,
        **kwargs,
    ) -> None:
        """Update the live progress display from a Dynesty callback.

        Parameters
        ----------
        results : object
            Partial Dynesty results object supplied by the sampler.
        niter : int
            Current nested-sampling iteration.
        ncall : int
            Current number of likelihood evaluations.
        dlogz : float
            Target stopping threshold on remaining evidence.
        *args
            Additional positional arguments supplied by Dynesty.
        **kwargs
            Additional keyword arguments supplied by Dynesty.
        """
        del args, kwargs

        if niter % self.update_every != 0:
            return

        logz = self._last(
            self._field(
                results,
                "logz",
            )
        )

        logl_new = self._last(
            self._field(
                results,
                "loglstar",
                "logl",
                "logl_birth",
            )
        )

        delta_logz_new = self._last(
            self._field(
                results,
                "delta_logz",
                "dlogz",
                "dlogz_current",
                "logz_remain",
            )
        )

        if np.isfinite(logl_new):
            self._last_logl = logl_new

        if np.isfinite(delta_logz_new):
            self._last_delta_logz = delta_logz_new

        efficiency = (
            100.0 * niter / ncall
            if ncall > 0
            else np.nan
        )

        lines = [
            "Dynesty fit",
            f"Model      : {self.label}",
            f"Iteration  : {niter}",
            f"Calls      : {ncall}",
            (
                "Efficiency : "
                f"{self._format_or_pending(efficiency, '.3f')}%"
            ),
            (
                "logL       : "
                f"{self._format_or_pending(self._last_logl, '.6g')}"
            ),
            (
                "logZ       : "
                f"{self._format_or_pending(logz, '.6g')}"
            ),
            (
                "dlogZ      : "
                f"{self._format_or_pending(self._last_delta_logz, '.6g')}"
                f" > {float(dlogz):.6g}"
            ),
        ]

        if self._isatty:
            print(
                f"\033[{self._n_lines}F",
                end="",
            )

            for line in lines:
                print(
                    "\033[K" + line,
                )

        elif self._display_handle is not None:
            html = "\n".join(lines)

            self._display_handle.update(
                HTML(
                    f"<pre>{html}</pre>"
                )
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
