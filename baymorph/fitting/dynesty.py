"""Dynesty fitting backend for BayMorph."""

from __future__ import annotations

import sys
from multiprocessing import Pool

import dynesty
import numpy as np

try:
    from IPython.display import HTML, DisplayHandle
except ImportError:
    DisplayHandle = None
    HTML = None

from baymorph.fitting.image_problem import ImageInferenceProblem
from baymorph.fitting.parameters import ParameterSet
from baymorph.fitting.problem import InferenceProblem
from baymorph.likelihood import ImageLikelihood
from baymorph.results import PosteriorResult


class DynestyResult(PosteriorResult):
    """Posterior result produced by the Dynesty fitting backend."""


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
    """Fit an image model with the Dynesty nested sampler.

    Parameters
    ----------
    model_factory : callable
        Callable receiving model parameters as keyword arguments and returning
        an object exposing a ``render(shape=...)`` method.
    parameters : ParameterSet
        Ordered definition of fitted parameters and their prior transforms.
    likelihood : ImageLikelihood
        Fully configured image likelihood. The likelihood owns the observed
        image, mask, and all fixed statistical quantities.
    psf : array-like, optional
        Point-spread function applied to the rendered model before likelihood
        evaluation. This should be omitted when ``model_factory`` already
        returns a PSF-convolved model.

    Notes
    -----
    The fitter is independent of the statistical model. It only requires a
    likelihood satisfying

    ``loglike = likelihood(model_image)``.

    YAML interpretation and likelihood construction belong to the pipeline,
    not to this fitting backend.
    """

    def __init__(
        self,
        model_factory=None,
        parameters: ParameterSet | None = None,
        likelihood: ImageLikelihood | None = None,
        psf=None,
        *,
        problem: InferenceProblem | None = None,
    ) -> None:
        if problem is not None:
            if any(
                value is not None
                for value in (
                    model_factory,
                    parameters,
                    likelihood,
                    psf,
                )
            ):
                raise ValueError(
                    "problem cannot be combined with model_factory, "
                    "parameters, likelihood, or psf."
                )

            if not isinstance(problem, InferenceProblem):
                raise TypeError(
                    "problem must be an InferenceProblem instance."
                )

            self.problem = problem
            return

        self.problem = ImageInferenceProblem(
            model_factory=model_factory,
            parameters=parameters,
            likelihood=likelihood,
            psf=psf,
        )

        # Preserve the existing public single-image interface while the
        # sampler backend delegates all inference operations to ``problem``.
        self.model_factory = model_factory
        self.parameters = parameters
        self.likelihood = likelihood
        self.psf = psf

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
        nlive: int = 100,
        sample: str = "auto",
        bound: str = "multi",
        dlogz: float | None = None,
        bootstrap: int | None = None,
        print_progress: bool = True,
        n_processes: int = 1,
        queue_size: int | None = None,
        seed: int | None = None,
        progress_label: str | None = None,
    ) -> DynestyResult:
        """Run nested sampling.

        Parameters
        ----------
        nlive : int, default=100
            Number of live points used by Dynesty.
        sample : str, default="auto"
            Dynesty sampling method.
        bound : str, default="multi"
            Dynesty bounding method.
        dlogz : float, optional
            Stopping threshold on the remaining evidence.
        bootstrap : int, optional
            Number of bootstrap realizations used for bounding.
        print_progress : bool, default=True
            If True, display Dynesty progress information.
        n_processes : int, default=1
            Number of worker processes used for likelihood evaluations.
        queue_size : int, optional
            Number of queued likelihood evaluations when multiprocessing is
            enabled. If ``None``, it defaults to ``n_processes``.
        seed : int, optional
            Random seed used to initialize Dynesty's random generator.
        progress_label : str, optional
            Human-readable label shown in the progress display.

        Returns
        -------
        DynestyResult
            Posterior samples, log-weights, log-likelihoods, evidence values,
            evidence uncertainties, and parameter names.
        """
        if nlive < 1:
            raise ValueError(
                "nlive must be at least one."
            )

        if n_processes < 1:
            raise ValueError(
                "n_processes must be at least one."
            )

        if queue_size is not None and queue_size < 1:
            raise ValueError(
                "queue_size must be at least one."
            )

        if dlogz is not None and dlogz <= 0.0:
            raise ValueError(
                "dlogz must be positive."
            )

        if bootstrap is not None and bootstrap < 0:
            raise ValueError(
                "bootstrap must be non-negative."
            )

        random_state = (
            np.random.default_rng(seed)
            if seed is not None
            else None
        )

        if progress_label is not None:
            print()
            print("────────────────────────────────────────────────────")
            print(f"Running Dynesty fit: {progress_label}")
            print(
                f"ndim={self.ndim}, "
                f"nlive={nlive}, "
                f"sample={sample}, "
                f"bound={bound}"
            )
            print("────────────────────────────────────────────────────")

        sampler_kwargs = {
            "nlive": nlive,
            "sample": sample,
            "bound": bound,
        }

        if random_state is not None:
            sampler_kwargs["rstate"] = random_state

        if bootstrap is not None:
            sampler_kwargs["bootstrap"] = bootstrap

        run_kwargs = {
            "print_progress": print_progress,
        }

        if dlogz is not None:
            run_kwargs["dlogz"] = dlogz

        if print_progress and progress_label is not None:
            run_kwargs["print_func"] = DynestyMultilineProgress(
                progress_label
            )

        if n_processes > 1:
            resolved_queue_size = (
                n_processes
                if queue_size is None
                else queue_size
            )

            with Pool(
                processes=n_processes,
            ) as pool:
                sampler = dynesty.NestedSampler(
                    self.loglikelihood,
                    self.prior_transform,
                    ndim=self.ndim,
                    pool=pool,
                    queue_size=resolved_queue_size,
                    **sampler_kwargs,
                )

                sampler.run_nested(
                    **run_kwargs,
                )

                results = sampler.results

        else:
            sampler = dynesty.NestedSampler(
                self.loglikelihood,
                self.prior_transform,
                ndim=self.ndim,
                **sampler_kwargs,
            )

            sampler.run_nested(
                **run_kwargs,
            )

            results = sampler.results

        if progress_label is not None:
            total_calls = (
                int(
                    np.sum(
                        results.ncall
                    )
                )
                if hasattr(results, "ncall")
                else "unknown"
            )

            print("────────────────────────────────────────────────────")
            print(f"Completed Dynesty fit: {progress_label}")
            print(f"logZ={float(results.logz[-1]):.6g}")
            print(f"logZerr={float(results.logzerr[-1]):.6g}")
            print(f"ncall={total_calls}")
            print("────────────────────────────────────────────────────")

        return DynestyResult(
            samples=results.samples,
            logwt=results.logwt,
            logz=results.logz,
            logl=results.logl,
            logzerr=results.logzerr,
            parameter_names=self.parameter_names,
        )