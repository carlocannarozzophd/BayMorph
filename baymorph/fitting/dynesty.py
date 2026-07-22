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

from baymorph.fitting.parameters import ParameterSet
from baymorph.likelihood import ImageLikelihood
from baymorph.results import PosteriorResult
from baymorph.simulation.psf import convolve_with_psf


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
        model_factory,
        parameters: ParameterSet,
        likelihood: ImageLikelihood,
        psf=None,
    ) -> None:
        if not callable(model_factory):
            raise TypeError(
                "model_factory must be callable."
            )

        if not isinstance(
            parameters,
            ParameterSet,
        ):
            raise TypeError(
                "parameters must be a ParameterSet instance."
            )

        if not isinstance(
            likelihood,
            ImageLikelihood,
        ):
            raise TypeError(
                "likelihood must be an ImageLikelihood instance."
            )

        if parameters.ndim < 1:
            raise ValueError(
                "ParameterSet must contain at least one parameter."
            )

        if len(set(parameters.names)) != parameters.ndim:
            raise ValueError(
                "Parameter names must be unique."
            )

        self.model_factory = model_factory
        self.parameters = parameters
        self.likelihood = likelihood
        self.psf = psf

    @property
    def ndim(self) -> int:
        """Number of fitted parameters."""
        return self.parameters.ndim

    @property
    def parameter_names(self) -> list[str]:
        """Parameter names in sampling order."""
        return self.parameters.names

    @property
    def image_shape(self) -> tuple[int, ...]:
        """Shape expected for rendered model images."""
        return self.likelihood.image_shape

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
            self.parameters.transform(
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

    def _theta_to_kwargs(
        self,
        theta,
    ) -> dict[str, float]:
        """Convert a parameter vector to model-factory keyword arguments."""
        theta_array = np.asarray(
            theta,
            dtype=float,
        )

        if theta_array.shape != (self.ndim,):
            raise ValueError(
                "Parameter vector must contain one value per parameter."
            )

        if not np.all(np.isfinite(theta_array)):
            raise ValueError(
                "Parameter values must be finite."
            )

        return {
            name: float(value)
            for name, value in zip(
                self.parameter_names,
                theta_array,
                strict=True,
            )
        }

    def render_model(
        self,
        theta,
    ) -> np.ndarray:
        """Render the model image for one physical parameter vector.

        Parameters
        ----------
        theta : array-like
            Physical parameter vector in ``ParameterSet`` order.

        Returns
        -------
        ndarray
            Rendered model image, including optional PSF convolution.

        Raises
        ------
        TypeError
            If the model returned by ``model_factory`` does not expose a
            callable ``render`` method.
        ValueError
            If the rendered image has an incompatible shape or contains
            non-finite values.
        """
        model_kwargs = self._theta_to_kwargs(
            theta
        )

        model = self.model_factory(
            **model_kwargs,
        )

        render = getattr(
            model,
            "render",
            None,
        )

        if not callable(render):
            raise TypeError(
                "model_factory must return an object with a callable "
                "render(shape=...) method."
            )

        model_image = np.asarray(
            render(
                shape=self.image_shape,
            ),
            dtype=float,
        )

        if self.psf is not None:
            model_image = np.asarray(
                convolve_with_psf(
                    model_image,
                    self.psf,
                ),
                dtype=float,
            )

        if model_image.shape != self.image_shape:
            raise ValueError(
                "Rendered model must have the same shape as the likelihood "
                "data."
            )

        if not np.all(np.isfinite(model_image)):
            raise ValueError(
                "Rendered model must contain only finite values."
            )

        return model_image

    def loglikelihood(
        self,
        theta,
    ) -> float:
        """Evaluate the configured likelihood for one parameter vector.

        Parameters
        ----------
        theta : array-like
            Physical parameter vector in ``ParameterSet`` order.

        Returns
        -------
        float
            Log-likelihood value.
        """
        model_image = self.render_model(
            theta
        )

        return float(
            self.likelihood(
                model_image
            )
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