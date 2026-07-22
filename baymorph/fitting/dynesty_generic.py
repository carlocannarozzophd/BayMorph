"""Generic Dynesty fitter."""

import sys
from multiprocessing import Pool

import dynesty
import numpy as np

try:
    from IPython.display import HTML, DisplayHandle
except ImportError:
    DisplayHandle = None
    HTML = None

from baymorph.likelihood import gaussian_log_likelihood
from baymorph.results import PosteriorResult
from baymorph.simulation.psf import convolve_with_psf


class DynestyResult(PosteriorResult):
    """Generic Dynesty posterior result."""


class DynestyMultilineProgress:
    """Live Dynesty progress display for terminals and Jupyter notebooks."""

    def __init__(
        self,
        label: str | None = None,
    ):
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
        self._n_lines = 9
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
    def _last(value):
        """Return the last finite-like value from a scalar or array."""
        arr = np.atleast_1d(value)

        if arr.size == 0:
            return np.nan

        return float(arr[-1])

    @staticmethod
    def _field(
        results,
        *names,
    ):
        """Return the first available field from a Dynesty results object."""
        for name in names:
            if hasattr(results, name):
                return getattr(results, name)

        return np.nan

    @staticmethod
    def _format_or_pending(
        value,
        fmt: str,
    ):
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
    ):
        """Update the live progress display from a Dynesty callback.

        Parameters
        ----------
        results : object
            Partial Dynesty results object supplied by the sampler.
        niter : int
            Current nested-sampling iteration.
        ncall : int
            Current number of likelihood calls.
        dlogz : float
            Target stopping threshold on remaining evidence.
        *args
            Additional positional arguments supplied by Dynesty.
        **kwargs
            Additional keyword arguments supplied by Dynesty.
        """
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

        eff = 100.0 * niter / ncall if ncall > 0 else np.nan

        lines = [
            "Dynesty fit",
            f"Model      : {self.label}",
            f"Iteration  : {niter}",
            f"Calls      : {ncall}",
            f"Efficiency : {self._format_or_pending(eff, '.3f')}%",
            f"logL       : {self._format_or_pending(self._last_logl, '.6g')}",
            f"logZ       : {self._format_or_pending(logz, '.6g')}",
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
                HTML(f"<pre>{html}</pre>"),
            )


class DynestyFitter:
    """Generic Dynesty fitter for arbitrary model parameters.

    Parameters
    ----------
    model_factory : callable
        Callable that receives model parameters as keyword arguments and
        returns an object with a ``render(shape=...)`` method.
    data : array-like
        Observed image data.
    noise : array-like
        Noise map with the same shape as ``data``.
    parameter_names : list of str, optional
        Names of fitted parameters. Required when ``parameters`` is None.
    prior_bounds : dict, optional
        Mapping from parameter name to ``(lower, upper)`` prior bounds.
        Required when ``parameters`` is None.
    parameters : object, optional
        Parameter container exposing ``names``, ``bounds``, and
        ``transform(u)``. If provided, it takes precedence over
        ``parameter_names`` and ``prior_bounds``.
    psf : array-like, optional
        Point-spread function used to convolve the rendered model image before
        likelihood evaluation. This is kept for backwards-compatible workflows;
        model factories may also return already-convolved models.
    mask : array-like of bool, optional
        Boolean mask identifying pixels excluded from the likelihood.
    """

    def __init__(
        self,
        model_factory,
        data,
        noise,
        parameter_names: list[str] | None = None,
        prior_bounds: dict[str, tuple[float, float]] | None = None,
        parameters=None,
        psf=None,
        mask=None,
    ):
        if parameters is not None:
            parameter_names = parameters.names
            prior_bounds = parameters.bounds

        if parameter_names is None:
            raise ValueError(
                "parameter_names must be provided when parameters is None."
            )

        if prior_bounds is None:
            raise ValueError(
                "prior_bounds must be provided when parameters is None."
            )

        self.model_factory = model_factory
        self.data = np.asarray(
            data,
            dtype=float,
        )
        self.noise = np.asarray(
            noise,
            dtype=float,
        )
        self.parameter_names = list(parameter_names)
        self.prior_bounds = dict(prior_bounds)
        self.parameters = parameters
        self.psf = psf
        self.mask = mask

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate data, noise, parameter names, and prior bounds.

        Raises
        ------
        ValueError
            If data and noise shapes differ, no parameters are provided, a
            parameter is missing prior bounds, or any prior has invalid bounds.
        """
        if self.data.shape != self.noise.shape:
            raise ValueError(
                "Data and noise must have the same shape."
            )

        if len(self.parameter_names) == 0:
            raise ValueError(
                "At least one parameter is required."
            )

        for name in self.parameter_names:
            if name not in self.prior_bounds:
                raise ValueError(
                    f"Missing prior bounds for parameter: {name}"
                )

            lower, upper = self.prior_bounds[name]

            if upper <= lower:
                raise ValueError(
                    f"Invalid prior bounds for parameter: {name}"
                )

    @property
    def ndim(self) -> int:
        """int : Number of fitted parameters."""
        return len(self.parameter_names)

    def prior_transform(
        self,
        u,
    ):
        """Transform unit-cube samples to physical parameters.

        Parameters
        ----------
        u : array-like
            Unit-cube sample with one value per fitted parameter.

        Returns
        -------
        ndarray
            Physical parameter vector in the order defined by
            ``parameter_names``.
        """
        if self.parameters is not None:
            return self.parameters.transform(u)

        theta = []

        for value, name in zip(
            u,
            self.parameter_names,
        ):
            lower, upper = self.prior_bounds[name]

            theta.append(
                lower
                + value
                * (
                    upper
                    - lower
                )
            )

        return np.asarray(theta)

    def theta_to_parameters(
        self,
        theta,
    ) -> dict[str, float]:
        """Convert a parameter vector to a parameter dictionary.

        Parameters
        ----------
        theta : array-like
            Physical parameter vector.

        Returns
        -------
        dict
            Mapping from parameter name to parameter value.
        """
        return {
            name: float(value)
            for name, value in zip(
                self.parameter_names,
                theta,
            )
        }

    def loglikelihood(
        self,
        theta,
    ):
        """Evaluate the Gaussian log-likelihood for one parameter vector.

        Parameters
        ----------
        theta : array-like
            Physical parameter vector.

        Returns
        -------
        float
            Gaussian log-likelihood value.
        """
        parameters = self.theta_to_parameters(
            theta,
        )

        model = self.model_factory(
            **parameters,
        )

        model_image = model.render(
            shape=self.data.shape,
        )

        if self.psf is not None:
            model_image = convolve_with_psf(
                model_image,
                self.psf,
            )

        return gaussian_log_likelihood(
            data=self.data,
            model=model_image,
            noise=self.noise,
            mask=self.mask,
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
            Number of bootstrap realizations used by Dynesty for bounding.
        print_progress : bool, default=True
            If True, display Dynesty progress information.
        n_processes : int, default=1
            Number of worker processes used for likelihood evaluations.
        queue_size : int, optional
            Number of queued likelihood evaluations when multiprocessing is
            enabled. If None, this defaults to ``n_processes``.
        seed : int, optional
            Random seed passed to Dynesty through a NumPy random generator.
        progress_label : str, optional
            Human-readable label shown in the custom progress display.

        Returns
        -------
        DynestyResult
            Posterior samples, log-weights, log-likelihoods, evidence values,
            evidence uncertainties, and fitted parameter names.
        """
        rstate = None

        if seed is not None:
            rstate = np.random.default_rng(seed)

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

        if rstate is not None:
            sampler_kwargs["rstate"] = rstate

        if bootstrap is not None:
            sampler_kwargs["bootstrap"] = bootstrap

        run_kwargs = {
            "print_progress": print_progress,
        }

        if dlogz is not None:
            run_kwargs["dlogz"] = dlogz

        if print_progress and progress_label is not None:
            run_kwargs["print_func"] = DynestyMultilineProgress(
                progress_label,
            )

        if n_processes > 1:
            if queue_size is None:
                queue_size = n_processes

            with Pool(
                processes=n_processes,
            ) as pool:
                sampler = dynesty.NestedSampler(
                    self.loglikelihood,
                    self.prior_transform,
                    ndim=self.ndim,
                    pool=pool,
                    queue_size=queue_size,
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
            print("────────────────────────────────────────────────────")
            print(f"Completed Dynesty fit: {progress_label}")
            print(f"logZ={float(results.logz[-1]):.6g}")
            print(f"logZerr={float(results.logzerr[-1]):.6g}")
            print(
                "ncall="
                f"{int(np.sum(results.ncall)) if hasattr(results, 'ncall') else 'unknown'}"
            )
            print("────────────────────────────────────────────────────")

        return DynestyResult(
            samples=results.samples,
            logwt=results.logwt,
            logz=results.logz,
            logl=results.logl,
            logzerr=results.logzerr,
            parameter_names=self.parameter_names,
        )