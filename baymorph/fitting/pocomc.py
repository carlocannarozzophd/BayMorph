"""pocoMC fitting backend for BayMorph."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import norm, uniform

from baymorph.fitting.image_problem import ImageInferenceProblem
from baymorph.fitting.parameters import ParameterSet
from baymorph.fitting.priors import NormalPrior, UniformPrior
from baymorph.fitting.problem import InferenceProblem
from baymorph.likelihood import ImageLikelihood
from baymorph.results import PosteriorResult


@dataclass
class PocoMCResult(PosteriorResult):
    """Posterior result produced by the pocoMC fitting backend."""

    diagnostics: dict[str, Any] = field(default_factory=dict)


class PocoMCFitter:
    """Fit a BayMorph inference problem with pocoMC.

    Parameters
    ----------
    problem : InferenceProblem, optional
        Backend-neutral inference problem.
    model_factory : callable, optional
        Legacy single-image model factory.
    parameters : ParameterSet, optional
        Legacy single-image fitted parameters.
    likelihood : ImageLikelihood, optional
        Legacy single-image likelihood.
    config : mapping, optional
        pocoMC-specific sampler configuration.

    Notes
    -----
    The backend intentionally reuses BayMorph's existing inference problem,
    likelihood, parameter, and posterior-result infrastructure.

    Dynesty-specific runtime arguments accepted by :meth:`run` are retained
    temporarily so that the existing pipeline can dispatch either backend
    without backend-specific branching.
    """

    def __init__(
        self,
        model_factory=None,
        parameters: ParameterSet | None = None,
        likelihood: ImageLikelihood | None = None,
        *,
        problem: InferenceProblem | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        if problem is not None:
            if any(
                value is not None
                for value in (
                    model_factory,
                    parameters,
                    likelihood,
                )
            ):
                raise ValueError(
                    "problem cannot be combined with model_factory, "
                    "parameters, or likelihood."
                )

            if not isinstance(problem, InferenceProblem):
                raise TypeError(
                    "problem must be an InferenceProblem instance."
                )

            self.problem = problem

        else:
            self.problem = ImageInferenceProblem(
                model_factory=model_factory,
                parameters=parameters,
                likelihood=likelihood,
            )

        self.config = dict(config or {})

    @property
    def ndim(self) -> int:
        """Return the number of sampled parameters."""
        return self.problem.ndim

    @property
    def parameter_names(self) -> list[str]:
        """Return parameter names in sampling order."""
        return self.problem.parameter_names

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
        nlive: int | None = None,
        sample: str | None = None,
        bound: str | None = None,
        dlogz: float | None = None,
        bootstrap: int | None = None,
        print_progress: bool = True,
        n_processes: int = 1,
        queue_size: int | None = None,
        seed: int | None = None,
        progress_label: str | None = None,
    ) -> PocoMCResult:
        """Run pocoMC and return a BayMorph posterior result.

        Dynesty-specific arguments are accepted for compatibility with the
        current BayMorph pipeline but are not translated into pocoMC control
        parameters.

        pocoMC-specific settings are read from ``config``.
        """
        del nlive
        del sample
        del bound
        del dlogz
        del bootstrap
        del queue_size

        if n_processes < 1:
            raise ValueError(
                "n_processes must be at least one."
            )

        if bool(
            self.config.get(
                "vectorize",
                False,
            )
        ):
            raise NotImplementedError(
                "Vectorized pocoMC likelihood evaluation is not yet "
                "supported by BayMorph."
            )

        try:
            import pocomc as pc
        except ImportError as exc:
            raise ImportError(
                "pocoMC support requires the optional pocomc dependency. "
                'Install it with `pip install "baymorph[pocomc]"`.'
            ) from exc

        parameter_set = _parameter_set_from_problem(
            self.problem
        )

        prior = _build_pocomc_prior(
            pc,
            parameter_set,
        )

        n_effective = int(
            self.config.get(
                "n_effective",
                512,
            )
        )
        n_active = int(
            self.config.get(
                "n_active",
                256,
            )
        )

        if n_effective < 2:
            raise ValueError(
                "n_effective must be at least two."
            )

        if n_active < 1:
            raise ValueError(
                "n_active must be at least one."
            )

        if n_active >= n_effective:
            raise ValueError(
                "n_active must be smaller than n_effective."
            )

        sampler_kwargs: dict[str, Any] = {
            "prior": prior,
            "likelihood": self.loglikelihood,
            "n_dim": self.ndim,
            "n_effective": n_effective,
            "n_active": n_active,
            "flow": self.config.get(
                "flow",
                "nsf6",
            ),
            "precondition": bool(
                self.config.get(
                    "precondition",
                    True,
                )
            ),
            "dynamic": bool(
                self.config.get(
                    "dynamic",
                    True,
                )
            ),
            "metric": self.config.get(
                "metric",
                "ess",
            ),
            "sample": self.config.get(
                "mcmc_sampler",
                "tpcn",
            ),
            "random_state": seed,
        }

        optional_integer_settings = (
            "train_frequency",
            "n_prior",
            "n_steps",
            "n_max_steps",
        )

        for name in optional_integer_settings:
            value = self.config.get(name)

            if value is not None:
                sampler_kwargs[name] = int(value)

        train_config = self.config.get(
            "train_config"
        )

        if train_config is not None:
            sampler_kwargs["train_config"] = dict(
                train_config
            )

        if n_processes > 1:
            sampler_kwargs["pool"] = n_processes

        if progress_label is not None:
            print()
            print(
                "────────────────────────────────────────────────────"
            )
            print(
                f"Running pocoMC fit: {progress_label}"
            )
            print(
                f"ndim={self.ndim}, "
                f"n_effective={n_effective}, "
                f"n_active={n_active}, "
                f"flow={sampler_kwargs['flow']}"
            )
            print(
                "────────────────────────────────────────────────────"
            )

        sampler = pc.Sampler(
            **sampler_kwargs
        )

        sampler.run(
            n_total=int(
                self.config.get(
                    "n_total",
                    4096,
                )
            ),
            n_evidence=int(
                self.config.get(
                    "n_evidence",
                    4096,
                )
            ),
            progress=print_progress,
        )

        samples, weights, logl, _logp = (
            sampler.posterior()
        )

        logz, logzerr = sampler.evidence()

        samples = np.asarray(
            samples,
            dtype=float,
        )
        weights = np.asarray(
            weights,
            dtype=float,
        )
        logl = np.asarray(
            logl,
            dtype=float,
        )

        normalization = float(
            np.sum(weights)
        )

        if (
            not np.isfinite(normalization)
            or normalization <= 0.0
        ):
            raise ValueError(
                "pocoMC returned posterior weights with an invalid "
                "normalization."
            )

        weights = weights / normalization

        logz_value = float(logz)

        with np.errstate(
            divide="ignore",
        ):
            logwt = (
                np.log(weights)
                + logz_value
            )

        native_results = dict(
            sampler.results
        )

        diagnostics = {
            "backend": "pocomc",
            "calls": _last_finite(
                native_results.get(
                    "calls"
                )
            ),
            "efficiency": _last_finite(
                native_results.get(
                    "efficiency"
                )
            ),
            "ess": _last_finite(
                native_results.get(
                    "ess"
                )
            ),
            "accept": _last_finite(
                native_results.get(
                    "accept"
                )
            ),
            "beta": _last_finite(
                native_results.get(
                    "beta"
                )
            ),
        }

        if progress_label is not None:
            print(
                "────────────────────────────────────────────────────"
            )
            print(
                f"Completed pocoMC fit: {progress_label}"
            )
            print(
                f"logZ={logz_value:.6g}"
            )
            print(
                f"logZerr={float(logzerr):.6g}"
            )

            if diagnostics["calls"] is not None:
                print(
                    "ncall="
                    f"{int(diagnostics['calls'])}"
                )

            print(
                "────────────────────────────────────────────────────"
            )

        return PocoMCResult(
            samples=samples,
            logwt=logwt,
            logz=np.asarray(
                [logz_value],
                dtype=float,
            ),
            logl=logl,
            logzerr=np.asarray(
                [float(logzerr)],
                dtype=float,
            ),
            parameter_names=self.parameter_names,
            diagnostics=diagnostics,
        )


def _parameter_set_from_problem(
    problem: InferenceProblem,
) -> ParameterSet:
    """Return the sampled ParameterSet owned by an inference problem."""
    parameters = getattr(
        problem,
        "parameters",
        None,
    )

    if isinstance(
        parameters,
        ParameterSet,
    ):
        return parameters

    parameter_layout = getattr(
        problem,
        "parameter_layout",
        None,
    )

    parameter_set = getattr(
        parameter_layout,
        "parameter_set",
        None,
    )

    if isinstance(
        parameter_set,
        ParameterSet,
    ):
        return parameter_set

    raise TypeError(
        "pocoMC currently requires an InferenceProblem backed by a "
        "BayMorph ParameterSet."
    )


def _build_pocomc_prior(
    pc,
    parameters: ParameterSet,
):
    """Translate BayMorph independent priors into a pocoMC Prior."""
    distributions = []

    for parameter in parameters.parameters:
        prior = parameter.prior

        if isinstance(
            prior,
            UniformPrior,
        ):
            distributions.append(
                uniform(
                    loc=prior.lower,
                    scale=(
                        prior.upper
                        - prior.lower
                    ),
                )
            )
            continue

        if isinstance(
            prior,
            NormalPrior,
        ):
            distributions.append(
                norm(
                    loc=prior.mean,
                    scale=prior.sigma,
                )
            )
            continue

        raise TypeError(
            "pocoMC currently supports BayMorph UniformPrior and "
            "NormalPrior sampled parameters. "
            f"Parameter '{parameter.name}' uses "
            f"'{type(prior).__name__}'."
        )

    return pc.Prior(
        distributions
    )


def _last_finite(
    value,
) -> float | None:
    """Return the final finite scalar from an array-like result field."""
    if value is None:
        return None

    array = np.asarray(
        value,
        dtype=float,
    ).reshape(-1)

    if array.size == 0:
        return None

    finite = array[
        np.isfinite(array)
    ]

    if finite.size == 0:
        return None

    return float(
        finite[-1]
    )
