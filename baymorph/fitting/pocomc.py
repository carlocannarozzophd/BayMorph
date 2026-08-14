"""pocoMC fitting backend for BayMorph."""

from __future__ import annotations


import numpy as np
from scipy.stats import (
    beta as beta_distribution,
    norm,
    uniform,
)

from baymorph.fitting.image_problem import ImageInferenceProblem
from baymorph.fitting.parameters import ParameterSet
from baymorph.fitting.priors import (
    BetaPrior,
    NormalPrior,
    UniformPrior,
)
from baymorph.fitting.problem import InferenceProblem
from baymorph.likelihood import ImageLikelihood
from baymorph.results import (
    EvidenceEstimate,
    InferenceResult,
    PosteriorSamples,
)


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
        *,
        init_kwargs=None,
        run_kwargs=None,
        progress_label: str | None = None,
    ) -> InferenceResult:
        """Run pocoMC using native constructor and run keyword arguments.

        Parameters
        ----------
        init_kwargs : mapping, optional
            Native keyword arguments forwarded unchanged to
            :class:`pocomc.Sampler`.
        run_kwargs : mapping, optional
            Native keyword arguments forwarded unchanged to
            ``pocomc.Sampler.run``.
        progress_label : str, optional
            Human-readable BayMorph label displayed around the fit.

        Returns
        -------
        InferenceResult
            Canonical backend-neutral BayMorph inference result.

        Notes
        -----
        BayMorph owns the prior, likelihood, and problem dimensionality.
        These constructor arguments therefore cannot be supplied through
        ``init_kwargs``.
        """
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
                "pocoMC constructor argument(s) owned by BayMorph "
                f"cannot be configured explicitly: {names}."
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

        if progress_label is not None:
            print()
            print(
                "────────────────────────────────────────────────────"
            )
            print(
                f"Running pocoMC fit: {progress_label}"
            )
            print(
                f"ndim={self.ndim}"
            )
            print(
                "────────────────────────────────────────────────────"
            )

        sampler = pc.Sampler(
            prior=prior,
            likelihood=self.loglikelihood,
            n_dim=self.ndim,
            **init_config,
        )

        sampler.run(
            **run_config
        )

        samples, weights, logl, logp = (
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
        logp = np.asarray(
            logp,
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

        weights = (
            weights
            / normalization
        )

        logz_value = float(
            logz
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

        posterior = PosteriorSamples(
            samples=samples,
            weights=weights,
            parameter_names=self.parameter_names,
            log_likelihood=logl,
            log_prior=logp,
        )

        sampler_configuration = {
            "name": "pocomc",
            "init": init_config,
            "run": run_config,
        }

        evidence = EvidenceEstimate(
            logz=logz_value,
            uncertainty=float(
                logzerr
            ),
            method="pocomc",
            uncertainty_kind="importance_sampling_internal",
            provenance={
                "backend": "pocomc",
                "sampler": sampler_configuration,
            },
        )

        return InferenceResult(
            posterior=posterior,
            evidence=evidence,
            diagnostics=diagnostics,
            provenance={
                "backend": "pocomc",
                "sampler": sampler_configuration,
            },
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
            BetaPrior,
        ):
            distributions.append(
                beta_distribution(
                    a=prior.alpha,
                    b=prior.beta,
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
            "pocoMC currently supports BayMorph UniformPrior, "
            "BetaPrior, and NormalPrior sampled parameters. "
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
