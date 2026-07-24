"""Native geometry-aware nested-sampling orchestration."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from baymorph.results.posterior import PosteriorResult

from baymorph.fitting.parameters import ParameterSet

from ..geometry_model import GeometryModel
from ..proposals import (
    PriorProposalEngine,
    ProposalEngine,
)
from .config import NestedSamplerConfig
from .evidence import (
    EvidenceAccumulator,
    LogSpaceEvidenceAccumulator,
)
from .iteration import IterationResult
from .live_points import LivePoints
from .run import NestedRun
from .shrinkage import (
    ExpectedShrinkageModel,
    ShrinkageModel,
)
from .state import DeadPoint
from .termination import (
    AnyTerminationCriterion,
    EvidenceTerminationCriterion,
    MaximumIterationTerminationCriterion,
    MaximumLikelihoodCallTerminationCriterion,
    TerminationCriterion,
    TerminationDecision,
)


class GeometryNestedSampler:
    """Experimental native geometry-aware nested sampler.

    Parameters
    ----------
    geometry : GeometryModel
        Geometry definition matching the fitted parameter ordering.
    parameter_set : ParameterSet
        Prior definitions and unit-cube transformation.
    likelihood : callable
        Callable receiving physical parameters and returning log-likelihood.
    proposal_engine : ProposalEngine, optional
        Constrained replacement proposal engine.
    shrinkage_model : ShrinkageModel, optional
        Prior-volume shrinkage strategy.
    evidence_accumulator : EvidenceAccumulator, optional
        Evidence quadrature strategy.
    config : NestedSamplerConfig, optional
        Immutable sampler configuration.
    nlive : int, optional
        Backward-compatible live-point count used when ``config`` is omitted.
    seed : int or None, optional
        Backward-compatible random seed used when ``config`` is omitted.
    """

    def __init__(
        self,
        *,
        geometry: GeometryModel,
        parameter_set: ParameterSet,
        likelihood: Callable[[np.ndarray], float],
        proposal_engine: ProposalEngine | None = None,
        shrinkage_model: ShrinkageModel | None = None,
        evidence_accumulator: EvidenceAccumulator | None = None,
        termination_criterion: TerminationCriterion | None = None,
        config: NestedSamplerConfig | None = None,
        nlive: int = 500,
        seed: int | None = None,
    ) -> None:
        if not isinstance(geometry, GeometryModel):
            raise TypeError(
                "geometry must be a GeometryModel instance."
            )

        if not isinstance(parameter_set, ParameterSet):
            raise TypeError(
                "parameter_set must be a ParameterSet instance."
            )

        if not callable(likelihood):
            raise TypeError(
                "likelihood must be callable."
            )

        geometry.validate_parameter_names(
            parameter_set.names
        )

        if config is not None and (
            nlive != 500 or seed is not None
        ):
            raise ValueError(
                "Pass either config or the legacy nlive/seed arguments, "
                "not both."
            )

        if config is None:
            config = NestedSamplerConfig(
                nlive=nlive,
                seed=seed,
            )

        if not isinstance(config, NestedSamplerConfig):
            raise TypeError(
                "config must be a NestedSamplerConfig instance."
            )

        if proposal_engine is None:
            proposal_engine = PriorProposalEngine()

        if not isinstance(proposal_engine, ProposalEngine):
            raise TypeError(
                "proposal_engine must be a ProposalEngine instance."
            )

        if shrinkage_model is None:
            shrinkage_model = ExpectedShrinkageModel()

        if not isinstance(shrinkage_model, ShrinkageModel):
            raise TypeError(
                "shrinkage_model must be a ShrinkageModel instance."
            )

        if evidence_accumulator is None:
            evidence_accumulator = LogSpaceEvidenceAccumulator()

        if not isinstance(
            evidence_accumulator,
            EvidenceAccumulator,
        ):
            raise TypeError(
                "evidence_accumulator must be an "
                "EvidenceAccumulator instance."
            )

        self.geometry = geometry
        self.parameter_set = parameter_set
        self.likelihood = likelihood
        self.config = config
        self.proposal_engine = proposal_engine
        self.shrinkage_model = shrinkage_model
        self.evidence_accumulator = evidence_accumulator

        if termination_criterion is None:
            termination_criterion = self._build_default_termination_criterion(
                config
            )

        if not isinstance(
            termination_criterion,
            TerminationCriterion,
        ):
            raise TypeError(
                "termination_criterion must be a "
                "TerminationCriterion instance."
            )

        self.termination_criterion = termination_criterion

        self.rng = np.random.default_rng(
            self.config.seed
        )

        live_points = LivePoints(
            self.geometry
        )

        self.run_state = NestedRun(
            config=self.config,
            live_points=live_points,
        )

        # Backward-compatible public aliases used by the current tests and API.
        self.live_points = self.run_state.live_points
        self.state = self.run_state

    @property
    def nlive(self) -> int:
        """Configured number of live points."""
        return self.config.nlive

    @property
    def worst(self):
        """Current worst live point."""
        return self.live_points.worst

    @property
    def best(self):
        """Current best live point."""
        return self.live_points.best

    def initialize(self) -> None:
        """Draw and evaluate the initial live-point population."""
        if self.run_state.initialized:
            raise RuntimeError(
                "GeometryNestedSampler is already initialized."
            )

        for _ in range(self.nlive):
            unit_values = self.rng.random(
                self.parameter_set.ndim
            )
            physical_values = self.parameter_set.transform(
                unit_values
            )

            # Validate transform-domain compatibility before likelihood work.
            self.geometry.to_proposal_space(
                physical_values
            )

            logl = float(
                self.likelihood(physical_values)
            )

            if np.isnan(logl):
                raise ValueError(
                    "Likelihood returned NaN during initialization."
                )

            self.live_points.append(
                physical_values,
                logl,
            )

        self.live_points.sort()

        self.run_state.mark_initialized(
            ncall=self.nlive
        )

    def step(self) -> IterationResult:
        """Execute and commit one complete nested-sampling iteration.

        The run remains unchanged if proposal generation or result
        construction fails.
        """
        plan = self.run_state.build_iteration_plan(
            shrinkage_model=self.shrinkage_model,
            rng=self.rng,
        )

        proposal = self.proposal_engine.propose(
            live_points=self.live_points,
            logl_min=plan.logl_min,
            parameter_set=self.parameter_set,
            likelihood=self.likelihood,
            rng=self.rng,
        )

        evidence_update = self.evidence_accumulator.accumulate(
            current_logz=self.run_state.logz,
            log_weight=plan.log_weight,
            logl=plan.worst_point.logl,
        )

        dead_point = DeadPoint(
            values=plan.worst_point.values,
            logl=plan.worst_point.logl,
            logx_previous=plan.logx_previous,
            logx_current=plan.logx_current,
            log_weight=plan.log_weight,
            log_weighted_likelihood=(
                evidence_update.contribution
            ),
            iteration=plan.iteration,
        )

        result = IterationResult(
            plan=plan,
            proposal=proposal,
            dead_point=dead_point,
            new_logz=evidence_update.new_logz,
            diagnostics={
                "logz_contribution": (
                    evidence_update.contribution
                ),
                **proposal.diagnostics,
            },
        )

        self.run_state.commit(result)

        return result

    def termination_decision(self) -> TerminationDecision:
        """Evaluate whether active nested sampling should stop."""
        if not self.run_state.initialized:
            raise RuntimeError(
                "GeometryNestedSampler must be initialized before "
                "termination can be evaluated."
            )

        return self.termination_criterion.evaluate(
            iteration=self.run_state.iteration,
            ncall=self.run_state.ncall,
            logz=self.run_state.logz,
            logx=self.run_state.log_prior_volume,
            live_logl=self.live_points.logl,
        )

    def run_active_phase(self) -> tuple[IterationResult, ...]:
        """Run constrained iterations until termination.

        Returns
        -------
        tuple of IterationResult
            Committed active-iteration results in execution order.

        Notes
        -----
        This method performs only the active constrained phase. It does not
        yet commit the final live-point evidence contribution.
        """
        if not self.run_state.initialized:
            self.initialize()

        if self.run_state.terminated:
            raise RuntimeError(
                "GeometryNestedSampler has already terminated."
            )

        results: list[IterationResult] = []

        while True:
            decision = self.termination_decision()

            if decision.should_stop:
                self.run_state.mark_terminated()
                return tuple(results)

            results.append(
                self.step()
            )

    def run(self) -> "PosteriorResult":
        """Execute the complete native nested-sampling workflow.

        The method initializes the live-point population when necessary,
        continues the active constrained phase until termination, commits
        the remaining live-point evidence contribution, and returns a
        backend-neutral posterior result.

        An already initialized active run is resumed from its current
        state. A terminated but not yet finalized run proceeds directly
        to finalization.

        Returns
        -------
        PosteriorResult
            Posterior samples, weights, likelihoods, and cumulative
            evidence produced by the completed run.

        Raises
        ------
        RuntimeError
            If the sampler has already been finalized.
        """
        if self.run_state.finalized:
            raise RuntimeError(
                "GeometryNestedSampler has already been finalized."
            )

        if not self.run_state.terminated:
            self.run_active_phase()

        return self.finalize()

    def finalize(self) -> "PosteriorResult":
        """Commit the remaining live points and build the posterior.

        Each final live point receives an equal share of the remaining
        prior mass. If the remaining logarithmic prior volume is
        ``logx_final``, its logarithmic quadrature weight is therefore

        ``logx_final - log(nlive)``.

        Returns
        -------
        PosteriorResult
            Backend-neutral posterior and evidence result.

        Raises
        ------
        RuntimeError
            If the sampler has not terminated or has already finalized.
        """
        from baymorph.results.posterior import PosteriorResult

        if not self.run_state.initialized:
            raise RuntimeError(
                "GeometryNestedSampler must be initialized before "
                "finalization."
            )

        if not self.run_state.terminated:
            raise RuntimeError(
                "GeometryNestedSampler must terminate before finalization."
            )

        if self.run_state.finalized:
            raise RuntimeError(
                "GeometryNestedSampler has already been finalized."
            )

        dead_points = tuple(
            self.run_state.dead_points
        )
        live_points = tuple(
            sorted(
                self.live_points,
                key=lambda point: point.logl,
            )
        )

        dead_samples = [
            point.values
            for point in dead_points
        ]
        dead_logl = [
            point.logl
            for point in dead_points
        ]
        dead_logwt = [
            point.log_weighted_likelihood
            for point in dead_points
        ]

        final_log_weight = (
            self.run_state.log_prior_volume
            - np.log(self.nlive)
        )

        live_samples = [
            point.values
            for point in live_points
        ]
        live_logl = [
            point.logl
            for point in live_points
        ]
        live_logwt = [
            final_log_weight + point.logl
            for point in live_points
        ]

        samples = np.stack(
            dead_samples + live_samples,
            axis=0,
        )
        logl = np.asarray(
            dead_logl + live_logl,
            dtype=float,
        )
        logwt = np.asarray(
            dead_logwt + live_logwt,
            dtype=float,
        )
        logz = np.logaddexp.accumulate(
            logwt
        )

        self.run_state.mark_finalized(
            logz=float(logz[-1])
        )

        return PosteriorResult(
            samples=samples,
            logwt=logwt,
            logz=logz,
            parameter_names=list(
                self.parameter_set.names
            ),
            logl=logl,
        )

    @staticmethod
    def _build_default_termination_criterion(
        config: NestedSamplerConfig,
    ) -> TerminationCriterion:
        """Build the termination policy implied by the configuration."""
        criteria: list[TerminationCriterion] = [
            EvidenceTerminationCriterion(
                dlogz=config.dlogz,
            )
        ]

        if config.max_iterations is not None:
            criteria.append(
                MaximumIterationTerminationCriterion(
                    max_iterations=config.max_iterations,
                )
            )

        if config.max_likelihood_calls is not None:
            criteria.append(
                MaximumLikelihoodCallTerminationCriterion(
                    max_calls=config.max_likelihood_calls,
                )
            )

        return AnyTerminationCriterion(
            *criteria
        )

    def replace_worst(self):
        """Execute one iteration using the legacy return contract.

        Returns
        -------
        removed : LivePoint
            Live point removed during the iteration.
        proposal : ProposalResult
            Accepted constrained replacement proposal.
        """
        result = self.step()

        return (
            result.plan.worst_point,
            result.proposal,
        )
