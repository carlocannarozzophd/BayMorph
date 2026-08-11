"""Repeated-noise multi-observation recovery campaigns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import yaml

from baymorph.pipeline.image_preparation import (
    prepare_real_image_from_config,
)
from baymorph.pipeline.multiband_fit import (
    run_multiband_fit_from_config,
)
from baymorph.validation.baymorph_adapter import (
    recovery_fit_from_inference_result,
)
from baymorph.validation.multi_observation_truth import (
    MultiObservationGroundTruth,
    ObservationTruth,
)
from baymorph.validation.multi_observation_workspace import (
    MultiObservationValidationWorkspace,
)
from baymorph.validation.recovery import RecoveryResult


@dataclass(frozen=True)
class MultiObservationNoiseCampaign:
    """Repeated-noise recovery experiment for one fixed physical target.

    The intrinsic morphology, WCS geometry, PSFs, pixel scales, and noise
    amplitudes remain fixed. Only observation-specific random seeds vary
    between cases.
    """

    root: str | Path
    truth: MultiObservationGroundTruth
    sampler: dict
    n_cases: int

    def __post_init__(self) -> None:
        root = Path(self.root)

        if self.n_cases <= 0:
            raise ValueError(
                "Campaign size must be positive."
            )

        object.__setattr__(
            self,
            "root",
            root,
        )

    def run(self) -> list[RecoveryResult]:
        """Execute all campaign cases sequentially."""
        results = []

        for index in range(
            self.n_cases
        ):
            result = self.run_case(
                index
            )
            results.append(
                result
            )

        return results

    def run_case(
        self,
        index: int,
    ) -> RecoveryResult:
        """Execute one deterministic noise realization."""
        if (
            index < 0
            or index >= self.n_cases
        ):
            raise IndexError(
                "Campaign case index out of range."
            )

        case_id = (
            f"noise-{index:03d}"
        )

        case_truth = (
            self._truth_for_case(
                index
            )
        )

        workspace = (
            MultiObservationValidationWorkspace(
                self.root
            )
        )

        joint_path = workspace.write_case(
            case_id,
            case_truth,
            sampler=self.sampler,
        )

        self._prepare_observations(
            joint_path
        )

        start = perf_counter()

        inference_result = (
            run_multiband_fit_from_config(
                joint_path
            )
        )

        runtime_seconds = (
            perf_counter()
            - start
        )

        inference_path = (
            joint_path.parent
            / "joint_inference_result.npz"
        )

        inference_result.save(
            inference_path
        )

        recovery_fit = (
            recovery_fit_from_inference_result(
                inference_result,
                metadata={
                    "campaign": (
                        "multi_observation_noise"
                    ),
                    "case_index": index,
                    "runtime_seconds": (
                        runtime_seconds
                    ),
                },
            )
        )

        likelihood_evaluations = (
            inference_result.diagnostics.get(
                "calls"
            )
        )

        if likelihood_evaluations is not None:
            likelihood_evaluations = int(
                likelihood_evaluations
            )

        result = RecoveryResult(
            case_id=case_id,
            truth=dict(
                case_truth.parameters
            ),
            estimates=dict(
                recovery_fit.estimates
            ),
            intervals=(
                None
                if recovery_fit.intervals
                is None
                else dict(
                    recovery_fit.intervals
                )
            ),
            log_evidence=(
                recovery_fit.log_evidence
            ),
            log_evidence_error=(
                recovery_fit
                .log_evidence_error
            ),
            runtime_seconds=(
                runtime_seconds
            ),
            likelihood_evaluations=(
                likelihood_evaluations
            ),
            metadata=dict(
                recovery_fit.metadata
            ),
        )

        return result

    def _truth_for_case(
        self,
        index: int,
    ) -> MultiObservationGroundTruth:
        """Return truth with deterministic per-observation noise seeds."""
        observations = {}

        for offset, (
            name,
            observation,
        ) in enumerate(
            self.truth.observations.items()
        ):
            metadata = dict(
                observation.metadata
            )

            metadata["seed"] = (
                10000
                + 100 * index
                + offset
            )

            observations[
                name
            ] = ObservationTruth(
                image_shape=(
                    observation.image_shape
                ),
                pixel_scale=(
                    observation.pixel_scale
                ),
                psf_metadata=dict(
                    observation.psf_metadata
                ),
                noise_metadata=dict(
                    observation.noise_metadata
                ),
                metadata=metadata,
            )

        return MultiObservationGroundTruth(
            model_name=(
                self.truth.model_name
            ),
            parameters=dict(
                self.truth.parameters
            ),
            observations=observations,
            random_seed=(
                20000
                + index
            ),
            metadata=dict(
                self.truth.metadata
            ),
        )

    @staticmethod
    def _prepare_observations(
        joint_path: Path,
    ) -> None:
        """Run production image preparation for every observation."""
        with joint_path.open() as stream:
            joint = yaml.safe_load(
                stream
            )

        for observation in (
            joint["observations"].values()
        ):
            config_path = (
                joint_path.parent
                / observation[
                    "config_path"
                ]
            )

            prepare_real_image_from_config(
                config_path,
                save=True,
            )
