"""Repeated-noise multi-observation recovery campaigns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import yaml

from baymorph.pipeline.image_preparation import (
    prepare_real_image_from_config,
)
from baymorph.pipeline.multi_observation_fit import (
    run_multi_observation_inference_from_config,
)
from baymorph.results import InferenceResult
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

    def run(
        self,
        *,
        resume: bool = True,
        continue_on_failure: bool = True,
    ) -> list[RecoveryResult]:
        """Execute the campaign, optionally reusing completed cases.

        Parameters
        ----------
        resume : bool, default=True
            Load existing inference results instead of rerunning completed
            cases.
        continue_on_failure : bool, default=True
            Record failed cases and continue with the remaining campaign.

        Returns
        -------
        list of RecoveryResult
            Successfully completed recovery results.
        """
        results = []

        for index in range(
            self.n_cases
        ):
            if (
                resume
                and self._result_path(
                    index
                ).exists()
            ):
                print(
                    f"Reusing completed "
                    f"{self._case_id(index)}"
                )

                results.append(
                    self.load_case(
                        index
                    )
                )
                continue

            try:
                result = self.run_case(
                    index
                )
            except Exception:
                if not continue_on_failure:
                    raise

                print(
                    f"Recorded failure for "
                    f"{self._case_id(index)}"
                )
                continue

            results.append(
                result
            )

        return results

    def run_case(
        self,
        index: int,
    ) -> RecoveryResult:
        """Execute one deterministic noise realization."""
        self._validate_index(
            index
        )

        case_id = self._case_id(
            index
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

        try:
            self._prepare_observations(
                joint_path
            )
        except Exception as exc:
            self._write_failure(
                index=index,
                stage="preparation",
                exc=exc,
            )
            raise

        start = perf_counter()

        try:
            inference_result = (
                run_multi_observation_inference_from_config(
                    joint_path
                )
            )
        except Exception as exc:
            self._write_failure(
                index=index,
                stage="inference",
                exc=exc,
            )
            raise

        runtime_seconds = (
            perf_counter()
            - start
        )

        inference_result.save(
            self._result_path(
                index
            )
        )

        failure_path = self._failure_path(
            index
        )

        if failure_path.exists():
            failure_path.unlink()

        return self._recovery_result(
            index=index,
            inference_result=inference_result,
            runtime_seconds=runtime_seconds,
        )

    def load_case(
        self,
        index: int,
    ) -> RecoveryResult:
        """Load one previously completed campaign case."""
        self._validate_index(
            index
        )

        result_path = self._result_path(
            index
        )

        if not result_path.exists():
            raise FileNotFoundError(
                f"Completed campaign result not found: "
                f"{result_path}"
            )

        inference_result = (
            InferenceResult.load(
                result_path
            )
        )

        return self._recovery_result(
            index=index,
            inference_result=inference_result,
            runtime_seconds=None,
        )

    def _recovery_result(
        self,
        *,
        index: int,
        inference_result: InferenceResult,
        runtime_seconds: float | None,
    ) -> RecoveryResult:
        """Convert one inference result to campaign recovery diagnostics."""
        case_truth = (
            self._truth_for_case(
                index
            )
        )

        recovery_fit = (
            recovery_fit_from_inference_result(
                inference_result,
                metadata={
                    "campaign": (
                        "multi_observation_noise"
                    ),
                    "case_index": index,
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

        return RecoveryResult(
            case_id=self._case_id(
                index
            ),
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

    def _truth_for_case(
        self,
        index: int,
    ) -> MultiObservationGroundTruth:
        """Return truth with deterministic per-observation noise seeds."""
        self._validate_index(
            index
        )

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

    def _case_id(
        self,
        index: int,
    ) -> str:
        """Return the stable identifier of one campaign case."""
        self._validate_index(
            index
        )

        return f"noise-{index:03d}"

    def _case_dir(
        self,
        index: int,
    ) -> Path:
        """Return the directory of one campaign case."""
        return (
            self.root
            / self._case_id(
                index
            )
        )

    def _result_path(
        self,
        index: int,
    ) -> Path:
        """Return the canonical inference-result path for one case."""
        return (
            self._case_dir(
                index
            )
            / "joint_inference_result.npz"
        )

    def _failure_path(
        self,
        index: int,
    ) -> Path:
        """Return the failure-record path for one case."""
        return (
            self._case_dir(
                index
            )
            / "failure.yaml"
        )

    def _write_failure(
        self,
        *,
        index: int,
        stage: str,
        exc: Exception,
    ) -> None:
        """Persist one campaign failure without discarding the case."""
        case_dir = self._case_dir(
            index
        )

        case_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        record = {
            "case_id": self._case_id(
                index
            ),
            "status": "failed",
            "stage": stage,
            "exception_type": type(
                exc
            ).__name__,
            "message": str(
                exc
            ),
        }

        with self._failure_path(
            index
        ).open(
            "w"
        ) as stream:
            yaml.safe_dump(
                record,
                stream,
                sort_keys=False,
            )

    def _validate_index(
        self,
        index: int,
    ) -> None:
        """Validate one campaign case index."""
        if (
            index < 0
            or index >= self.n_cases
        ):
            raise IndexError(
                "Campaign case index out of range."
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
