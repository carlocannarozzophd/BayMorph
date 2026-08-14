"""Prior-predictive SBC construction for multi-observation inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from baymorph.fitting.multi_observation import MultiObservationParameterLayout
from baymorph.pipeline.multi_observation_config import build_multi_observation_parameter_layout
from baymorph.validation.multi_observation_preparation_free import (
    PreparationFreeMultiObservationCase,
    build_preparation_free_multi_observation_case,
)
from baymorph.validation.multi_observation_truth import (
    MultiObservationGroundTruth,
    ObservationTruth,
)


@dataclass(frozen=True)
class MultiObservationSBCTruthDraw:
    """One deterministic prior-predictive SBC truth realization."""

    truth: MultiObservationGroundTruth
    unit_cube: np.ndarray
    parameter_layout: MultiObservationParameterLayout
    truth_seed: int


def reference_multi_observation_sbc_parameter_config() -> dict[str, Any]:
    """Return the fixed prior contract for the initial two-band SBC study."""
    arcsec_in_degrees = 1.0 / 3600.0

    def uniform(lower: float, upper: float) -> dict[str, float | str]:
        return {
            "type": "uniform",
            "lower": float(lower),
            "upper": float(upper),
        }

    return {
        "flux": {
            "coupling": "independent",
            "priors": {
                "g": uniform(450.0, 1350.0),
                "r": uniform(600.0, 1800.0),
            },
        },
        "re_arcsec": {
            "coupling": "shared",
            "prior": uniform(0.6, 1.8),
        },
        "n": {
            "coupling": "shared",
            "prior": uniform(0.8, 4.0),
        },
        "q": {
            "coupling": "shared",
            "prior": uniform(0.3, 1.0),
        },
        "position_angle": {
            "coupling": "shared",
            "prior": uniform(
                -0.5 * np.pi,
                0.5 * np.pi,
            ),
        },
        "ra": {
            "coupling": "shared",
            "prior": uniform(
                150.0 - arcsec_in_degrees,
                150.0 + arcsec_in_degrees,
            ),
        },
        "dec": {
            "coupling": "shared",
            "prior": uniform(
                30.0 - arcsec_in_degrees,
                30.0 + arcsec_in_degrees,
            ),
        },
        "sky": {
            "coupling": "independent",
            "priors": {
                "g": uniform(-20.0, 20.0),
                "r": uniform(-20.0, 20.0),
            },
        },
    }


def draw_multi_observation_sbc_truth(
    reference_truth: MultiObservationGroundTruth,
    *,
    index: int,
    truth_seed_base: int = 30000,
    noise_seed_base: int = 40000,
) -> MultiObservationSBCTruthDraw:
    """Draw one reproducible SBC truth from the fixed inference prior."""
    if not isinstance(reference_truth, MultiObservationGroundTruth):
        raise TypeError(
            "reference_truth must be a MultiObservationGroundTruth instance."
        )

    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("SBC case index must be a non-negative integer.")

    observations = tuple(reference_truth.observations)
    if observations != ("g", "r"):
        raise ValueError(
            "The initial SBC reference contract requires observations "
            "('g', 'r') in that order."
        )

    parameter_config = reference_multi_observation_sbc_parameter_config()

    layout_config = {
        "observations": {
            name: {"config_path": f"{name}.yaml"}
            for name in observations
        },
        "fit": {
            "models": [reference_truth.model_name],
            "multi_observation": {
                "parameter_frame": "sky",
                "parameters": parameter_config,
            },
        },
    }

    parameter_layout = build_multi_observation_parameter_layout(layout_config)

    truth_seed = truth_seed_base + index
    rng = np.random.default_rng(truth_seed)
    unit_cube = rng.random(parameter_layout.ndim)

    sampled_truth = parameter_layout.parameter_set.transform(unit_cube)

    parameters = {
        name: float(value)
        for name, value in zip(
            parameter_layout.parameter_set.names,
            sampled_truth,
            strict=True,
        )
    }

    drawn_observations: dict[str, ObservationTruth] = {}

    for offset, name in enumerate(observations):
        reference = reference_truth.observations[name]
        metadata = dict(reference.metadata)
        metadata["seed"] = (
            noise_seed_base
            + index * len(observations)
            + offset
        )

        drawn_observations[name] = ObservationTruth(
            image_shape=reference.image_shape,
            pixel_scale=reference.pixel_scale,
            psf_metadata=dict(reference.psf_metadata),
            noise_metadata=dict(reference.noise_metadata),
            metadata=metadata,
        )

    truth = MultiObservationGroundTruth(
        model_name=reference_truth.model_name,
        parameters=parameters,
        observations=drawn_observations,
        random_seed=truth_seed,
        metadata={
            **dict(reference_truth.metadata),
            "validation_mode": "prior_predictive_sbc",
            "truth_seed": truth_seed,
        },
    )

    return MultiObservationSBCTruthDraw(
        truth=truth,
        unit_cube=unit_cube,
        parameter_layout=parameter_layout,
        truth_seed=truth_seed,
    )


def build_preparation_free_sbc_case(
    reference_truth: MultiObservationGroundTruth,
    *,
    index: int,
) -> tuple[
    MultiObservationSBCTruthDraw,
    PreparationFreeMultiObservationCase,
]:
    """Construct one prior-predictive preparation-free SBC case."""
    draw = draw_multi_observation_sbc_truth(
        reference_truth,
        index=index,
    )

    case = build_preparation_free_multi_observation_case(
        draw.truth,
        parameter_config=reference_multi_observation_sbc_parameter_config(),
    )

    return draw, case


class PreparationFreeMultiObservationSBCCampaign(
    __import__(
        "baymorph.validation.multi_observation_preparation_free",
        fromlist=["PreparationFreeMultiObservationNoiseCampaign"],
    ).PreparationFreeMultiObservationNoiseCampaign
):
    """Prior-predictive preparation-free SBC campaign.

    The campaign reuses the preparation-free recovery machinery while
    replacing the fixed physical truth with one deterministic prior draw per
    case. Truth draws and detector-noise realizations use independent seed
    schedules.
    """

    def draw_case_truth(
        self,
        index: int,
    ) -> MultiObservationSBCTruthDraw:
        """Return the deterministic prior-predictive truth for one case."""
        self._validate_index(index)

        return draw_multi_observation_sbc_truth(
            self.truth,
            index=index,
        )

    def _truth_for_case(
        self,
        index: int,
    ) -> MultiObservationGroundTruth:
        """Return the prior-predictive truth used by one SBC case."""
        return self.draw_case_truth(index).truth

    def build_case(
        self,
        index: int,
    ) -> PreparationFreeMultiObservationCase:
        """Construct one preparation-free SBC inference problem."""
        self._validate_index(index)

        draw = self.draw_case_truth(index)

        return build_preparation_free_multi_observation_case(
            draw.truth,
            parameter_config=(
                reference_multi_observation_sbc_parameter_config()
            ),
        )

    def run_case(
        self,
        index: int,
    ):
        """Persist the SBC truth draw and execute one preparation-free fit."""
        self._validate_index(index)

        try:
            self._save_truth_draw(
                self.draw_case_truth(index)
            )
        except Exception as exc:
            self._write_failure(
                index=index,
                stage="truth_persistence",
                exc=exc,
            )
            raise

        return super().run_case(index)

    def _case_id(
        self,
        index: int,
    ) -> str:
        """Return the stable SBC case identifier."""
        self._validate_index(index)
        return f"sbc-{index:03d}"

    def _case_dir(
        self,
        index: int,
    ):
        """Return an isolated directory for one preparation-free SBC case."""
        self._validate_index(index)

        return (
            self.root
            / "preparation_free_sbc"
            / self._case_id(index)
        )

    def _truth_draw_path(
        self,
        index: int,
    ):
        """Return the persisted prior-predictive truth-draw path."""
        return self._case_dir(index) / "sbc_truth_draw.npz"

    def _save_truth_draw(
        self,
        draw: MultiObservationSBCTruthDraw,
    ):
        """Persist one exact unit-cube and physical SBC truth draw."""
        index = draw.truth_seed - 30000
        path = self._truth_draw_path(index)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        parameter_names = (
            draw.parameter_layout.parameter_set.names
        )

        truth_vector = np.asarray(
            [
                draw.truth.parameter(name)
                for name in parameter_names
            ],
            dtype=float,
        )

        arrays: dict[str, np.ndarray] = {
            "unit_cube": np.asarray(
                draw.unit_cube,
                dtype=float,
            ),
            "truth_seed": np.asarray(
                draw.truth_seed,
                dtype=int,
            ),
            "parameter_names": np.asarray(
                parameter_names,
                dtype=str,
            ),
            "truth_vector": truth_vector,
        }

        for name, observation in draw.truth.observations.items():
            arrays[f"noise_seed__{name}"] = np.asarray(
                observation.metadata["seed"],
                dtype=int,
            )

        np.savez_compressed(
            path,
            **arrays,
        )

        return path

    def _recovery_result(
        self,
        *,
        index: int,
        inference_result,
        runtime_seconds: float | None,
    ):
        """Attach prior-predictive SBC provenance to one recovery result."""
        result = super()._recovery_result(
            index=index,
            inference_result=inference_result,
            runtime_seconds=runtime_seconds,
        )

        metadata = dict(
            result.metadata or {}
        )
        metadata.update(
            {
                "campaign": (
                    "preparation_free_multi_observation_sbc"
                ),
                "prior_predictive": True,
                "truth_seed": (
                    self.draw_case_truth(index).truth_seed
                ),
            }
        )

        return type(result)(
            case_id=result.case_id,
            truth=dict(result.truth),
            estimates=dict(result.estimates),
            intervals=(
                None
                if result.intervals is None
                else dict(result.intervals)
            ),
            log_evidence=result.log_evidence,
            log_evidence_error=result.log_evidence_error,
            runtime_seconds=result.runtime_seconds,
            likelihood_evaluations=(
                result.likelihood_evaluations
            ),
            metadata=metadata,
        )
