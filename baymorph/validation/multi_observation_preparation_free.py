"""Preparation-free recovery for heterogeneous multi-observation inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from astropy.wcs import WCS

from baymorph.fitting import make_sampler, make_single_sersic_psf_sky_model
from baymorph.fitting.multi_observation_problem import MultiObservationInferenceProblem
from baymorph.fitting.representations import RepresentationFactory
from baymorph.likelihood import GaussianImageLikelihood
from baymorph.likelihood.multi_observation import MultiObservationLikelihood
from baymorph.models.geometry import get_model_geometry
from baymorph.observation.model_parameter_transforms import (
    SkyModelParameterTransform,
)
from baymorph.pipeline.multi_observation_config import build_multi_observation_parameter_layout
from baymorph.results import InferenceResult
from baymorph.simulation.psf import GaussianPSF
from baymorph.validation.multi_observation_campaign import (
    MultiObservationNoiseCampaign,
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
class PreparationFreeMultiObservationCase:
    """Direct native-grid validation case with known Gaussian noise.

    No source detection, masking, background estimation, plateau detection,
    cutout construction, or empirical noise estimation is performed.
    """

    truth: MultiObservationGroundTruth
    problem: MultiObservationInferenceProblem
    noiseless_images: Mapping[str, np.ndarray]
    observed_images: Mapping[str, np.ndarray]
    noise_maps: Mapping[str, np.ndarray]
    noise_seeds: Mapping[str, int]
    wcs_by_observation: Mapping[str, WCS]

    @property
    def truth_vector(self) -> np.ndarray:
        """Return the sampled truth in inference-parameter order."""
        return np.asarray(
            [self.truth.parameter(name) for name in self.problem.parameter_names],
            dtype=float,
        )

    def save_inputs(self, path: str | Path) -> Path:
        """Persist the exact arrays consumed by the preparation-free fit."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, np.ndarray] = {
            "parameter_names": np.asarray(self.problem.parameter_names, dtype=str),
            "truth_vector": self.truth_vector,
        }

        for observation in self.problem.parameter_layout.observations:
            arrays[f"noiseless__{observation}"] = np.asarray(
                self.noiseless_images[observation], dtype=float
            )
            arrays[f"observed__{observation}"] = np.asarray(
                self.observed_images[observation], dtype=float
            )
            arrays[f"noise__{observation}"] = np.asarray(
                self.noise_maps[observation], dtype=float
            )
            arrays[f"noise_seed__{observation}"] = np.asarray(
                self.noise_seeds[observation], dtype=int
            )

        np.savez_compressed(output_path, **arrays)
        return output_path


def build_preparation_free_multi_observation_case(
    truth: MultiObservationGroundTruth,
    *,
    parameter_config: Mapping[str, Any] | None = None,
) -> PreparationFreeMultiObservationCase:
    """Construct a preparation-free native-grid recovery problem.

    The layout and priors come from the same validation prior contract used by
    :class:`MultiObservationValidationWorkspace`. Rendering and inference use
    the ordinary production WCS transform, representation factory, likelihood,
    and backend-neutral multi-observation problem.
    """
    if not isinstance(truth, MultiObservationGroundTruth):
        raise TypeError("truth must be a MultiObservationGroundTruth instance.")

    if truth.model_name != "single_sersic":
        raise ValueError(
            "Initial preparation-free multi-observation recovery supports "
            "only the single_sersic reference model."
        )

    parameter_layout = build_multi_observation_parameter_layout(
        _layout_config(
            truth,
            parameter_config=parameter_config,
        )
    )
    geometry = get_model_geometry(truth.model_name)

    wcs_by_observation: dict[str, WCS] = {}
    fixed_kwargs: dict[str, dict[str, Any]] = {}
    transforms: dict[str, SkyModelParameterTransform] = {}
    noise_sigmas: dict[str, float] = {}
    noise_seeds: dict[str, int] = {}

    for observation_name, observation_truth in truth.observations.items():
        metadata = dict(observation_truth.metadata)
        wcs = _make_observation_wcs(truth, observation_truth, metadata)
        psf_fwhm = _required_finite_positive_value(
            observation_truth.psf_metadata,
            key="fwhm_pixels",
            observation=observation_name,
            quantity="PSF FWHM",
        )
        noise_sigma = _required_finite_positive_value(
            observation_truth.noise_metadata,
            key="sigma",
            observation=observation_name,
            quantity="Gaussian noise sigma",
        )
        seed = _noise_seed(
            metadata=metadata,
            fallback=truth.random_seed,
            observation=observation_name,
        )

        wcs_by_observation[observation_name] = wcs
        fixed_kwargs[observation_name] = {
            "psf": GaussianPSF(fwhm=psf_fwhm),
        }
        transforms[observation_name] = SkyModelParameterTransform(
            wcs=wcs,
            geometry=geometry,
        )
        noise_sigmas[observation_name] = noise_sigma
        noise_seeds[observation_name] = seed

    if len(set(noise_seeds.values())) != len(noise_seeds):
        raise ValueError(
            "Preparation-free observations require distinct noise seeds so "
            "the validation does not reuse one pseudorandom realization "
            "across conditionally independent observations."
        )

    representation_factory = RepresentationFactory(
        factory=make_single_sersic_psf_sky_model,
        fixed_kwargs=fixed_kwargs,
        parameter_transforms=transforms,
    )
    sampled_truth = {
        name: truth.parameter(name)
        for name in parameter_layout.parameter_set.names
    }
    representations = representation_factory.build_all(
        parameter_layout.route(sampled_truth)
    )

    noiseless_images: dict[str, np.ndarray] = {}
    observed_images: dict[str, np.ndarray] = {}
    noise_maps: dict[str, np.ndarray] = {}

    for observation_name in parameter_layout.observations:
        shape = truth.observations[observation_name].image_shape
        noiseless = np.asarray(
            representations[observation_name].render(shape=shape),
            dtype=float,
        )
        sigma = noise_sigmas[observation_name]
        noise_map = np.full(shape, sigma, dtype=float)
        rng = np.random.default_rng(noise_seeds[observation_name])

        # This direct control models residual sky, not the raw detector
        # background removed by the end-to-end preparation workflow.
        observed = noiseless + rng.normal(
            loc=0.0,
            scale=sigma,
            size=shape,
        )

        noiseless_images[observation_name] = noiseless
        observed_images[observation_name] = observed
        noise_maps[observation_name] = noise_map

    likelihood = MultiObservationLikelihood(
        {
            observation_name: GaussianImageLikelihood(
                data=observed_images[observation_name],
                noise=noise_maps[observation_name],
                mask=None,
                include_normalization=True,
            )
            for observation_name in parameter_layout.observations
        }
    )
    problem = MultiObservationInferenceProblem(
        representation_factory=representation_factory,
        parameter_layout=parameter_layout,
        likelihood=likelihood,
    )

    return PreparationFreeMultiObservationCase(
        truth=truth,
        problem=problem,
        noiseless_images=noiseless_images,
        observed_images=observed_images,
        noise_maps=noise_maps,
        noise_seeds=noise_seeds,
        wcs_by_observation=wcs_by_observation,
    )


class PreparationFreeMultiObservationNoiseCampaign(MultiObservationNoiseCampaign):
    """Repeated-noise campaign that bypasses image preparation entirely."""

    def build_case(self, index: int) -> PreparationFreeMultiObservationCase:
        """Construct one deterministic case without running inference."""
        self._validate_index(index)
        return build_preparation_free_multi_observation_case(
            self._truth_for_case(index)
        )

    def run_case(self, index: int) -> RecoveryResult:
        """Execute one preparation-free native-grid noise realization."""
        self._validate_index(index)

        try:
            case = self.build_case(index)
        except Exception as exc:
            self._write_failure(
                index=index,
                stage="problem_construction",
                exc=exc,
            )
            raise

        try:
            case.save_inputs(self._inputs_path(index))
        except Exception as exc:
            self._write_failure(
                index=index,
                stage="input_persistence",
                exc=exc,
            )
            raise

        start = perf_counter()
        try:
            backend = make_sampler(self.sampler, problem=case.problem)
            inference_result = backend.fit(
                progress_label=f"preparation_free_{self._case_id(index)}"
            )
        except Exception as exc:
            self._write_failure(
                index=index,
                stage="inference",
                exc=exc,
            )
            raise

        runtime_seconds = perf_counter() - start
        inference_result.save(self._result_path(index))

        failure_path = self._failure_path(index)
        if failure_path.exists():
            failure_path.unlink()

        return self._recovery_result(
            index=index,
            inference_result=inference_result,
            runtime_seconds=runtime_seconds,
        )

    def _case_dir(self, index: int) -> Path:
        """Return an isolated directory for one preparation-free case."""
        self._validate_index(index)
        return self.root / "preparation_free" / self._case_id(index)

    def _inputs_path(self, index: int) -> Path:
        """Return the persisted exact-input path for one campaign case."""
        return self._case_dir(index) / "preparation_free_inputs.npz"

    def _recovery_result(
        self,
        *,
        index: int,
        inference_result: InferenceResult,
        runtime_seconds: float | None,
    ) -> RecoveryResult:
        """Attach preparation-free provenance to one recovery result."""
        result = super()._recovery_result(
            index=index,
            inference_result=inference_result,
            runtime_seconds=runtime_seconds,
        )
        metadata = dict(result.metadata or {})
        metadata.update(
            {
                "campaign": "preparation_free_multi_observation_noise",
                "preparation_free": True,
                "likelihood_grid": "native_detector",
            }
        )
        return replace(result, metadata=metadata)


def _make_observation_wcs(
    truth: MultiObservationGroundTruth,
    observation_truth: ObservationTruth,
    metadata: Mapping[str, Any],
) -> WCS:
    """Build one native synthetic WCS from the validation truth."""
    crpix = metadata.get("crpix")
    if crpix is None:
        ny, nx = observation_truth.image_shape
        crpix = (0.5 * (nx + 1), 0.5 * (ny + 1))

    return MultiObservationValidationWorkspace._make_wcs(
        ra=truth.parameter("ra"),
        dec=truth.parameter("dec"),
        pixel_scale_arcsec=observation_truth.pixel_scale,
        rotation_deg=float(metadata.get("rotation_deg", 0.0)),
        crpix=(float(crpix[0]), float(crpix[1])),
    )


def _required_finite_positive_value(
    metadata: Mapping[str, Any],
    *,
    key: str,
    observation: str,
    quantity: str,
) -> float:
    """Return one required finite positive observation metadata value."""
    try:
        value = float(metadata[key])
    except KeyError as exc:
        raise ValueError(
            f"Preparation-free validation requires {key!r} for observation "
            f"{observation!r}."
        ) from exc

    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"Preparation-free {quantity} must be finite and positive for "
            f"observation {observation!r}."
        )
    return value


def _noise_seed(
    *,
    metadata: Mapping[str, Any],
    fallback: int | None,
    observation: str,
) -> int:
    """Return one explicit reproducible observation-specific noise seed."""
    seed = metadata.get("seed", fallback)
    if (
        not isinstance(seed, (int, np.integer))
        or isinstance(seed, (bool, np.bool_))
        or int(seed) < 0
    ):
        raise ValueError(
            "Preparation-free validation requires a non-negative integer "
            f"noise seed for observation {observation!r}."
        )
    return int(seed)


def _layout_config(
    truth: MultiObservationGroundTruth,
    *,
    parameter_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the minimal production layout config for direct validation."""
    parameters = (
        MultiObservationValidationWorkspace._parameter_config(truth)
        if parameter_config is None
        else dict(parameter_config)
    )

    return {
        "observations": {
            observation_name: {"config_path": f"{observation_name}.yaml"}
            for observation_name in truth.observations
        },
        "fit": {
            "models": [truth.model_name],
            "multi_observation": {
                "parameter_frame": "sky",
                "parameters": parameters,
            },
        },
    }
