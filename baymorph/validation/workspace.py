"""Disk workspace for end-to-end validation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from baymorph.validation.simulation import (
    build_synthetic_observation,
    simulated_effective_gain,
)
from baymorph.validation.truth import GroundTruth


def validate_case_id(case_id: str) -> str:
    """Validate and normalize one validation case identifier."""
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("Case ID must be a non-empty string.")

    normalized = case_id.strip()

    if Path(normalized).name != normalized:
        raise ValueError("Case ID must not contain path components.")

    return normalized


@dataclass(frozen=True)
class ValidationWorkspace:
    """Materialize one validation case for the public BayMorph pipeline.

    Parameters
    ----------
    root : str or pathlib.Path
        Directory containing validation inputs, configuration, and outputs.
    """

    root: str | Path

    def __post_init__(self) -> None:
        root = Path(self.root)

        if root.exists() and not root.is_dir():
            raise ValueError("Validation workspace root must be a directory.")

        object.__setattr__(self, "root", root)

    def write_case(
        self,
        case_id: str,
        truth: GroundTruth,
        *,
        models: tuple[str, ...] = ("single_sersic",),
        sampler: dict[str, Any] | None = None,
    ) -> Path:
        """Write a simulated FITS image and resolved validation YAML.

        Parameters
        ----------
        case_id : str
            Stable identifier for the validation case.
        truth : GroundTruth
            Ground truth used to generate the synthetic observation.
        models : tuple of str, default=("single_sersic",)
            Morphological models fitted by BayMorph.
        sampler : dict, optional
            Sampler envelope entries (``name``, ``mode``, ``execution``,
            ``init``, ``run``) overriding the validation defaults. The
            ``execution``, ``init`` and ``run`` sections are merged
            key by key.

        Returns
        -------
        pathlib.Path
            Path to the generated YAML configuration.

        Raises
        ------
        ValueError
            If identifiers, image geometry, models, or PSF metadata are
            unsuitable for the initial end-to-end validation workflow.
        """
        normalized_case_id = self._validate_case_id(case_id)
        self._validate_truth(truth)

        if not models:
            raise ValueError("At least one fitting model is required.")

        case_dir = self.root / normalized_case_id
        input_dir = case_dir / "input"
        output_dir = case_dir / "runs"

        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        observation = build_synthetic_observation(truth)

        image_path = input_dir / "synthetic_observation.fits"
        observation.to_fits(image_path)

        sampler_config: dict[str, Any] = {
            "name": "dynesty",
            "mode": "static",
            "execution": {
                "processes": 1,
            },
            "init": {
                "nlive": 100,
                "sample": "auto",
            },
            "run": {
                "dlogz": 0.5,
            },
        }

        if sampler is not None:
            for key, value in sampler.items():
                if (
                    key in {"execution", "init", "run"}
                    and isinstance(value, dict)
                    and isinstance(sampler_config.get(key), dict)
                ):
                    sampler_config[key] = {
                        **sampler_config[key],
                        **value,
                    }
                else:
                    sampler_config[key] = value

        config = {
            "object": {
                "name": normalized_case_id,
            },
            "data": {
                "image_path": str(image_path),
                "pixel_scale": truth.pixel_scale,
                "target_position": {
                    "x": truth.parameter("x0"),
                    "y": truth.parameter("y0"),
                },
                "cutout": {
                    "mode": "fixed",
                    "size": truth.image_shape[0],
                },
            },
            "outputs": {
                "root": str(output_dir),
                "overwrite": True,
            },
            "preparation": {
                "background": {
                    "box_size": 30,
                    "filter_size": 3,
                },
                "noise": {},
                "detection": {
                    "n_sigma": 3.0,
                    "n_pixels": 20,
                    "n_levels": 32,
                },
            },
            "psf": {
                "type": "gaussian",
                "gaussian": {
                    "fwhm": float(truth.psf_metadata["fwhm"]),
                },
            },
            "fit": {
                "models": list(models),
                "likelihood": {
                    "type": "correlated_gaussian",
                    "effective_gain": simulated_effective_gain(
                        truth.noise_metadata
                    ),
                },
                "sampler": sampler_config,
                "comparison": {
                    "enabled": len(models) > 1,
                    "metric": "bayes_factor",
                },
            },
        }

        config_path = case_dir / "validation.yaml"

        with config_path.open("w") as stream:
            yaml.safe_dump(
                config,
                stream,
                sort_keys=False,
            )

        return config_path

    _validate_case_id = staticmethod(validate_case_id)

    @staticmethod
    def _validate_truth(truth: GroundTruth) -> None:
        """Validate truth requirements for the initial disk workflow."""
        if truth.image_shape[0] != truth.image_shape[1]:
            raise ValueError(
                "Initial validation workspaces require square images."
            )

        if "fwhm" not in truth.psf_metadata:
            raise ValueError(
                "Validation truth must define Gaussian PSF metadata 'fwhm'."
            )

        fwhm = float(truth.psf_metadata["fwhm"])

        if fwhm <= 0:
            raise ValueError("Gaussian PSF FWHM must be positive.")

        for name in ("x0", "y0"):
            if name not in truth.parameters:
                raise ValueError(
                    f"Validation truth must define parameter '{name}'."
                )