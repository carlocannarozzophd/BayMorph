"""Ground-truth contracts for heterogeneous multi-observation validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class ObservationTruth:
    """Ground-truth description of one synthetic observation.

    Parameters
    ----------
    image_shape : tuple of int
        Native detector image shape as ``(ny, nx)``.
    pixel_scale : float
        Local detector pixel scale in arcsec per pixel.
    psf_metadata, noise_metadata, metadata : mapping, optional
        Observation-specific simulation metadata.
    """

    image_shape: tuple[int, int]
    pixel_scale: float
    psf_metadata: Mapping[str, Any] = field(default_factory=dict)
    noise_metadata: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.image_shape) != 2:
            raise ValueError(
                "Observation image shape must contain exactly two dimensions."
            )

        normalized_shape = tuple(
            int(value)
            for value in self.image_shape
        )

        if any(
            value <= 0
            for value in normalized_shape
        ):
            raise ValueError(
                "Observation image-shape dimensions must be positive."
            )

        pixel_scale = float(
            self.pixel_scale
        )

        if (
            not isfinite(pixel_scale)
            or pixel_scale <= 0.0
        ):
            raise ValueError(
                "Observation pixel scale must be finite and positive."
            )

        object.__setattr__(
            self,
            "image_shape",
            normalized_shape,
        )
        object.__setattr__(
            self,
            "pixel_scale",
            pixel_scale,
        )
        object.__setattr__(
            self,
            "psf_metadata",
            MappingProxyType(
                dict(self.psf_metadata)
            ),
        )
        object.__setattr__(
            self,
            "noise_metadata",
            MappingProxyType(
                dict(self.noise_metadata)
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                dict(self.metadata)
            ),
        )


@dataclass(frozen=True)
class MultiObservationGroundTruth:
    """Ground truth for one galaxy observed through multiple detectors.

    Parameters
    ----------
    model_name : str
        Registered morphology-model name.
    parameters : mapping of str to float
        Complete true inferential state in the common scientific frame.
        Observation-specific parameters use the same global naming convention
        adopted by the multi-observation parameter layout.
    observations : mapping of str to ObservationTruth
        Native observation-specific detector and simulation properties.
    random_seed : int, optional
        Campaign-level random seed.
    metadata : mapping, optional
        Additional validation metadata.

    Notes
    -----
    The entries in ``observations`` describe multiple measurements of one
    physical galaxy. They do not represent independent astronomical objects.
    """

    model_name: str
    parameters: Mapping[str, float]
    observations: Mapping[str, ObservationTruth]
    random_seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.model_name, str)
            or not self.model_name.strip()
        ):
            raise ValueError(
                "Model name must be a non-empty string."
            )

        normalized_parameters: dict[str, float] = {}

        for name, value in self.parameters.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "Parameter names must be non-empty strings."
                )

            numeric_value = float(
                value
            )

            if not isfinite(
                numeric_value
            ):
                raise ValueError(
                    "Ground-truth parameters must be finite."
                )

            normalized_parameters[name] = numeric_value

        if not normalized_parameters:
            raise ValueError(
                "Ground truth must contain at least one parameter."
            )

        normalized_observations = {}

        for name, observation in self.observations.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "Observation names must be non-empty strings."
                )

            if not isinstance(
                observation,
                ObservationTruth,
            ):
                raise TypeError(
                    "Multi-observation truth values must be "
                    "ObservationTruth instances."
                )

            normalized_observations[name] = observation

        if not normalized_observations:
            raise ValueError(
                "Multi-observation ground truth requires at least "
                "one observation."
            )

        if self.random_seed is not None:
            if (
                not isinstance(self.random_seed, int)
                or self.random_seed < 0
            ):
                raise ValueError(
                    "Random seed must be a non-negative integer."
                )

        object.__setattr__(
            self,
            "model_name",
            self.model_name.strip(),
        )
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(
                normalized_parameters
            ),
        )
        object.__setattr__(
            self,
            "observations",
            MappingProxyType(
                normalized_observations
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                dict(self.metadata)
            ),
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return true parameter names in insertion order."""
        return tuple(
            self.parameters
        )

    def parameter(
        self,
        name: str,
    ) -> float:
        """Return one true inferential parameter."""
        return self.parameters[name]
