"""Ground-truth descriptions for scientific validation experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class GroundTruth:
    """Immutable description of one simulated validation target.

    Parameters
    ----------
    model_name : str
        Name of the morphology model used to generate the target.
    parameters : mapping of str to float
        True model-parameter values.
    image_shape : tuple of int
        Simulated image shape as ``(ny, nx)``.
    pixel_scale : float, optional
        Pixel scale in the units adopted by the simulation.
    sky_level : float, optional
        True additive sky level.
    random_seed : int, optional
        Random seed used to generate stochastic simulation components.
    psf_metadata, noise_metadata, metadata : mapping, optional
        Descriptive metadata associated with the simulation.
    """

    model_name: str
    parameters: Mapping[str, float]
    image_shape: tuple[int, int]
    pixel_scale: float | None = None
    sky_level: float | None = None
    random_seed: int | None = None
    psf_metadata: Mapping[str, Any] = field(default_factory=dict)
    noise_metadata: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("Model name must be a non-empty string.")

        normalized_parameters: dict[str, float] = {}
        for name, value in self.parameters.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Parameter names must be non-empty strings.")

            numeric_value = float(value)
            if not isfinite(numeric_value):
                raise ValueError("Ground-truth parameters must be finite.")

            normalized_parameters[name] = numeric_value

        if not normalized_parameters:
            raise ValueError("Ground truth must contain at least one parameter.")

        if len(self.image_shape) != 2:
            raise ValueError("Image shape must contain exactly two dimensions.")

        normalized_shape = tuple(int(value) for value in self.image_shape)
        if any(value <= 0 for value in normalized_shape):
            raise ValueError("Image-shape dimensions must be positive.")

        if self.pixel_scale is not None:
            pixel_scale = float(self.pixel_scale)
            if not isfinite(pixel_scale) or pixel_scale <= 0:
                raise ValueError("Pixel scale must be finite and positive.")

            object.__setattr__(self, "pixel_scale", pixel_scale)

        if self.sky_level is not None:
            sky_level = float(self.sky_level)
            if not isfinite(sky_level):
                raise ValueError("Sky level must be finite.")

            object.__setattr__(self, "sky_level", sky_level)

        if self.random_seed is not None:
            if not isinstance(self.random_seed, int) or self.random_seed < 0:
                raise ValueError("Random seed must be a non-negative integer.")

        object.__setattr__(self, "model_name", self.model_name.strip())
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(normalized_parameters),
        )
        object.__setattr__(self, "image_shape", normalized_shape)
        object.__setattr__(
            self,
            "psf_metadata",
            MappingProxyType(dict(self.psf_metadata)),
        )
        object.__setattr__(
            self,
            "noise_metadata",
            MappingProxyType(dict(self.noise_metadata)),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return parameter names in insertion order."""
        return tuple(self.parameters)

    def parameter(self, name: str) -> float:
        """Return one true model-parameter value."""
        return self.parameters[name]