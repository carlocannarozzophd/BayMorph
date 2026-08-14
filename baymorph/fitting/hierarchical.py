"""Domain transforms for hierarchical multi-observation parameters."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.special import expit


class HierarchicalDomainTransform(ABC):
    """Map one unconstrained hierarchical latent value to physical space."""

    @abstractmethod
    def from_latent(self, value: float) -> float:
        """Map one finite latent-space value to a physical parameter."""


class IdentityHierarchicalTransform(HierarchicalDomainTransform):
    """Identity mapping for real-valued physical parameters."""

    def from_latent(self, value: float) -> float:
        value = float(value)

        if not np.isfinite(value):
            raise ValueError(
                "Hierarchical latent value must be finite."
            )

        return value


class ExpHierarchicalTransform(HierarchicalDomainTransform):
    """Exponential mapping for strictly positive physical parameters."""

    def from_latent(self, value: float) -> float:
        value = float(value)

        if not np.isfinite(value):
            raise ValueError(
                "Hierarchical latent value must be finite."
            )

        with np.errstate(
            over="ignore",
            invalid="ignore",
        ):
            result = float(
                np.exp(value)
            )

        if (
            not np.isfinite(result)
            or result <= 0.0
        ):
            raise ValueError(
                "Exponential hierarchical transform produced "
                "a non-finite or non-positive value."
            )

        return result


class LogisticHierarchicalTransform(
    HierarchicalDomainTransform
):
    """Logistic mapping for physical parameters restricted to (0, 1)."""

    def from_latent(self, value: float) -> float:
        value = float(value)

        if not np.isfinite(value):
            raise ValueError(
                "Hierarchical latent value must be finite."
            )

        result = float(
            expit(value)
        )

        lower_open = np.nextafter(
            0.0,
            1.0,
        )
        upper_open = np.nextafter(
            1.0,
            0.0,
        )

        return float(
            np.clip(
                result,
                lower_open,
                upper_open,
            )
        )


class AxialAngleHierarchicalTransform(
    HierarchicalDomainTransform
):
    """Wrap an axial angle into the canonical interval [-pi/2, pi/2)."""

    def from_latent(self, value: float) -> float:
        value = float(value)

        if not np.isfinite(value):
            raise ValueError(
                "Hierarchical latent value must be finite."
            )

        half_period = float(
            np.pi / 2.0
        )
        period = float(
            np.pi
        )

        return float(
            (
                value
                + half_period
            )
            % period
            - half_period
        )


def hierarchical_domain_transform_from_name(
    name: str,
) -> HierarchicalDomainTransform:
    """Build a hierarchical domain transform from its configuration name."""
    normalized = str(name).strip().lower()

    if normalized == "identity":
        return IdentityHierarchicalTransform()

    if normalized == "exp":
        return ExpHierarchicalTransform()

    if normalized == "logistic":
        return LogisticHierarchicalTransform()

    if normalized == "axial_angle":
        return AxialAngleHierarchicalTransform()

    raise ValueError(
        "Hierarchical transform must be "
        "'identity', 'exp', 'logistic', or 'axial_angle'."
    )
