"""Proposal blocks used by the geometry-aware sampler."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProposalBlock:
    """Named group of parameters proposed together.

    Parameters
    ----------
    name : str
        Unique name identifying the proposal block.
    parameters : tuple of str
        Ordered parameter names belonging to the block.
    """

    name: str
    parameters: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the proposal-block definition."""
        if not self.name:
            raise ValueError("Proposal block name cannot be empty.")

        if not self.parameters:
            raise ValueError(
                f"Proposal block {self.name!r} must contain at least one parameter."
            )

        if any(not parameter for parameter in self.parameters):
            raise ValueError(
                f"Proposal block {self.name!r} contains an empty parameter name."
            )

        if len(set(self.parameters)) != len(self.parameters):
            raise ValueError(
                f"Proposal block {self.name!r} contains duplicate parameters."
            )
