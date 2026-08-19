"""Proposal engines for the native geometry-aware nested sampler."""

from .base import ProposalEngine, ProposalFailure, ProposalResult
from .gaussian import GaussianProposalEngine
from .prior import PriorProposalEngine
from .slice import WhitenedSliceProposalEngine

__all__ = [
    "ProposalEngine",
    "ProposalFailure",
    "ProposalResult",
    "PriorProposalEngine",
    "GaussianProposalEngine",
    "WhitenedSliceProposalEngine",
]
