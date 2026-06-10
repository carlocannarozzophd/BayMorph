"""Core data structures and utilities for BayMorph."""

from .background import (
    BackgroundEstimate,
    estimate_background_sigma,
    subtract_background,
)
from .image import ImageData

__all__ = [
    "ImageData",
    "BackgroundEstimate",
    "estimate_background_sigma",
    "subtract_background",
]