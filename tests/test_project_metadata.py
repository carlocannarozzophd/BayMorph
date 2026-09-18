"""Tests for BayMorph project dependency metadata."""

from __future__ import annotations

import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_metadata() -> dict:
    """Load project metadata from pyproject.toml."""
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def test_interactive_mask_dependencies_are_declared() -> None:
    """Interactive masking must expose a complete optional dependency set."""
    interactive = set(
        project_metadata()["project"]["optional-dependencies"]["interactive"]
    )

    assert interactive == {
        "anywidget",
        "ipython",
        "ipywidgets",
        "plotly",
    }


def test_screening_training_dependency_is_declared() -> None:
    """Screening training must declare its direct development dependency."""
    development = set(
        project_metadata()["project"]["optional-dependencies"]["dev"]
    )

    assert "scikit-learn" in development
