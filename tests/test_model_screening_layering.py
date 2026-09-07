"""Tests for model-screening dependency direction."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_model_screening_runtime_does_not_import_validation() -> None:
    """Public runtime modules must not depend on validation infrastructure."""
    runtime_files = [
        PROJECT_ROOT / "baymorph" / "__init__.py",
        (
            PROJECT_ROOT
            / "baymorph"
            / "pipeline"
            / "model_screening_advisor.py"
        ),
        *sorted(
            (
                PROJECT_ROOT
                / "baymorph"
                / "analysis"
            ).glob(
                "model_screening*.py"
            )
        ),
    ]

    offenders = []

    for filename in runtime_files:
        source = filename.read_text()

        if "baymorph.validation" in source:
            offenders.append(
                str(
                    filename.relative_to(
                        PROJECT_ROOT
                    )
                )
            )

    assert offenders == []
