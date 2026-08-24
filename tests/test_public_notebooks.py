"""Tests for the canonical public BayMorph notebooks."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIRECTORY = PROJECT_ROOT / "notebooks"

EXPECTED_NOTEBOOKS = {
    "01_Simulated_SingleBand.ipynb",
    "02_Observed_SingleBand.ipynb",
}


def _notebook_source(
    path: Path,
) -> str:
    """Return concatenated code and Markdown source from one notebook."""
    notebook = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    return "\n".join(
        "".join(
            cell.get(
                "source",
                [],
            )
        )
        for cell in notebook.get(
            "cells",
            []
        )
    )


def test_exactly_two_public_notebooks_exist() -> None:
    """The repository exposes exactly two canonical public notebooks."""
    notebooks = {
        path.name
        for path in NOTEBOOK_DIRECTORY.glob(
            "*.ipynb"
        )
        if path.is_file()
    }

    assert notebooks == EXPECTED_NOTEBOOKS


def test_no_other_repository_notebooks_exist() -> None:
    """Notebook duplication must not reappear elsewhere in the repository."""
    notebooks = {
        path.relative_to(
            PROJECT_ROOT
        )
        for path in PROJECT_ROOT.rglob(
            "*.ipynb"
        )
        if ".git" not in path.parts
    }

    assert notebooks == {
        Path("notebooks")
        / name
        for name in EXPECTED_NOTEBOOKS
    }


def test_simulated_notebook_uses_public_runner_workflow() -> None:
    """The simulated tutorial must remain thin and API-driven."""
    source = _notebook_source(
        NOTEBOOK_DIRECTORY
        / "01_Simulated_SingleBand.ipynb"
    )

    assert "BayMorphRunner" in source
    assert "make_config_from_fits" in source
    assert "make_observation" in source
    assert "runner.run(diagnostics=True)" in source
    assert "runner.prepare_image()" in source
    assert "runner.fit()" in source

    assert "configs/simulated_galaxy.yaml" not in source
    assert "pool=" not in source
    assert "min_factor" not in source
    assert "max_factor" not in source


def test_observed_notebook_uses_config_maker() -> None:
    """The observed tutorial must delegate configuration to the public maker."""
    source = _notebook_source(
        NOTEBOOK_DIRECTORY
        / "02_Observed_SingleBand.ipynb"
    )

    assert "make_config_from_fits" in source
    assert "BayMorphRunner" in source
    assert "runner.run(diagnostics=True)" in source
    assert "model_priors" in source
    assert "correlated_gaussian" in source
    assert "runner.prepare_image()" in source
    assert "runner.build_psf()" in source
    assert "runner.fit()" in source

    assert "pool=" not in source
    assert "min_factor" not in source
    assert "max_factor" not in source


def test_public_notebooks_use_repository_root_relative_paths() -> None:
    """Canonical notebooks must not depend on notebook-directory paths."""
    for name in EXPECTED_NOTEBOOKS:
        source = _notebook_source(
            NOTEBOOK_DIRECTORY
            / name
        )

        assert "../configs/" not in source
        assert "../data/" not in source
        assert "../runs" not in source


def test_public_notebooks_do_not_use_matplotlib_directly() -> None:
    """Public notebooks must delegate all plotting to BayMorph."""
    forbidden = (
        "import matplotlib",
        "from matplotlib",
        "plt.",
        ".imshow(",
        ".subplots(",
        ".colorbar(",
    )

    for name in EXPECTED_NOTEBOOKS:
        source = _notebook_source(
            NOTEBOOK_DIRECTORY
            / name
        )

        for token in forbidden:
            assert token not in source


def test_observed_notebook_exposes_direct_and_guided_configuration() -> None:
    """The real-data tutorial must expose both canonical config workflows."""
    source = _notebook_source(
        NOTEBOOK_DIRECTORY
        / "02_Observed_SingleBand.ipynb"
    )

    assert "ObservationAdvisor" in source
    assert "advisor.summary()" in source
    assert "advisor.make_config(" in source
    assert "advisor.calibrate_flux_prior(" in source
    assert "advisor.configure(" in source
    assert "make_config_from_fits(" in source

    assert 'configuration_mode = "guided"' in source
    assert 'configuration_mode == "guided"' in source
    assert 'configuration_mode == "direct"' in source

    assert "flux_calibration=flux_calibration" in source
    assert "prepare=True" in source
    assert "build_psf=True" in source
    assert "direct_flux_prior = None" in source

    assert "1.0e10" not in source
    assert '"include_normalization": True' in source


def test_public_notebooks_do_not_expose_fit_region() -> None:
    """Canonical workflows must use the complete prepared usable footprint."""
    for name in EXPECTED_NOTEBOOKS:
        source = _notebook_source(
            NOTEBOOK_DIRECTORY
            / name
        )

        assert "fit_region" not in source
        assert '"region"' not in source
