"""Tests for the canonical public BayMorph notebooks."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIRECTORY = PROJECT_ROOT / "notebooks"

EXPECTED_NOTEBOOKS = {
    "01_Simulated_SingleBand.ipynb",
    "02A_Observed_Assessment_and_Config.ipynb",
    "02B_Observed_Fit.ipynb",
    "03_MultiObservation_JointFit.ipynb",
}


def _notebook_source(
    path: Path,
) -> str:
    """Return concatenated code and Markdown source from one notebook."""
    notebook = json.loads(path.read_text(encoding="utf-8"))

    return "\n".join(
        "".join(
            cell.get(
                "source",
                [],
            )
        )
        for cell in notebook.get("cells", [])
    )


def test_exactly_four_public_notebooks_exist() -> None:
    """The repository exposes exactly four canonical public notebooks."""
    notebooks = {
        path.name for path in NOTEBOOK_DIRECTORY.glob("*.ipynb") if path.is_file()
    }

    assert notebooks == EXPECTED_NOTEBOOKS


def test_only_public_and_packaged_notebooks_exist() -> None:
    """Notebook copies must remain limited to public and packaged resources."""
    notebooks = {
        path.relative_to(PROJECT_ROOT)
        for path in PROJECT_ROOT.rglob("*.ipynb")
        if ".git" not in path.parts
    }

    expected_public = {Path("notebooks") / name for name in EXPECTED_NOTEBOOKS}
    expected_packaged = {
        Path("baymorph/resources/notebooks") / name for name in EXPECTED_NOTEBOOKS
    }

    assert notebooks == (expected_public | expected_packaged)


def test_simulated_notebook_uses_public_runner_workflow() -> None:
    """The simulated tutorial must remain a linear recovery example."""
    source = _notebook_source(NOTEBOOK_DIRECTORY / "01_Simulated_SingleBand.ipynb")

    assert "BayMorphRunner" in source
    assert "configs/simulated_galaxy.yaml" in source
    assert "runner.run(diagnostics=True)" in source
    assert "runner.simulation.image" in source
    assert "fit.posterior.summary(parameter)" in source
    assert "recovery_table" in source

    assert "make_config_from_fits" not in source
    assert "make_observation" not in source
    assert "SyntheticSersic" not in source
    assert "runner.prepare_image()" not in source
    assert "runner.fit()" not in source
    assert "project_root" not in source
    assert "os.chdir" not in source
    assert "Products:" not in source
    assert "pool=" not in source


def test_observed_assessment_notebook_is_advisor_only() -> None:
    """Notebook 02A must stop after writing a reviewed configuration."""
    source = _notebook_source(
        NOTEBOOK_DIRECTORY / "02A_Observed_Assessment_and_Config.ipynb"
    )

    assert "ObservationAdvisor.from_project" in source
    assert "advisor.select_target(" in source
    assert "advisor.inspect()" in source
    assert "advisor.summary()" in source
    assert "advisor.calibrate_flux_prior(" in source
    assert "advisor.propose_config(" in source
    assert "proposal.write(" in source
    assert "accept=True" in source
    assert 'starting_model = "single_sersic"' in source
    assert '"include_normalization": True' in source
    assert "reviewed_flux_prior = None" in source
    assert "proposal = None" in source
    assert "if proposal is None:" in source
    assert "flux_prior=reviewed_flux_prior" in source
    assert "flux_calibration=flux_calibration" not in source
    assert "operational diagnostics only" in source

    assert "BayMorphRunner" not in source
    assert "make_config_from_fits" not in source
    assert "runner.run(" not in source
    assert 'target_position = "center"' not in source


def test_observed_fit_notebook_is_runner_only() -> None:
    """Notebook 02B must consume the YAML without repeating assessment."""
    source = _notebook_source(NOTEBOOK_DIRECTORY / "02B_Observed_Fit.ipynb")

    assert "BayMorphRunner" in source
    assert "configs/observed_galaxy.yaml" in source
    assert "runner.run(diagnostics=True)" in source
    assert "fit.summary()" in source

    assert "ObservationAdvisor" not in source
    assert "select_target(" not in source
    assert "calibrate_flux_prior(" not in source
    assert "propose_config(" not in source
    assert "make_config_from_fits" not in source


def test_multi_observation_notebook_uses_public_joint_workflow() -> None:
    """The joint tutorial must use only the public multi-observation workflow."""
    source = _notebook_source(NOTEBOOK_DIRECTORY / "03_MultiObservation_JointFit.ipynb")

    assert "BayMorphRunner" in source
    assert "make_multi_observation_config" in source
    assert "make_multi_parameters" in source
    assert 'parameter_frame="sky"' in source
    assert '"q"' in source
    assert '"position_angle"' in source
    assert "shared_parameters" in source
    assert "shared=shared_parameters" in source
    assert "flux_prior_bounds" in source
    assert "observed_galaxy_g.fits" in source
    assert "validated astrometric registration" in source
    assert '"name": "nautilus"' in source
    assert "runner.run(" in source
    assert "diagnostics=True" in source

    assert "from baymorph.pipeline" not in source
    assert "from baymorph.fitting" not in source
    assert "from baymorph.likelihood" not in source
    assert "pool=" not in source
    assert "fitter=" not in source
    assert "n_processes=" not in source
    assert "find_repository_root" not in source
    assert "repository_root" not in source


def test_public_notebooks_use_repository_root_relative_paths() -> None:
    """Canonical notebooks must not depend on notebook-directory paths."""
    for name in EXPECTED_NOTEBOOKS:
        source = _notebook_source(NOTEBOOK_DIRECTORY / name)

        assert "../configs/" not in source
        assert "../data/" not in source
        assert "../runs" not in source


def test_public_notebooks_use_stable_user_facing_imports() -> None:
    """Tutorials must use only stable top-level BayMorph interfaces."""
    for name in EXPECTED_NOTEBOOKS:
        source = _notebook_source(NOTEBOOK_DIRECTORY / name)

        assert "from baymorph.pipeline" not in source
        assert "from baymorph.simulation" not in source
        assert "find_repository_root" not in source
        assert "repository_root" not in source


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
        source = _notebook_source(NOTEBOOK_DIRECTORY / name)

        for token in forbidden:
            assert token not in source


def test_public_notebooks_do_not_expose_fit_region() -> None:
    """Canonical workflows must use the complete prepared usable footprint."""
    for name in EXPECTED_NOTEBOOKS:
        source = _notebook_source(NOTEBOOK_DIRECTORY / name)

        assert "fit_region" not in source
        assert '"region"' not in source
