"""Tests for the canonical public BayMorph notebooks."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIRECTORY = PROJECT_ROOT / "notebooks"

ASSESSMENT_NOTEBOOK = "02A_Observed_Assessment_and_Config.ipynb"
OBSERVED_FIT_NOTEBOOK = "02B_Observed_Fit.ipynb"

EXPECTED_NOTEBOOKS: set[str] = {
    "01_Simulated_SingleBand.ipynb",
    ASSESSMENT_NOTEBOOK,
    OBSERVED_FIT_NOTEBOOK,
}

# Research campaigns under validation/ are not tutorials and are not packaged.
# They keep the hygiene contract but may use internal APIs and plot directly,
# so they are listed separately rather than relaxing the public-notebook rules.
EXPECTED_VALIDATION_NOTEBOOKS: set[str] = {
    "cv_source_identification/cv_methods_on_one_image.ipynb",
    "repeated_noise_recovery/repeated_noise_recovery.ipynb",
    "structure_informed_evidence/nautilus_vs_model_informed.ipynb",
}


def _load_notebook(
    path: Path,
) -> dict:
    """Load one notebook JSON document."""
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def _notebook_source(
    path: Path,
) -> str:
    """Return concatenated code and Markdown source from one notebook."""
    notebook = _load_notebook(
        path
    )

    return "\n".join(
        "".join(
            cell.get(
                "source",
                [],
            )
        )
        for cell in notebook.get("cells", [])
    )


def test_public_notebook_set_is_exact() -> None:
    """The repository exposes exactly the canonical public notebooks."""
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
    expected_validation = {
        Path("validation") / name for name in EXPECTED_VALIDATION_NOTEBOOKS
    }

    assert notebooks == (
        expected_public | expected_packaged | expected_validation
    )


def test_validation_notebooks_are_not_packaged() -> None:
    """Research campaigns must stay out of the distributed package."""
    packaged = PROJECT_ROOT / "baymorph" / "resources" / "notebooks"

    for name in EXPECTED_VALIDATION_NOTEBOOKS:
        assert not (packaged / Path(name).name).is_file()


def test_validation_notebooks_are_clean_and_code_cells_compile() -> None:
    """Validation notebooks must be committed unexecuted and be valid."""
    for name in EXPECTED_VALIDATION_NOTEBOOKS:
        path = PROJECT_ROOT / "validation" / name

        assert path.is_file()

        notebook = _load_notebook(
            path
        )

        assert notebook["nbformat"] == 4

        for index, cell in enumerate(
            notebook["cells"]
        ):
            if cell["cell_type"] != "code":
                continue

            assert cell.get(
                "execution_count"
            ) is None
            assert cell.get(
                "outputs",
                [],
            ) == []

            compile(
                "".join(
                    cell.get(
                        "source",
                        [],
                    )
                ),
                f"{path}:cell-{index}",
                "exec",
            )


def test_public_notebooks_avoid_parent_relative_paths() -> None:
    """Canonical notebooks must not depend on parent-directory traversal."""
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


def test_public_notebooks_are_clean_and_code_cells_compile() -> None:
    """Canonical notebooks must contain clean, syntactically valid cells."""
    for name in EXPECTED_NOTEBOOKS:
        path = NOTEBOOK_DIRECTORY / name
        notebook = _load_notebook(
            path
        )

        assert notebook["nbformat"] == 4

        for index, cell in enumerate(
            notebook["cells"]
        ):
            if cell["cell_type"] != "code":
                continue

            assert cell.get(
                "execution_count"
            ) is None
            assert cell.get(
                "outputs",
                [],
            ) == []

            compile(
                "".join(
                    cell.get(
                        "source",
                        [],
                    )
                ),
                f"{path}:cell-{index}",
                "exec",
            )


def test_public_notebooks_never_prompt_for_input() -> None:
    """Tutorials must run linearly without blocking on ``input()``."""
    for name in EXPECTED_NOTEBOOKS:
        source = _notebook_source(NOTEBOOK_DIRECTORY / name)

        assert "input(" not in source


def test_assessment_notebook_only_writes_a_reviewed_configuration() -> None:
    """02A inspects and configures; it never prepares, builds a PSF, or fits."""
    source = _notebook_source(NOTEBOOK_DIRECTORY / ASSESSMENT_NOTEBOOK)

    assert "ObservationAdvisor.from_project(" in source
    assert "advisor.select_target(" in source
    assert "advisor.propose_config(" in source
    assert "proposal.write(accept=True)" in source

    for forbidden in (
        "BayMorphRunner",
        "calibrate_flux_prior",
        "overwrite=True",
    ):
        assert forbidden not in source


def test_observed_fit_notebook_runs_the_pipeline_from_the_written_yaml() -> None:
    """02B consumes the 02A YAML and never rebuilds the configuration."""
    assessment = _notebook_source(NOTEBOOK_DIRECTORY / ASSESSMENT_NOTEBOOK)
    source = _notebook_source(NOTEBOOK_DIRECTORY / OBSERVED_FIT_NOTEBOOK)

    for stage in (
        "runner.prepare_image()",
        "runner.select_psf_candidates()",
        "runner.build_psf()",
        "runner.fit()",
        "runner.diagnostics()",
    ):
        assert stage in source

    for forbidden in (
        "ObservationAdvisor",
        "propose_config",
        "make_config_from_fits",
    ):
        assert forbidden not in source

    assert 'analysis_name = "galaxy_r"' in assessment
    assert 'config_path = "configs/galaxy_r.yaml"' in source
