"""Tests for the canonical public BayMorph notebooks."""

from __future__ import annotations

import json
import subprocess
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
# Executed examples: tracked, not packaged, not copied by baymorph init.
EXAMPLE_NOTEBOOKS: set[Path] = {
    Path("examples") / "roman_cases.ipynb",
    Path("examples") / "roman_isim_fits.ipynb",
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


def _tracked_notebooks() -> set[Path]:
    """Return the notebooks tracked by Git, relative to the project root."""
    result = subprocess.run(
        ["git", "ls-files", "*.ipynb"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    return {Path(line) for line in result.stdout.splitlines() if line}


def test_only_public_and_packaged_notebooks_exist() -> None:
    """The repository holds the public notebooks, their packaged copies and the
    executed examples only.

    Only tracked files count: notebooks that a developer keeps on disk
    outside the repository, or copies left in a build directory, are not
    part of what BayMorph publishes.
    """
    expected_public = {Path("notebooks") / name for name in EXPECTED_NOTEBOOKS}
    expected_packaged = {
        Path("baymorph/resources/notebooks") / name for name in EXPECTED_NOTEBOOKS
    }

    assert _tracked_notebooks() == (
        expected_public | expected_packaged | EXAMPLE_NOTEBOOKS
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
    """02A guides the advisor to a reviewed configuration; it never fits.

    The advisor prepares the image and builds its PSF, the fixed inputs
    that the fit of 02B reuses, before it proposes the priors of the chosen
    models with the same options.
    """
    source = _notebook_source(NOTEBOOK_DIRECTORY / ASSESSMENT_NOTEBOOK)

    for step in (
        "ObservationAdvisor.from_project(",
        "advisor.header_report()",
        "advisor.select_target(",
        "advisor.measure_fixed_inputs(**stage_options)",
        "advisor.propose_config(",
        "models=models",
        "fixed_inputs=fixed_inputs",
        "**stage_options",
        "proposal.write(accept=True)",
    ):
        assert step in source

    for forbidden in (
        "BayMorphRunner",
        "calibrate_flux_prior",
        "overwrite=True",
        ".fit(",
    ):
        assert forbidden not in source


def test_observed_fit_notebook_shows_every_fitted_model() -> None:
    """02B must show the posterior, residuals and evidence of each model."""
    source = _notebook_source(NOTEBOOK_DIRECTORY / OBSERVED_FIT_NOTEBOOK)

    assert 'runner.config["fit"]["models"]' in source
    assert "for model_name, fit in fits.items():" in source
    assert "logZ - best" in source


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
