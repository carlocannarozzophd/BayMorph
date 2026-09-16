"""Tests for the canonical public BayMorph notebooks."""

from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from baymorph.pipeline.config import resolve_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIRECTORY = PROJECT_ROOT / "notebooks"

EXPECTED_NOTEBOOKS = {
    "01_Simulated_SingleBand.ipynb",
    "02A_Observed_Assessment_and_Config.ipynb",
    "02B_Observed_Fit.ipynb",
    "03_MultiObservation_JointFit.ipynb",
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


def _notebook_cell_source(
    path: Path,
    cell_id: str,
) -> str:
    """Return the source of one notebook cell selected by stable ID."""
    notebook = _load_notebook(
        path
    )

    for cell in notebook["cells"]:
        if cell.get("id") == cell_id:
            return "".join(
                cell.get(
                    "source",
                    [],
                )
            )

    raise AssertionError(
        f"Notebook cell {cell_id!r} was not found in {path}."
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
    """The simulated tutorial must select a sampler before writing YAML."""
    source = _notebook_source(NOTEBOOK_DIRECTORY / "01_Simulated_SingleBand.ipynb")

    assert "BayMorphRunner" in source
    assert "configs/simulated_galaxy.yaml" in source
    assert 'input("Choose one sampler: ")' in source
    assert "yaml.safe_load" in source
    assert "yaml.safe_dump" in source
    assert "config_path.write_text(" in source
    assert 'confirmation != "write"' in source
    assert '"checkpoint_path": "auto"' in source
    assert '"checkpoint_mode": checkpoint_mode' in source
    assert '"comparison"] = {' in source
    assert '"enabled": False' in source
    assert 'inference.evidence is None' in source
    assert "posterior-only backend" in source
    assert "inference.effective_sample_size" in source

    for sampler_name in (
        "nautilus",
        "dynesty-static",
        "dynesty-dynamic",
        "pocomc",
        "emcee",
    ):
        assert sampler_name in source

    assert "runner.run(diagnostics=True)" in source
    assert "runner.simulate()" in source
    assert "runner.prepare_image()" in source
    assert "runner.fit()" in source
    assert "requires no empirical PSF" in source
    assert "runner.simulation.image" in source
    assert "fit.posterior.summary(parameter)" in source
    assert "recovery_table" in source

    assert "make_config_from_fits" not in source
    assert "make_observation" not in source
    assert "SyntheticSersic" not in source
    assert "project_root" not in source
    assert "os.chdir" not in source
    assert "Products:" not in source
    assert "pool=" not in source


@pytest.mark.parametrize(
    (
        "responses",
        "expected_name",
        "expected_mode",
        "expected_checkpoint_mode",
    ),
    [
        (("nautilus",), "nautilus", None, None),
        (("dynesty-static",), "dynesty", "static", None),
        (("dynesty-dynamic",), "dynesty", "dynamic", None),
        (("pocomc",), "pocomc", None, None),
        (("emcee", "fresh", "1000"), "emcee", None, "fresh"),
        (("emcee", "resume", "3000"), "emcee", None, "resume"),
    ],
)
def test_simulated_notebook_builds_valid_yaml_for_each_sampler(
    responses,
    expected_name,
    expected_mode,
    expected_checkpoint_mode,
) -> None:
    """Every interactive choice must produce a resolvable sampler YAML."""
    notebook_path = (
        NOTEBOOK_DIRECTORY
        / "01_Simulated_SingleBand.ipynb"
    )

    class TemplateRunner:
        def __init__(
            self,
            request,
            *,
            verbose,
        ) -> None:
            assert verbose is False
            self.config_path = (
                PROJECT_ROOT
                / request
            )

    response_iterator = iter(
        responses
    )
    namespace = {
        "BayMorphRunner": TemplateRunner,
        "Path": Path,
        "deepcopy": deepcopy,
        "input": lambda _: next(
            response_iterator
        ),
        "yaml": yaml,
    }

    for cell_id in (
        "sim-select",
        "sim-build-yaml",
    ):
        source = _notebook_cell_source(
            notebook_path,
            cell_id,
        )
        exec(
            compile(
                source,
                f"{notebook_path}:{cell_id}",
                "exec",
            ),
            namespace,
        )

    generated = yaml.safe_load(
        namespace["yaml_text"]
    )
    resolved = resolve_config(
        generated
    )
    run_key = responses[0].replace(
        "-",
        "_",
    )

    assert generated == namespace["selected_config"]
    assert resolved["object"]["name"] == (
        f"simulated_galaxy_{run_key}"
    )
    assert run_key in resolved["data"]["image_path"]
    assert namespace["config_path"].name == (
        f"simulated_galaxy_{run_key}.yaml"
    )
    assert resolved["fit"]["comparison"]["enabled"] is False

    sampler = resolved["fit"]["sampler"]
    assert sampler["name"] == expected_name

    if expected_mode is None:
        assert "mode" not in sampler
    else:
        assert sampler["mode"] == expected_mode

    if expected_name == "emcee":
        assert sampler["init"]["checkpoint_path"] == "auto"
        assert sampler["init"]["checkpoint_mode"] == (
            expected_checkpoint_mode
        )
        assert sampler["run"]["max_steps"] == int(
            responses[-1]
        )
    else:
        assert "checkpoint_path" not in sampler["init"]


def test_observed_assessment_notebook_is_advisor_only() -> None:
    """Notebook 02A must select a sampler and stop after writing YAML."""
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
    assert 'input("Choose one sampler: ")' in source
    assert '"sampler": sampler_configurations[sampler_choice]' in source
    assert '"checkpoint_path": "auto"' in source
    assert "posterior only; no evidence" in source
    assert "print(proposal.summary())" in source
    assert "yaml.safe_dump(" in source
    assert "proposal.config" in source

    for sampler_name in (
        "nautilus",
        "dynesty-static",
        "dynesty-dynamic",
        "pocomc",
        "emcee",
    ):
        assert sampler_name in source

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
    assert 'sampler_config = runner.config["fit"]["sampler"]' in source
    assert "yaml.safe_dump(" in source
    assert "sampler_config," in source
    assert "inference.evidence is None" in source
    assert "inference.effective_sample_size" in source
    assert "checkpoint_mode: fresh" in source
    assert "checkpoint_mode" in source
    assert "runner.prepare_image()" in source
    assert "runner.build_psf()" in source
    assert "runner.fit()" in source

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
    assert "workspace_directory = Path(" in source
    assert 'outputs_root = workspace_directory / "runs"' in source
    assert "outputs_root=outputs_root" in source
    assert "validated astrometric registration" in source
    assert '"name": "nautilus"' in source
    assert "result.summary()" in source
    assert "managed" in source
    assert "single-observation" in source
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


def test_multi_observation_notebook_builds_absolute_paths() -> None:
    """Joint products must remain valid outside the launch directory."""
    path = (
        NOTEBOOK_DIRECTORY
        / "03_MultiObservation_JointFit.ipynb"
    )
    data_source = _notebook_cell_source(
        path,
        "joint-data",
    )
    data_tree = ast.parse(
        data_source
    )
    assignment_tree = ast.Module(
        body=[
            node
            for node in data_tree.body
            if isinstance(
                node,
                ast.Assign,
            )
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(
        assignment_tree
    )
    namespace = {
        "Path": Path,
    }
    exec(
        compile(
            assignment_tree,
            f"{path}:joint-data-assignments",
            "exec",
        ),
        namespace,
    )

    assert namespace[
        "workspace_directory"
    ].is_absolute()
    assert all(
        observation_path.is_absolute()
        for observation_path in namespace[
            "observations"
        ].values()
    )

    yaml_source = _notebook_cell_source(
        path,
        "joint-yaml",
    )
    path_setup, separator, _ = yaml_source.partition(
        "joint_config = make_multi_observation_config("
    )
    assert separator
    exec(
        compile(
            path_setup,
            f"{path}:joint-yaml-paths",
            "exec",
        ),
        namespace,
    )

    for name in (
        "outputs_root",
        "config_dir",
        "joint_config_path",
    ):
        assert namespace[
            name
        ].is_absolute()


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
