"""Tests for runner-driven simulated workflows."""

from __future__ import annotations

from pathlib import Path

import yaml

from baymorph import BayMorphRunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _simulated_config() -> dict:
    """Return the canonical simulated-galaxy configuration."""
    return yaml.safe_load(
        (
            PROJECT_ROOT
            / "configs"
            / "simulated_galaxy.yaml"
        ).read_text(
            encoding="utf-8"
        )
    )


def test_runner_simulate_returns_and_stores_observation(
    monkeypatch,
) -> None:
    """The public runner must expose the simulation product."""
    expected = object()

    monkeypatch.setattr(
        "baymorph.runner.run_simulation_from_config",
        lambda config: expected,
    )

    runner = BayMorphRunner(
        _simulated_config()
    )

    assert runner.simulation is None
    assert runner.simulate() is expected
    assert runner.simulation is expected


def test_runner_run_generates_configured_simulation_first(
    monkeypatch,
) -> None:
    """The all-in-one workflow must simulate before preparation and fit."""
    runner = BayMorphRunner(
        _simulated_config()
    )

    events: list[str] = []
    expected_result = object()

    monkeypatch.setattr(
        runner,
        "simulate",
        lambda: events.append(
            "simulate"
        ),
    )
    monkeypatch.setattr(
        runner,
        "prepare_image",
        lambda: events.append(
            "prepare"
        ),
    )

    def fake_fit():
        events.append(
            "fit"
        )
        return expected_result

    monkeypatch.setattr(
        runner,
        "fit",
        fake_fit,
    )

    result = runner.run()

    assert result is expected_result
    assert events == [
        "simulate",
        "prepare",
        "fit",
    ]


def test_runner_anchors_standard_project_configuration(
    tmp_path,
    monkeypatch,
) -> None:
    """Standard config paths must resolve from the project root."""
    project_root = (
        tmp_path
        / "analysis"
    )
    config_directory = (
        project_root
        / "configs"
    )
    notebook_directory = (
        project_root
        / "notebooks"
    )

    config_directory.mkdir(
        parents=True
    )
    notebook_directory.mkdir()

    config_file = (
        config_directory
        / "simulated_galaxy.yaml"
    )
    config_file.write_text(
        yaml.safe_dump(
            _simulated_config(),
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(
        notebook_directory
    )

    runner = BayMorphRunner(
        Path(
            "configs/simulated_galaxy.yaml"
        )
    )

    assert runner.project_root == project_root.resolve()
    assert runner.config_path == config_file.resolve()
    assert runner.config["outputs"]["root"] == str(
        (
            project_root
            / "runs"
        ).resolve()
    )
    assert runner.config["data"]["image_path"] == str(
        (
            project_root
            / "runs"
            / "simulated_galaxy"
            / "input"
            / "simulated_galaxy.fits"
        ).resolve()
    )
