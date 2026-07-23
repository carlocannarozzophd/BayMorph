"""Tests for the high-level BayMorph workflow runner."""

from pathlib import Path

import pytest

from baymorph import BayMorphRunner



def make_runner_config(
    *,
    psf_type: str = "gaussian",
):
    """Return a minimal valid BayMorph configuration."""
    return {
        "object": {
            "name": "test_galaxy",
        },
        "data": {
            "image_path": "data/test_galaxy.fits",
        },
        "outputs": {
            "root": "outputs/test_galaxy",
        },
        "psf": {
            "type": psf_type,
            "gaussian": {
                "fwhm": 2.0,
            },
        },
    }


def test_runner_can_be_created_from_config():
    runner = BayMorphRunner.from_config(
        make_runner_config(),
        verbose=False,
    )

    assert runner.config["object"]["name"] == "test_galaxy"
    assert runner.output_dir == Path("outputs/test_galaxy")

    assert runner.image_preparation_products is None
    assert runner.psf_candidate_products is None
    assert runner.psf_products is None
    assert runner.fit_result is None
    assert runner.profile_bands is None


def test_runner_resolves_default_configuration():
    runner = BayMorphRunner(
        make_runner_config(),
        verbose=False,
    )

    assert runner.config["fit"]["sampler"]["name"] == "dynesty"
    assert runner.config["fit"]["likelihood"]["type"] == "gaussian"
    assert runner.config["diagnostics"]["corner"] is True


def test_runner_rejects_incomplete_configuration():
    with pytest.raises(
        ValueError,
        match="Missing required config sections",
    ):
        BayMorphRunner(
            {
                "object": {
                    "name": "test_galaxy",
                },
            },
            verbose=False,
        )


def test_runner_initializes_workflow_products_to_none():
    runner = BayMorphRunner(
        make_runner_config(),
        verbose=False,
    )

    assert runner.image is None
    assert runner.psf_candidates is None
    assert runner.psf is None
    assert runner.results is None
    assert runner.profile_bands is None


def test_runner_prepare_image_returns_and_stores_products(
    monkeypatch,
):
    expected_products = object()

    def fake_run_image_preparation(config):
        assert config["object"]["name"] == "test_galaxy"
        return expected_products

    monkeypatch.setattr(
        "baymorph.runner.run_image_preparation",
        fake_run_image_preparation,
    )

    runner = BayMorphRunner(
        make_runner_config(),
        verbose=False,
    )

    result = runner.prepare_image()

    assert result is expected_products
    assert runner.image is expected_products


def test_runner_select_psf_candidates_returns_and_stores_products(
    monkeypatch,
):
    expected_products = object()

    def fake_run_psf_candidate_selection(config):
        assert config["object"]["name"] == "test_galaxy"
        return expected_products

    monkeypatch.setattr(
        "baymorph.runner.run_psf_candidate_selection",
        fake_run_psf_candidate_selection,
    )

    runner = BayMorphRunner(
        make_runner_config(
            psf_type="empirical",
        ),
        verbose=False,
    )

    result = runner.select_psf_candidates()

    assert result is expected_products
    assert runner.psf_candidates is expected_products


def test_runner_build_psf_returns_and_stores_products(
    monkeypatch,
):
    expected_products = object()

    def fake_run_psf_construction(config):
        assert config["object"]["name"] == "test_galaxy"
        return expected_products

    monkeypatch.setattr(
        "baymorph.runner.run_psf_construction",
        fake_run_psf_construction,
    )

    runner = BayMorphRunner(
        make_runner_config(
            psf_type="empirical",
        ),
        verbose=False,
    )

    result = runner.build_psf()

    assert result is expected_products
    assert runner.psf is expected_products


def test_runner_fit_returns_and_stores_result(
    monkeypatch,
):
    expected_result = object()

    def fake_run_real_fit(config):
        assert config["object"]["name"] == "test_galaxy"
        return expected_result

    monkeypatch.setattr(
        "baymorph.runner.run_real_fit",
        fake_run_real_fit,
    )

    runner = BayMorphRunner(
        make_runner_config(),
        verbose=False,
    )

    result = runner.fit()

    assert result is expected_result
    assert runner.results is expected_result
    assert runner.profile_bands is None


def test_runner_fit_resets_existing_profile_bands(
    monkeypatch,
):
    expected_result = object()

    monkeypatch.setattr(
        "baymorph.runner.run_real_fit",
        lambda config: expected_result,
    )

    runner = BayMorphRunner(
        make_runner_config(),
        verbose=False,
    )

    runner.profile_bands = object()

    runner.fit()

    assert runner.profile_bands is None


def test_runner_diagnostics_require_fit():
    runner = BayMorphRunner(
        make_runner_config(),
        verbose=False,
    )

    with pytest.raises(
        RuntimeError,
        match=r"Run fit\(\) before corner\(\)",
    ):
        runner.corner()

    with pytest.raises(
        RuntimeError,
        match=r"Run fit\(\) before residual_maps\(\)",
    ):
        runner.residual_maps()

    with pytest.raises(
        RuntimeError,
        match=r"Run fit\(\) before radial_profile\(\)",
    ):
        runner.radial_profile()

    with pytest.raises(
        RuntimeError,
        match=r"Run fit\(\) before diagnostics\(\)",
    ):
        runner.diagnostics()


def test_runner_radial_profile_returns_and_stores_bands(
    monkeypatch,
):
    expected_bands = object()

    def fake_run_radial_profile_diagnostics(
        config,
        fit_result,
    ):
        assert config["object"]["name"] == "test_galaxy"
        assert fit_result is runner.fit_result
        return expected_bands

    monkeypatch.setattr(
        "baymorph.pipeline.run_radial_profile_diagnostics",
        fake_run_radial_profile_diagnostics,
    )

    runner = BayMorphRunner(
        make_runner_config(),
        verbose=False,
    )

    runner.fit_result = object()

    result = runner.radial_profile()

    assert result is expected_bands
    assert runner.profile_bands is expected_bands


def test_runner_run_with_gaussian_psf_skips_empirical_workflow(
    monkeypatch,
):
    calls = []

    monkeypatch.setattr(
        "baymorph.runner.run_image_preparation",
        lambda config: calls.append("prepare") or object(),
    )

    monkeypatch.setattr(
        "baymorph.runner.run_psf_candidate_selection",
        lambda config: calls.append("select") or object(),
    )

    monkeypatch.setattr(
        "baymorph.runner.run_psf_construction",
        lambda config: calls.append("build") or object(),
    )

    expected_result = object()

    monkeypatch.setattr(
        "baymorph.runner.run_real_fit",
        lambda config: calls.append("fit") or expected_result,
    )

    runner = BayMorphRunner(
        make_runner_config(
            psf_type="gaussian",
        ),
        verbose=False,
    )

    result = runner.run()

    assert result is expected_result
    assert calls == [
        "prepare",
        "fit",
    ]


def test_runner_run_with_empirical_psf_executes_full_workflow(
    monkeypatch,
):
    calls = []

    monkeypatch.setattr(
        "baymorph.runner.run_image_preparation",
        lambda config: calls.append("prepare") or object(),
    )

    monkeypatch.setattr(
        "baymorph.runner.run_psf_candidate_selection",
        lambda config: calls.append("select") or object(),
    )

    monkeypatch.setattr(
        "baymorph.runner.run_psf_construction",
        lambda config: calls.append("build") or object(),
    )

    expected_result = object()

    monkeypatch.setattr(
        "baymorph.runner.run_real_fit",
        lambda config: calls.append("fit") or expected_result,
    )

    runner = BayMorphRunner(
        make_runner_config(
            psf_type="empirical",
        ),
        verbose=False,
    )

    result = runner.run()

    assert result is expected_result
    assert calls == [
        "prepare",
        "select",
        "build",
        "fit",
    ]


def test_runner_run_can_generate_diagnostics(
    monkeypatch,
):
    expected_result = object()
    calls = []

    monkeypatch.setattr(
        "baymorph.runner.run_image_preparation",
        lambda config: object(),
    )

    monkeypatch.setattr(
        "baymorph.runner.run_real_fit",
        lambda config: expected_result,
    )

    runner = BayMorphRunner(
        make_runner_config(),
        verbose=False,
    )

    monkeypatch.setattr(
        runner,
        "diagnostics",
        lambda: calls.append("diagnostics"),
    )

    result = runner.run(
        diagnostics=True,
    )

    assert result is expected_result
    assert calls == [
        "diagnostics",
    ]


def test_runner_verbose_false_suppresses_status_messages(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        "baymorph.runner.run_image_preparation",
        lambda config: object(),
    )

    runner = BayMorphRunner(
        make_runner_config(),
        verbose=False,
    )

    runner.prepare_image()

    captured = capsys.readouterr()

    assert captured.out == ""


def test_runner_run_dir(tmp_path):
    """Runner should expose the object-specific output directory."""
    config = {
        "object": {
            "name": "tutorial_galaxy",
        },
        "data": {
            "image_path": str(
                tmp_path / "tutorial_galaxy.fits"
            ),
        },
        "outputs": {
            "root": str(
                tmp_path / "runs"
            ),
        },
    }

    runner = BayMorphRunner(config)

    assert runner.run_dir == (
        tmp_path
        / "runs"
        / "tutorial_galaxy"
    )