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

    assert runner.config["fit"]["sampler"]["name"] == "pocomc"
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

def test_runner_diagnostics_honor_enabled_toggles(
    monkeypatch,
):
    """The diagnostic orchestrator must run only enabled products."""
    calls = []
    expected_bands = object()

    config = make_runner_config()
    config["diagnostics"] = {
        "corner": True,
        "residual": False,
        "radial_profile": True,
    }

    runner = BayMorphRunner(
        config,
        verbose=False,
    )
    runner.fit_result = object()

    monkeypatch.setattr(
        runner,
        "corner",
        lambda: calls.append("corner"),
    )
    monkeypatch.setattr(
        runner,
        "residual_maps",
        lambda: calls.append("residual"),
    )
    monkeypatch.setattr(
        runner,
        "radial_profile",
        lambda: calls.append("radial") or expected_bands,
    )

    result = runner.diagnostics()

    assert result is expected_bands
    assert calls == [
        "corner",
        "radial",
    ]


def test_runner_diagnostics_can_disable_all_products(
    monkeypatch,
):
    """Disabling all diagnostics must perform no diagnostic work."""
    calls = []

    config = make_runner_config()
    config["diagnostics"] = {
        "corner": False,
        "residual": False,
        "radial_profile": False,
    }

    runner = BayMorphRunner(
        config,
        verbose=False,
    )
    runner.fit_result = object()
    runner.profile_bands = object()

    monkeypatch.setattr(
        runner,
        "corner",
        lambda: calls.append("corner"),
    )
    monkeypatch.setattr(
        runner,
        "residual_maps",
        lambda: calls.append("residual"),
    )
    monkeypatch.setattr(
        runner,
        "radial_profile",
        lambda: calls.append("radial"),
    )

    result = runner.diagnostics()

    assert result is None
    assert runner.profile_bands is None
    assert calls == []


def test_runner_diagnostics_can_run_residuals_only(
    monkeypatch,
):
    """Each diagnostic toggle must operate independently."""
    calls = []

    config = make_runner_config()
    config["diagnostics"] = {
        "corner": False,
        "residual": True,
        "radial_profile": False,
    }

    runner = BayMorphRunner(
        config,
        verbose=False,
    )
    runner.fit_result = object()

    monkeypatch.setattr(
        runner,
        "corner",
        lambda: calls.append("corner"),
    )
    monkeypatch.setattr(
        runner,
        "residual_maps",
        lambda: calls.append("residual"),
    )
    monkeypatch.setattr(
        runner,
        "radial_profile",
        lambda: calls.append("radial"),
    )

    result = runner.diagnostics()

    assert result is None
    assert calls == ["residual"]


def _multi_observation_runner_config():
    """Return a minimal structural multi-observation configuration."""
    return {
        "observations": {
            "g": {
                "config_path": "g.yaml",
            },
            "r": {
                "config_path": "r.yaml",
            },
        },
        "fit": {
            "models": [
                "single_sersic",
            ],
            "multiband": {
                "parameters": {},
            },
        },
    }


def test_runner_identifies_multi_observation_workflow():
    runner = BayMorphRunner(
        _multi_observation_runner_config(),
        verbose=False,
    )

    assert runner.workflow_mode == "multi_observation"
    assert runner.is_multi_observation is True


def test_runner_rejects_empty_multi_observation_mapping():
    with pytest.raises(
        ValueError,
        match="non-empty 'observations' mapping",
    ):
        BayMorphRunner(
            {
                "observations": {},
            },
            verbose=False,
        )


def test_runner_prepares_all_multi_observation_images(
    monkeypatch,
):
    observation_configs = {
        "g": {
            "object": {
                "name": "galaxy_g",
            },
        },
        "r": {
            "object": {
                "name": "galaxy_r",
            },
        },
    }
    expected = {
        "g": object(),
        "r": object(),
    }

    monkeypatch.setattr(
        "baymorph.runner.resolve_multiband_observation_configs",
        lambda config, base_dir=None: observation_configs,
    )

    def fake_prepare(config):
        band = config["object"]["name"].rsplit("_", 1)[-1]
        return expected[band]

    monkeypatch.setattr(
        "baymorph.runner.run_image_preparation",
        fake_prepare,
    )

    runner = BayMorphRunner(
        _multi_observation_runner_config(),
        verbose=False,
    )

    result = runner.prepare_image()

    assert result == expected
    assert runner.image == expected


def test_runner_selects_psf_candidates_for_all_observations(
    monkeypatch,
):
    observation_configs = {
        "g": {
            "object": {
                "name": "galaxy_g",
            },
        },
        "r": {
            "object": {
                "name": "galaxy_r",
            },
        },
    }
    expected = {
        "g": object(),
        "r": object(),
    }

    monkeypatch.setattr(
        "baymorph.runner.resolve_multiband_observation_configs",
        lambda config, base_dir=None: observation_configs,
    )

    def fake_select(config):
        band = config["object"]["name"].rsplit("_", 1)[-1]
        return expected[band]

    monkeypatch.setattr(
        "baymorph.runner.run_psf_candidate_selection",
        fake_select,
    )

    runner = BayMorphRunner(
        _multi_observation_runner_config(),
        verbose=False,
    )

    result = runner.select_psf_candidates()

    assert result == expected
    assert runner.psf_candidates == expected


def test_runner_builds_psfs_for_all_observations(
    monkeypatch,
):
    observation_configs = {
        "g": {
            "object": {
                "name": "galaxy_g",
            },
        },
        "r": {
            "object": {
                "name": "galaxy_r",
            },
        },
    }
    expected = {
        "g": object(),
        "r": object(),
    }

    monkeypatch.setattr(
        "baymorph.runner.resolve_multiband_observation_configs",
        lambda config, base_dir=None: observation_configs,
    )

    def fake_build(config):
        band = config["object"]["name"].rsplit("_", 1)[-1]
        return expected[band]

    monkeypatch.setattr(
        "baymorph.runner.run_psf_construction",
        fake_build,
    )

    runner = BayMorphRunner(
        _multi_observation_runner_config(),
        verbose=False,
    )

    result = runner.build_psf()

    assert result == expected
    assert runner.psf == expected


def test_runner_runs_multi_observation_fit(
    monkeypatch,
):
    expected = object()
    captured = {}

    runner = BayMorphRunner(
        _multi_observation_runner_config(),
        verbose=False,
    )

    def fake_fit(config):
        captured["config"] = config
        return expected

    monkeypatch.setattr(
        "baymorph.runner.run_multi_observation_fit_from_config",
        fake_fit,
    )

    result = runner.fit()

    assert result is expected
    assert runner.results is expected
    assert captured["config"] is runner.config


def test_runner_multi_observation_fit_preserves_config_path(
    monkeypatch,
    tmp_path,
):
    config_path = tmp_path / "joint.yaml"

    import yaml

    config_path.write_text(
        yaml.safe_dump(
            _multi_observation_runner_config(),
            sort_keys=False,
        )
    )

    expected = object()
    captured = {}

    def fake_fit(config):
        captured["config"] = config
        return expected

    monkeypatch.setattr(
        "baymorph.runner.run_multi_observation_fit_from_config",
        fake_fit,
    )

    runner = BayMorphRunner(
        config_path,
        verbose=False,
    )

    result = runner.fit()

    assert result is expected
    assert captured["config"] == config_path


def test_runner_runs_multi_observation_workflow(
    monkeypatch,
):
    observation_configs = {
        "g": {
            "object": {
                "name": "galaxy_g",
            },
            "psf": {
                "type": "empirical",
            },
        },
        "r": {
            "object": {
                "name": "galaxy_r",
            },
            "psf": {
                "type": "empirical",
            },
        },
    }

    expected_fit = object()
    calls = []

    monkeypatch.setattr(
        "baymorph.runner.resolve_multiband_observation_configs",
        lambda config, base_dir=None: observation_configs,
    )

    monkeypatch.setattr(
        "baymorph.runner.run_image_preparation",
        lambda config: calls.append(
            (
                "prepare",
                config["object"]["name"],
            )
        )
        or object(),
    )

    monkeypatch.setattr(
        "baymorph.runner.run_psf_candidate_selection",
        lambda config: calls.append(
            (
                "select",
                config["object"]["name"],
            )
        )
        or object(),
    )

    monkeypatch.setattr(
        "baymorph.runner.run_psf_construction",
        lambda config: calls.append(
            (
                "build",
                config["object"]["name"],
            )
        )
        or object(),
    )

    runner = BayMorphRunner(
        _multi_observation_runner_config(),
        verbose=False,
    )

    monkeypatch.setattr(
        runner,
        "fit",
        lambda: calls.append(
            ("fit", None)
        )
        or expected_fit,
    )

    result = runner.run()

    assert result is expected_fit
    assert calls == [
        ("prepare", "galaxy_g"),
        ("prepare", "galaxy_r"),
        ("select", "galaxy_g"),
        ("select", "galaxy_r"),
        ("build", "galaxy_g"),
        ("build", "galaxy_r"),
        ("fit", None),
    ]


def test_runner_multi_observation_run_skips_gaussian_psf(
    monkeypatch,
):
    observation_configs = {
        "g": {
            "object": {
                "name": "galaxy_g",
            },
            "psf": {
                "type": "empirical",
            },
        },
        "r": {
            "object": {
                "name": "galaxy_r",
            },
            "psf": {
                "type": "gaussian",
            },
        },
    }

    selected = []
    built = []

    monkeypatch.setattr(
        "baymorph.runner.resolve_multiband_observation_configs",
        lambda config, base_dir=None: observation_configs,
    )

    monkeypatch.setattr(
        "baymorph.runner.run_image_preparation",
        lambda config: object(),
    )

    monkeypatch.setattr(
        "baymorph.runner.run_psf_candidate_selection",
        lambda config: selected.append(
            config["object"]["name"]
        )
        or object(),
    )

    monkeypatch.setattr(
        "baymorph.runner.run_psf_construction",
        lambda config: built.append(
            config["object"]["name"]
        )
        or object(),
    )

    runner = BayMorphRunner(
        _multi_observation_runner_config(),
        verbose=False,
    )

    monkeypatch.setattr(
        runner,
        "fit",
        lambda: object(),
    )

    runner.run()

    assert selected == [
        "galaxy_g",
    ]
    assert built == [
        "galaxy_g",
    ]


def test_runner_multi_observation_run_calls_fit_once(
    monkeypatch,
):
    observation_configs = {
        "g": {
            "psf": {
                "type": "gaussian",
            },
        },
        "r": {
            "psf": {
                "type": "gaussian",
            },
        },
    }

    monkeypatch.setattr(
        "baymorph.runner.resolve_multiband_observation_configs",
        lambda config, base_dir=None: observation_configs,
    )

    monkeypatch.setattr(
        "baymorph.runner.run_image_preparation",
        lambda config: object(),
    )

    runner = BayMorphRunner(
        _multi_observation_runner_config(),
        verbose=False,
    )

    calls = []

    monkeypatch.setattr(
        runner,
        "fit",
        lambda: calls.append("fit") or object(),
    )

    runner.run()

    assert calls == [
        "fit",
    ]


def test_runner_multi_observation_run_can_generate_diagnostics(
    monkeypatch,
):
    """Joint execution should run configured diagnostics after one fit."""
    observation_configs = {
        "g": {
            "psf": {
                "type": "gaussian",
            },
        },
        "r": {
            "psf": {
                "type": "gaussian",
            },
        },
    }

    monkeypatch.setattr(
        "baymorph.runner.resolve_multiband_observation_configs",
        lambda config, base_dir=None: observation_configs,
    )

    monkeypatch.setattr(
        "baymorph.runner.run_image_preparation",
        lambda config: object(),
    )

    runner = BayMorphRunner(
        _multi_observation_runner_config(),
        verbose=False,
    )

    expected_result = object()
    calls = []

    monkeypatch.setattr(
        runner,
        "fit",
        lambda: calls.append(
            "fit"
        )
        or expected_result,
    )

    monkeypatch.setattr(
        runner,
        "diagnostics",
        lambda: calls.append(
            "diagnostics"
        ),
    )

    result = runner.run(
        diagnostics=True,
    )

    assert result is expected_result
    assert calls == [
        "fit",
        "diagnostics",
    ]


def test_runner_multi_observation_residual_maps_use_joint_result(
    monkeypatch,
):
    """Runner should expose observation-specific residual diagnostics."""
    runner = BayMorphRunner(
        _multi_observation_runner_config(),
        verbose=False,
    )

    fit_result = object()
    runner.fit_result = fit_result

    captured = {}

    def fake_run_residual_map_diagnostics(
        config,
        result,
    ):
        captured["config"] = config
        captured["result"] = result

    monkeypatch.setattr(
        "baymorph.pipeline.run_residual_map_diagnostics",
        fake_run_residual_map_diagnostics,
    )

    runner.residual_maps()

    assert captured["config"] is runner.config
    assert captured["result"] is fit_result


def test_runner_multi_observation_radial_profile_stores_observation_bands(
    monkeypatch,
):
    """Runner should retain observation-indexed joint radial-profile bands."""
    runner = BayMorphRunner(
        _multi_observation_runner_config(),
        verbose=False,
    )

    fit_result = object()
    runner.fit_result = fit_result

    expected_bands = {
        "g": object(),
        "r": object(),
    }

    captured = {}

    def fake_run_radial_profile_diagnostics(
        config,
        result,
    ):
        captured["config"] = config
        captured["result"] = result
        return expected_bands

    monkeypatch.setattr(
        "baymorph.pipeline.run_radial_profile_diagnostics",
        fake_run_radial_profile_diagnostics,
    )

    result = runner.radial_profile()

    assert result is expected_bands
    assert runner.profile_bands is expected_bands
    assert captured["config"] is runner.config
    assert captured["result"] is fit_result


def test_runner_multi_observation_diagnostics_default_to_all_products(
    monkeypatch,
):
    """Missing joint diagnostic settings should enable standard diagnostics."""
    runner = BayMorphRunner(
        _multi_observation_runner_config(),
        verbose=False,
    )

    runner.fit_result = object()

    expected_bands = {
        "g": object(),
        "r": object(),
    }

    calls = []

    monkeypatch.setattr(
        runner,
        "corner",
        lambda: calls.append(
            "corner"
        ),
    )

    monkeypatch.setattr(
        runner,
        "residual_maps",
        lambda: calls.append(
            "residual"
        ),
    )

    monkeypatch.setattr(
        runner,
        "radial_profile",
        lambda: calls.append(
            "radial"
        )
        or expected_bands,
    )

    result = runner.diagnostics()

    assert result is expected_bands
    assert calls == [
        "corner",
        "residual",
        "radial",
    ]


def test_runner_multi_observation_diagnostics_honor_toggles(
    monkeypatch,
):
    """Joint diagnostics should honor independent output toggles."""
    config = _multi_observation_runner_config()

    config["diagnostics"] = {
        "corner": False,
        "residual": True,
        "radial_profile": {},
    }

    runner = BayMorphRunner(
        config,
        verbose=False,
    )

    runner.fit_result = object()

    expected_bands = {
        "g": object(),
        "r": object(),
    }

    calls = []

    monkeypatch.setattr(
        runner,
        "corner",
        lambda: calls.append(
            "corner"
        ),
    )

    monkeypatch.setattr(
        runner,
        "residual_maps",
        lambda: calls.append(
            "residual"
        ),
    )

    monkeypatch.setattr(
        runner,
        "radial_profile",
        lambda: calls.append(
            "radial"
        )
        or expected_bands,
    )

    result = runner.diagnostics()

    assert result is expected_bands
    assert calls == [
        "residual",
        "radial",
    ]


def test_runner_preserves_arbitrary_observation_identifiers(
    monkeypatch,
):
    """Observation identifiers must be opaque user-defined names."""
    observation_configs = {
        "hst_f814w_epoch1": {
            "object": {
                "name": "galaxy_hst",
            },
        },
        "jwst_f444w_epoch2": {
            "object": {
                "name": "galaxy_jwst",
            },
        },
    }

    monkeypatch.setattr(
        "baymorph.runner.resolve_multiband_observation_configs",
        lambda config, base_dir=None: observation_configs,
    )

    seen = []

    def fake_prepare(config):
        seen.append(
            config["object"]["name"]
        )
        return object()

    monkeypatch.setattr(
        "baymorph.runner.run_image_preparation",
        fake_prepare,
    )

    runner = BayMorphRunner(
        _multi_observation_runner_config(),
        verbose=False,
    )

    products = runner.prepare_image()

    assert tuple(products) == (
        "hst_f814w_epoch1",
        "jwst_f444w_epoch2",
    )

    assert seen == [
        "galaxy_hst",
        "galaxy_jwst",
    ]


def test_runner_multi_observation_corner_uses_joint_diagnostic(
    monkeypatch,
):
    """Multi-observation runner should expose the joint corner diagnostic."""
    runner = BayMorphRunner(
        _multi_observation_runner_config(),
        verbose=False,
    )

    fit_result = object()
    runner.fit_result = fit_result

    captured = {}

    def fake_run_corner_diagnostics(
        config,
        result,
    ):
        captured["config"] = config
        captured["result"] = result

    monkeypatch.setattr(
        "baymorph.pipeline.run_corner_diagnostics",
        fake_run_corner_diagnostics,
    )

    runner.corner()

    assert captured["config"] is runner.config
    assert captured["result"] is fit_result


