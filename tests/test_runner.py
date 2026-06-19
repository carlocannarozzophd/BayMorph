import pytest

from baymorph import BayMorphRunner


def test_runner_can_be_created_from_config():
    runner = BayMorphRunner.from_config(
        {
            "object_name": "test_galaxy",
            "output_dir": "outputs/test_galaxy",
        }
    )

    assert runner.config["object_name"] == "test_galaxy"
    assert str(runner.output_dir) == "outputs/test_galaxy"
    assert runner.image_preparation_products is None
    assert runner.psf_candidate_products is None
    assert runner.psf_products is None
    assert runner.fit_result is None
    assert runner.profile_bands is None

def test_runner_output_dir_can_be_missing():
    runner = BayMorphRunner.from_config(
        {
            "object_name": "test_galaxy",
        }
    )

    assert runner.output_dir is None


def test_runner_step_methods_are_not_implemented_yet():
    runner = BayMorphRunner.from_config(
        {
            "object_name": "test_galaxy",
        }
    )

    with pytest.raises(NotImplementedError):
        runner.posterior_profiles()


def test_runner_initializes_workflow_products_to_none():
    runner = BayMorphRunner.from_config(
        {
            "object_name": "test_galaxy",
        }
    )

    assert runner.image_preparation_products is None
    assert runner.psf_candidate_products is None
    assert runner.psf_products is None
    assert runner.fit_result is None
    assert runner.profile_bands is None


def test_runner_prepare_image_stores_products(monkeypatch):
    expected_products = object()

    def fake_run_image_preparation(config):
        assert config["object_name"] == "test_galaxy"
        return expected_products

    monkeypatch.setattr(
        "baymorph.runner.run_image_preparation",
        fake_run_image_preparation,
    )

    runner = BayMorphRunner.from_config(
        {
            "object_name": "test_galaxy",
        }
    )

    result = runner.prepare_image()

    assert result is runner
    assert runner.image_preparation_products is expected_products


def test_runner_select_psf_candidates_stores_products(
    monkeypatch,
):
    expected_products = object()

    def fake_run_psf_candidate_selection(
        config,
    ):
        assert (
            config["object_name"]
            == "test_galaxy"
        )
        return expected_products

    monkeypatch.setattr(
        "baymorph.runner.run_psf_candidate_selection",
        fake_run_psf_candidate_selection,
    )

    runner = BayMorphRunner.from_config(
        {
            "object_name": "test_galaxy",
        }
    )

    result = runner.select_psf_candidates()

    assert result is runner
    assert (
        runner.psf_candidate_products
        is expected_products
    )


def test_runner_build_psf_stores_products(monkeypatch):
    expected_products = object()

    def fake_run_psf_construction(config):
        assert config["object_name"] == "test_galaxy"
        return expected_products

    monkeypatch.setattr(
        "baymorph.runner.run_psf_construction",
        fake_run_psf_construction,
    )

    runner = BayMorphRunner.from_config(
        {
            "object_name": "test_galaxy",
        }
    )

    result = runner.build_psf()

    assert result is runner
    assert runner.psf_products is expected_products


def test_runner_fit_stores_result(monkeypatch):
    expected_result = object()

    def fake_run_real_fit(config):
        assert config["object_name"] == "test_galaxy"
        return expected_result

    monkeypatch.setattr(
        "baymorph.runner.run_real_fit",
        fake_run_real_fit,
    )

    runner = BayMorphRunner.from_config(
        {
            "object_name": "test_galaxy",
        }
    )

    result = runner.fit()

    assert result is runner
    assert runner.fit_result is expected_result

def test_runner_run_all_runs_available_steps(monkeypatch):
    calls = []

    def fake_prepare_image(self):
        calls.append("prepare_image")
        return self

    def fake_select_psf_candidates(self):
        calls.append("select_psf_candidates")
        return self

    def fake_build_psf(self):
        calls.append("build_psf")
        return self

    def fake_fit(self):
        calls.append("fit")
        return self

    monkeypatch.setattr(
        BayMorphRunner,
        "prepare_image",
        fake_prepare_image,
    )
    monkeypatch.setattr(
        BayMorphRunner,
        "select_psf_candidates",
        fake_select_psf_candidates,
    )
    monkeypatch.setattr(
        BayMorphRunner,
        "build_psf",
        fake_build_psf,
    )
    monkeypatch.setattr(
        BayMorphRunner,
        "fit",
        fake_fit,
    )

    runner = BayMorphRunner.from_config(
        {
            "object_name": "test_galaxy",
        }
    )

    result = runner.run_all()

    assert result is runner
    assert calls == [
        "prepare_image",
        "select_psf_candidates",
        "build_psf",
        "fit",
    ]