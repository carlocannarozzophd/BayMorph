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

    runner.prepare_image()

    assert runner.image is expected_products


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

    runner.select_psf_candidates()

    assert runner.psf_candidates is expected_products


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

    runner.build_psf()

    assert runner.psf is expected_products


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

    runner.fit()

    assert runner.results is expected_result


def test_runner_fit_stores_posterior_result(monkeypatch):
    expected_result = object()

    def fake_run_real_fit(config):
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

    runner.fit()

    assert runner.results is expected_result