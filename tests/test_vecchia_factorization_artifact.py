"""Production-critical tests for immutable Vecchia artifacts."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import baymorph.pipeline.vecchia_factorization as artifacts


def _kernel() -> np.ndarray:
    return np.array(
        [
            [0.02, 0.06, 0.02],
            [0.08, 1.00, 0.08],
            [0.02, 0.06, 0.02],
        ],
        dtype=float,
    )


@pytest.fixture
def artifact_environment(tmp_path, monkeypatch):
    """Install deterministic prepared-image and calibration test doubles."""
    contexts = {}
    calibrations = {}

    def register(name):
        root = tmp_path / name / "image_preparation"
        root.mkdir(parents=True)
        calibration_path = root / "correlated_gaussian_noise.npz"
        calibration_path.write_bytes(b"fixed calibration bytes\n")

        shape = (9, 9)
        prepared = (
            np.arange(
                np.prod(shape),
                dtype=float,
            ).reshape(shape)
            / 9.0
        )
        fitting_mask = np.zeros(shape, dtype=bool)
        fitting_mask[1, 2] = True
        fitting_mask[5, 6] = True
        sky_mask = np.ones(shape, dtype=bool)
        sky_mask[3:5, 3:5] = False

        products = SimpleNamespace(
            cutout=prepared + 20.0,
            prepared_data=prepared,
            fitting_mask=fitting_mask,
            sky_region=SimpleNamespace(
                mask=sky_mask
            ),
        )
        calibration = SimpleNamespace(
            correlation_kernel=_kernel(),
            metadata={
                "estimator": artifacts.CORRELATION_ESTIMATOR,
                "max_lag": 1,
                "generator_identity": {
                    "name": "test-calibration",
                    "version": 1,
                },
            },
        )
        config = {
            "test_context": name
        }
        contexts[name] = {
            "path": calibration_path,
            "products": products,
        }
        calibrations[calibration_path] = calibration
        return config, products

    def context(config):
        return contexts[
            config["test_context"]
        ]

    monkeypatch.setattr(
        artifacts,
        "correlated_noise_calibration_path",
        lambda config: context(config)["path"],
    )
    monkeypatch.setattr(
        artifacts,
        "load_image_preparation_products",
        lambda config: context(config)["products"],
    )
    monkeypatch.setattr(
        artifacts,
        "load_correlated_noise_calibration",
        lambda path: calibrations[Path(path)],
    )
    # A kernel to lag 1 keeps the artifact tests small; production kernels
    # reach the lag of AD-047.
    monkeypatch.setattr(
        artifacts,
        "CORRELATION_MAX_LAG",
        1,
    )
    monkeypatch.setattr(
        artifacts,
        "validate_correlated_noise_calibration_sources",
        lambda *args, **kwargs: None,
    )

    return register


def _bundle_bytes(path):
    return {
        child.name: child.read_bytes()
        for child in path.iterdir()
        if child.is_file()
    }


def _is_mmap_backed(array):
    current = array

    while current is not None:
        if isinstance(current, np.memmap):
            return True

        current = getattr(
            current,
            "base",
            None,
        )

    return False


def test_artifact_is_deterministic_immutable_and_mapped(
    artifact_environment,
):
    """Equal scientific inputs must persist as equal authenticated bytes."""
    first_config, _ = artifact_environment(
        "first"
    )
    second_config, _ = artifact_environment(
        "second"
    )

    first = (
        artifacts.build_vecchia_factorization_artifact(
            first_config
        )
    )
    second = (
        artifacts.build_vecchia_factorization_artifact(
            second_config
        )
    )

    first_path = (
        artifacts.vecchia_factorization_path(
            first_config
        )
    )
    second_path = (
        artifacts.vecchia_factorization_path(
            second_config
        )
    )

    assert (
        _bundle_bytes(first_path)
        == _bundle_bytes(second_path)
    )

    loaded = (
        artifacts.load_vecchia_factorization_artifact(
            first_config
        )
    )

    for name in artifacts._ARRAY_FILES:
        np.testing.assert_array_equal(
            getattr(loaded, name),
            getattr(first, name),
        )
        np.testing.assert_array_equal(
            getattr(loaded, name),
            getattr(second, name),
        )

        array = getattr(
            loaded,
            name,
        )

        assert _is_mmap_backed(array)
        assert not array.flags.writeable

    for child in first_path.iterdir():
        assert (
            child.stat().st_mode
            & 0o222
            == 0
        )


def test_frozen_orders_use_separate_namespaces(
    artifact_environment,
):
    """Production m64 and validation m96 artifacts may coexist."""
    config, _ = artifact_environment(
        "orders"
    )

    m64 = (
        artifacts.build_vecchia_factorization_artifact(
            config,
            max_neighbors=64,
        )
    )
    m96 = (
        artifacts.build_vecchia_factorization_artifact(
            config,
            max_neighbors=96,
        )
    )

    path64 = artifacts.vecchia_factorization_path(
        config,
        max_neighbors=64,
    )
    path96 = artifacts.vecchia_factorization_path(
        config,
        max_neighbors=96,
    )

    assert path64.name == "m64"
    assert path96.name == "m96"
    assert path64 != path96
    assert m64.max_neighbors == 64
    assert m96.max_neighbors == 96

    with pytest.raises(
        ValueError,
        match="frozen orders",
    ):
        artifacts.vecchia_factorization_path(
            config,
            max_neighbors=128,
        )


def test_completed_artifact_is_never_overwritten(
    artifact_environment,
):
    """Build must reject replacement while ensure only validates."""
    config, _ = artifact_environment(
        "immutable"
    )

    expected = (
        artifacts.build_vecchia_factorization_artifact(
            config
        )
    )

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        artifacts.build_vecchia_factorization_artifact(
            config
        )

    loaded = (
        artifacts.ensure_vecchia_factorization_artifact(
            config
        )
    )

    np.testing.assert_array_equal(
        loaded.weights,
        expected.weights,
    )


def test_partial_artifact_is_never_replaced(
    artifact_environment,
    monkeypatch,
):
    """A crashed publication must remain visible and fail closed."""
    config, _ = artifact_environment(
        "partial"
    )
    path = artifacts.vecchia_factorization_path(
        config
    )
    path.mkdir(parents=True)

    partial = path / "manifest.json"
    partial.write_text(
        "partial\n"
    )

    monkeypatch.setattr(
        artifacts,
        "VECCHIA_PUBLICATION_WAIT_SECONDS",
        0.0,
    )

    with pytest.raises(
        RuntimeError,
        match="exists but is incomplete",
    ):
        artifacts.ensure_vecchia_factorization_artifact(
            config
        )

    assert partial.read_text() == "partial\n"


def test_simultaneous_publishers_cannot_replace_each_other(
    artifact_environment,
    monkeypatch,
):
    """Exactly one competing publisher may claim an artifact namespace."""
    config, _ = artifact_environment(
        "race"
    )

    snapshot = artifacts._capture_source_snapshot(
        config
    )
    factorization = (
        artifacts.build_vecchia_factorization(
            snapshot.fitting_mask,
            snapshot.correlation_kernel,
            max_neighbors=64,
        )
    )
    path = artifacts.vecchia_factorization_path(
        config
    )

    barrier = threading.Barrier(2)
    original_write_json = artifacts._write_json

    def synchronized_write(output, payload):
        original_write_json(
            output,
            payload,
        )

        if (
            output.name
            == artifacts.VECCHIA_COMPLETE_FILENAME
        ):
            barrier.wait(
                timeout=5.0
            )

    monkeypatch.setattr(
        artifacts,
        "_write_json",
        synchronized_write,
    )

    def publish():
        try:
            artifacts._save_vecchia_bundle(
                path,
                factorization,
                binding=snapshot.binding,
                min_conditional_variance=(
                    artifacts
                    .DEFAULT_MIN_CONDITIONAL_VARIANCE
                ),
            )
        except FileExistsError:
            return "lost"

        return "won"

    with ThreadPoolExecutor(
        max_workers=2
    ) as executor:
        outcomes = list(
            executor.map(
                lambda _: publish(),
                range(2),
            )
        )

    assert sorted(outcomes) == [
        "lost",
        "won",
    ]


def test_array_tamper_is_rejected(
    artifact_environment,
):
    """An altered factor array must fail before numerical use."""
    config, _ = artifact_environment(
        "tamper"
    )

    artifacts.build_vecchia_factorization_artifact(
        config
    )
    path = artifacts.vecchia_factorization_path(
        config
    )
    weights = path / "weights.npy"
    weights.chmod(0o644)

    with weights.open("ab") as stream:
        stream.write(
            b"tamper"
        )

    with pytest.raises(
        ValueError,
        match="checksum.*weights",
    ):
        artifacts.load_vecchia_factorization_artifact(
            config
        )


@pytest.mark.parametrize(
    "change",
    [
        "mask",
        "calibration",
    ],
)
def test_changed_scientific_sources_are_stale(
    artifact_environment,
    change,
):
    """Mask or calibration changes must invalidate an artifact."""
    config, products = artifact_environment(
        f"stale-{change}"
    )

    artifacts.build_vecchia_factorization_artifact(
        config
    )

    if change == "mask":
        products.fitting_mask[
            1,
            2,
        ] = False
        products.fitting_mask[
            1,
            3,
        ] = True
    else:
        path = (
            artifacts
            .correlated_noise_calibration_path(
                config
            )
        )
        path.write_bytes(
            b"different calibration bytes\n"
        )

    with pytest.raises(
        ValueError,
        match="does not match the current",
    ):
        artifacts.load_vecchia_factorization_artifact(
            config
        )


def test_settings_and_path_bypasses_are_rejected(
    artifact_environment,
):
    """Inference must retain settings and source revalidation."""
    config, _ = artifact_environment(
        "bypass"
    )

    artifacts.build_vecchia_factorization_artifact(
        config
    )
    path = artifacts.vecchia_factorization_path(
        config
    )

    with pytest.raises(
        ValueError,
        match="min_conditional_variance",
    ):
        artifacts.load_vecchia_factorization_artifact(
            config,
            min_conditional_variance=2.0e-12,
        )

    with pytest.raises(
        ValueError,
        match="by artifact path is disabled",
    ):
        artifacts.load_vecchia_factorization_artifact(
            path
        )

    manifest = (
        artifacts.load_vecchia_factorization_manifest(
            path
        )
    )

    assert manifest["max_neighbors"] == 64


def test_manifest_binds_the_sky_correlation_of_ad047(
    artifact_environment,
):
    """The artifact records its schema and the estimator of its kernel."""
    config, _ = artifact_environment(
        "schema"
    )

    artifacts.build_vecchia_factorization_artifact(
        config
    )
    manifest = artifacts.load_vecchia_factorization_manifest(
        config
    )

    assert manifest["schema_version"] == 2
    assert (
        manifest["source_binding"]["correlation_estimator"]
        == artifacts.CORRELATION_ESTIMATOR
    )
    assert "marginal_sigma_sha256" not in manifest["source_binding"]


def test_calibration_of_another_estimator_is_rejected(
    artifact_environment,
):
    """A kernel not estimated as AD-047 prescribes cannot be factorized."""
    config, _ = artifact_environment(
        "estimator"
    )
    calibration = artifacts.load_correlated_noise_calibration(
        artifacts.correlated_noise_calibration_path(
            config
        )
    )
    calibration.metadata["estimator"] = "tapered_tiles"

    with pytest.raises(
        ValueError,
        match="AD-047",
    ):
        artifacts.build_vecchia_factorization_artifact(
            config
        )
