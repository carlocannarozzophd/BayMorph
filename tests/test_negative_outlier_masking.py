"""Exclusion of impossibly negative pixels from the likelihood support."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from baymorph.core import negative_outlier_mask
from baymorph.pipeline.config import resolve_config, validate_config
from baymorph.pipeline.image_preparation import (
    load_image_preparation_products,
    prepare_real_image_from_config,
)
from baymorph.pipeline.outputs import pipeline_paths


# ---------------------------------------------------------------------------
# Criterion
# ---------------------------------------------------------------------------


def test_negative_outlier_mask_flags_only_deep_negative_pixels() -> None:
    """Pixels below minus the threshold are excluded; others are kept."""
    data = np.zeros((4, 4), dtype=float)
    data[1, 1] = -30.0
    data[2, 2] = -4.0
    data[3, 3] = +30.0

    mask = negative_outlier_mask(
        data,
        2.0,
        threshold_sigma=5.0,
    )

    assert mask[1, 1]
    assert not mask[2, 2]
    assert not mask[3, 3]
    assert mask.sum() == 1
    assert mask.dtype == bool


def test_negative_outlier_mask_is_not_symmetric() -> None:
    """A bright pixel may be a source; a deeply negative one cannot be."""
    data = np.zeros((3, 3), dtype=float)
    data[0, 0] = -1000.0
    data[2, 2] = +1000.0

    mask = negative_outlier_mask(
        data,
        1.0,
        threshold_sigma=5.0,
    )

    assert mask[0, 0]
    assert not mask[2, 2]


def test_negative_outlier_mask_accepts_a_noise_map() -> None:
    """A per-pixel noise map scales the threshold pixel by pixel."""
    data = np.full((2, 2), -10.0)
    noise = np.array(
        [
            [1.0, 4.0],
            [1.0, 4.0],
        ]
    )

    mask = negative_outlier_mask(
        data,
        noise,
        threshold_sigma=5.0,
    )

    assert mask[0, 0]
    assert mask[1, 0]
    assert not mask[0, 1]
    assert not mask[1, 1]


@pytest.mark.parametrize(
    "threshold",
    [0.0, -1.0, float("nan"), float("inf")],
)
def test_negative_outlier_mask_rejects_invalid_thresholds(
    threshold: float,
) -> None:
    """The threshold must be finite and positive."""
    with pytest.raises(ValueError, match="threshold_sigma"):
        negative_outlier_mask(
            np.zeros((2, 2)),
            1.0,
            threshold_sigma=threshold,
        )


def test_negative_outlier_mask_rejects_non_positive_noise() -> None:
    """A noise scale of zero has no meaning for this criterion."""
    with pytest.raises(ValueError, match="strictly positive"):
        negative_outlier_mask(
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            threshold_sigma=5.0,
        )


def test_negative_outlier_mask_rejects_shape_mismatch() -> None:
    """A noise map must match the image it describes."""
    with pytest.raises(ValueError, match="same shape"):
        negative_outlier_mask(
            np.zeros((2, 2)),
            np.ones((3, 3)),
            threshold_sigma=5.0,
        )


def test_negative_outlier_mask_rejects_non_finite_data() -> None:
    """Non-finite pixels are a preparation failure, not an exclusion."""
    data = np.zeros((2, 2))
    data[0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        negative_outlier_mask(
            data,
            1.0,
            threshold_sigma=5.0,
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _minimal_config(tmp_path: Path) -> dict:
    """Return one resolvable single-observation configuration."""
    return {
        "object": {"name": "example"},
        "data": {
            "image_path": str(tmp_path / "image.fits"),
            "target_position": {"x": 5, "y": 5},
        },
        "outputs": {"root": str(tmp_path / "runs")},
    }


def test_default_configuration_declares_the_threshold(
    tmp_path: Path,
) -> None:
    """The criterion is active by default at five noise scales."""
    resolved = resolve_config(_minimal_config(tmp_path))

    assert (
        resolved["preparation"]["masking"]["negative_threshold_sigma"] == 5.0
    )


def test_configuration_rejects_unknown_masking_entries(
    tmp_path: Path,
) -> None:
    """Only the declared masking control is accepted."""
    config = _minimal_config(tmp_path)
    config["preparation"] = {"masking": {"clip_positive": True}}

    with pytest.raises(ValueError, match="preparation.masking"):
        validate_config(resolve_config(config))


@pytest.mark.parametrize(
    "threshold",
    [0.0, -1.0, "five", True],
)
def test_configuration_rejects_invalid_thresholds(
    tmp_path: Path,
    threshold: object,
) -> None:
    """The declared threshold must be a positive number, or null."""
    config = _minimal_config(tmp_path)
    config["preparation"] = {
        "masking": {"negative_threshold_sigma": threshold}
    }

    with pytest.raises(ValueError, match="negative_threshold_sigma"):
        validate_config(resolve_config(config))


def test_configuration_accepts_null_to_disable(
    tmp_path: Path,
) -> None:
    """Declaring null disables the criterion without error."""
    config = _minimal_config(tmp_path)
    config["preparation"] = {"masking": {"negative_threshold_sigma": None}}

    resolved = resolve_config(config)
    validate_config(resolved)

    assert (
        resolved["preparation"]["masking"]["negative_threshold_sigma"] is None
    )


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


ARTIFACT_PIXELS = ((60, 30), (61, 30), (62, 30))


def _image_with_negative_artifact(path: Path) -> None:
    """Write one detectable source plus a deep negative detector artifact."""
    rng = np.random.default_rng(20260920)
    size = 120
    yy, xx = np.mgrid[0:size, 0:size]

    image = 100.0 + rng.normal(0.0, 2.0, size=(size, size))
    image += 400.0 * np.exp(
        -0.5 * ((yy - 60.0) ** 2 + (xx - 60.0) ** 2) / 6.0**2
    )

    for y, x in ARTIFACT_PIXELS:
        image[y, x] = -900.0

    fits.writeto(path, image, overwrite=True)


def _preparation_config(
    tmp_path: Path,
    *,
    threshold: float | None | str = "default",
) -> dict:
    """Return one fixed-cutout preparation configuration."""
    config = {
        "object": {"name": "negative_artifact"},
        "data": {
            "image_path": str(tmp_path / "image.fits"),
            "target_position": {"x": 60, "y": 60},
            "cutout": {"mode": "fixed", "size": 100},
        },
        "outputs": {
            "root": str(tmp_path / "runs"),
            "overwrite": True,
        },
    }

    if threshold != "default":
        config["preparation"] = {
            "masking": {"negative_threshold_sigma": threshold}
        }

    return resolve_config(config)


def _artifact_positions_in_cutout(products) -> list[tuple[int, int]]:
    """Map the injected artifact pixels into cutout coordinates."""
    bounds = products.cutout_bounds

    return [
        (y - bounds["ymin"], x - bounds["xmin"])
        for y, x in ARTIFACT_PIXELS
    ]


def test_preparation_excludes_negative_artifacts(
    tmp_path: Path,
) -> None:
    """Deeply negative detector pixels never reach the likelihood."""
    _image_with_negative_artifact(tmp_path / "image.fits")

    products = prepare_real_image_from_config(
        _preparation_config(tmp_path),
    )

    for y, x in _artifact_positions_in_cutout(products):
        assert products.prepared_data[y, x] < -5.0 * products.noise[y, x]
        assert products.fitting_mask[y, x]


def test_preparation_keeps_negative_artifacts_when_disabled(
    tmp_path: Path,
) -> None:
    """Declaring null leaves the criterion out of the likelihood support."""
    _image_with_negative_artifact(tmp_path / "image.fits")

    products = prepare_real_image_from_config(
        _preparation_config(tmp_path, threshold=None),
    )

    positions = _artifact_positions_in_cutout(products)

    assert not any(
        products.fitting_mask[y, x] for y, x in positions
    )


def test_preparation_records_the_exclusion_in_metadata(
    tmp_path: Path,
) -> None:
    """The declared threshold and the excluded count are provenance."""
    import json

    _image_with_negative_artifact(tmp_path / "image.fits")
    config = _preparation_config(tmp_path)

    prepare_real_image_from_config(config)

    paths = pipeline_paths(config)["image_preparation"]

    with open(paths["dir"] / "image_preparation_metadata.json") as stream:
        metadata = json.load(stream)

    assert metadata["negative_outliers"]["threshold_sigma"] == 5.0
    assert metadata["negative_outliers"]["excluded_pixels"] >= len(
        ARTIFACT_PIXELS
    )


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_persisted_products_reload_with_the_criterion_applied(
    tmp_path: Path,
) -> None:
    """A mask built with the criterion still matches its persisted inputs."""
    _image_with_negative_artifact(tmp_path / "image.fits")
    config = _preparation_config(tmp_path)

    built = prepare_real_image_from_config(config)

    config["outputs"]["overwrite"] = False
    reloaded = load_image_preparation_products(config)

    assert np.array_equal(
        reloaded.fitting_mask,
        built.fitting_mask,
    )


def test_reload_rejects_a_mask_that_readmits_a_negative_artifact(
    tmp_path: Path,
) -> None:
    """The likelihood support cannot be widened behind the criterion."""
    _image_with_negative_artifact(tmp_path / "image.fits")
    config = _preparation_config(tmp_path)

    built = prepare_real_image_from_config(config)
    paths = pipeline_paths(config)["image_preparation"]

    tampered = np.asarray(built.fitting_mask, dtype=bool).copy()

    for y, x in _artifact_positions_in_cutout(built):
        tampered[y, x] = False

    np.save(paths["dir"] / "fitting_mask.npy", tampered)
    fits.writeto(
        paths["dir"] / "fitting_mask.fits",
        tampered.astype(np.uint8),
        overwrite=True,
    )

    config["outputs"]["overwrite"] = False

    with pytest.raises(RuntimeError):
        load_image_preparation_products(config)
