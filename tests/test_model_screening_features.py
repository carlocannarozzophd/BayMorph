"""Tests for interpretable model-screening image features."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.analysis.model_screening_features import (
    SCREENING_FEATURE_NAMES,
    extract_model_screening_features,
)


def make_test_image() -> np.ndarray:
    """Return one smooth elliptical target with constant background."""
    y, x = np.indices(
        (
            81,
            81,
        ),
        dtype=float,
    )

    source = (
        100.0
        * np.exp(
            -0.5
            * (
                (
                    (
                        x
                        - 40.0
                    )
                    / 6.0
                )
                ** 2
                + (
                    (
                        y
                        - 40.0
                    )
                    / 4.0
                )
                ** 2
            )
        )
    )

    return (
        10.0
        + source
    )


def test_feature_vector_has_stable_named_contract() -> None:
    """The extractor must return finite interpretable features."""
    features = extract_model_screening_features(
        make_test_image(),
        target_x=40.0,
        target_y=40.0,
        psf_fwhm=2.0,
        background=10.0,
        noise_sigma=1.0,
    )

    assert features.names == SCREENING_FEATURE_NAMES
    assert features.values.shape == (
        len(
            SCREENING_FEATURE_NAMES
        ),
    )

    assert np.all(
        np.isfinite(
            features.values
        )
    )

    values = features.as_mapping()

    assert (
        0.0
        < values[
            "r20_pixels"
        ]
        < values[
            "r50_pixels"
        ]
        < values[
            "r80_pixels"
        ]
    )

    assert (
        0.0
        < values[
            "axis_ratio_moment"
        ]
        <= 1.0
    )

    assert (
        0.0
        < values[
            "central_fraction_one_fwhm"
        ]
        < 1.0
    )

    assert (
        0.0
        <= values[
            "outer_fraction_beyond_two_r50"
        ]
        < 1.0
    )


def test_feature_extraction_is_deterministic() -> None:
    """Identical inputs must produce identical features."""
    options = {
        "target_x": 40.0,
        "target_y": 40.0,
        "psf_fwhm": 2.0,
        "background": 10.0,
        "noise_sigma": 1.0,
    }

    first = extract_model_screening_features(
        make_test_image(),
        **options,
    )
    second = extract_model_screening_features(
        make_test_image(),
        **options,
    )

    assert np.array_equal(
        first.values,
        second.values,
    )


def test_supplied_target_mask_is_not_modified() -> None:
    """Screening must not mutate a caller-owned target mask."""
    image = make_test_image()

    y, x = np.indices(
        image.shape
    )

    target_mask = (
        (
            x
            - 40
        )
        ** 2
        + (
            y
            - 40
        )
        ** 2
        <= 15**2
    )
    original_mask = target_mask.copy()

    features = extract_model_screening_features(
        image,
        target_x=40.0,
        target_y=40.0,
        psf_fwhm=2.0,
        background=10.0,
        noise_sigma=1.0,
        target_mask=target_mask,
    )

    assert np.array_equal(
        target_mask,
        original_mask,
    )
    assert np.all(
        np.isfinite(
            features.values
        )
    )


def test_feature_extraction_rejects_wrong_mask_shape() -> None:
    """A provisional target mask must match the image footprint."""
    with pytest.raises(
        ValueError,
        match="must match the image shape",
    ):
        extract_model_screening_features(
            make_test_image(),
            target_x=40.0,
            target_y=40.0,
            psf_fwhm=2.0,
            background=10.0,
            noise_sigma=1.0,
            target_mask=np.ones(
                (
                    80,
                    81,
                ),
                dtype=bool,
            ),
        )


def test_feature_extraction_rejects_target_outside_image() -> None:
    """The target seed must identify a location inside the image."""
    with pytest.raises(
        ValueError,
        match="must lie inside the image",
    ):
        extract_model_screening_features(
            make_test_image(),
            target_x=100.0,
            target_y=40.0,
            psf_fwhm=2.0,
            background=10.0,
            noise_sigma=1.0,
        )


def test_feature_extraction_rejects_missing_target() -> None:
    """The extractor must abstain when no source is detected."""
    with pytest.raises(
        ValueError,
        match="No provisional source segment",
    ):
        extract_model_screening_features(
            np.ones(
                (
                    32,
                    32,
                )
            ),
            target_x=15.5,
            target_y=15.5,
            psf_fwhm=2.0,
            background=1.0,
            noise_sigma=1.0,
        )
