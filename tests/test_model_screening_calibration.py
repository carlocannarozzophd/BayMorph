"""Tests for model-screening probability calibration."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.analysis.model_screening_features import (
    SCREENING_FEATURE_NAMES,
)
from baymorph.validation.model_screening_calibration import (
    calibrate_probability_matrix,
    classifier_probability_matrix,
    evaluate_probability_calibration,
    fit_temperature_calibration,
)
from baymorph.validation.model_screening_classifier import (
    fit_linear_discriminant_screening_classifier,
)


MODEL_NAMES = (
    "bulge_disk",
    "double_sersic",
    "sersic_point_source",
    "single_sersic",
)


def overconfident_sample() -> tuple[np.ndarray, tuple[str, ...]]:
    """Return an intentionally overconfident labelled sample."""
    rows = []
    labels = []

    for true_index, model_name in enumerate(MODEL_NAMES):
        for case_index in range(8):
            predicted_index = (
                true_index
                if case_index < 6
                else (
                    true_index + 1
                )
                % len(MODEL_NAMES)
            )
            probability = np.full(
                len(MODEL_NAMES),
                0.01,
            )
            probability[predicted_index] = 0.97
            rows.append(probability)
            labels.append(model_name)

    return np.asarray(rows), tuple(labels)


def test_temperature_calibration_normalizes_rows() -> None:
    """Temperature scaling must preserve valid probability rows."""
    probabilities = np.asarray(
        [
            [8.0, 1.0, 1.0, 0.0],
            [1.0, 2.0, 3.0, 4.0],
        ]
    )

    calibrated = calibrate_probability_matrix(
        probabilities,
        temperature=2.0,
    )

    assert calibrated.shape == probabilities.shape
    assert np.all(calibrated >= 0.0)
    assert np.allclose(
        calibrated.sum(axis=1),
        1.0,
    )


def test_fitted_temperature_reduces_overconfidence() -> None:
    """Temperature fitting must improve NLL for overconfident scores."""
    probabilities, labels = overconfident_sample()

    calibration = fit_temperature_calibration(
        probabilities,
        labels,
        model_names=MODEL_NAMES,
    )

    assert calibration.temperature > 1.0
    assert (
        calibration.after.mean_log_loss
        < calibration.before.mean_log_loss
    )
    assert (
        calibration.after.expected_calibration_error
        < calibration.before.expected_calibration_error
    )
    assert "conditional on the adopted simulations" in (
        calibration.summary()
    )
    assert "not real-data posterior model probabilities" in (
        calibration.summary()
    )


def test_calibrated_score_mapping_preserves_model_order() -> None:
    """One score mapping must retain all calibrated model names."""
    probabilities, labels = overconfident_sample()
    calibration = fit_temperature_calibration(
        probabilities,
        labels,
        model_names=MODEL_NAMES,
    )
    scores = {
        model_name: float(probability)
        for model_name, probability in zip(
            MODEL_NAMES,
            probabilities[0],
            strict=True,
        )
    }

    calibrated = calibration.calibrate_scores(scores)

    assert tuple(calibrated) == MODEL_NAMES
    assert sum(calibrated.values()) == pytest.approx(1.0)


def test_calibration_metrics_preserve_accuracy() -> None:
    """Temperature scaling changes confidence but not class ranking."""
    probabilities, labels = overconfident_sample()
    before = evaluate_probability_calibration(
        probabilities,
        labels,
        model_names=MODEL_NAMES,
    )
    calibration = fit_temperature_calibration(
        probabilities,
        labels,
        model_names=MODEL_NAMES,
    )

    assert calibration.before == before
    assert calibration.after.accuracy == before.accuracy


def test_classifier_probability_matrix_matches_contract() -> None:
    """The BayMorph classifier must produce normalized score rows."""
    generator = np.random.default_rng(9)
    feature_blocks = []
    labels = []

    for model_index, model_name in enumerate(MODEL_NAMES):
        feature_blocks.append(
            generator.normal(
                loc=float(model_index * 4),
                scale=0.1,
                size=(
                    10,
                    len(SCREENING_FEATURE_NAMES),
                ),
            )
        )
        labels.extend([model_name] * 10)

    features = np.vstack(feature_blocks)
    classifier = fit_linear_discriminant_screening_classifier(
        features,
        labels,
        feature_names=SCREENING_FEATURE_NAMES,
    )
    probabilities = classifier_probability_matrix(
        classifier,
        features,
    )

    assert probabilities.shape == (
        len(labels),
        len(MODEL_NAMES),
    )
    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
    )


def test_calibration_rejects_missing_model_class() -> None:
    """Calibration data must represent every model class."""
    probabilities, labels = overconfident_sample()
    selected = np.asarray(labels) != "single_sersic"

    with pytest.raises(
        ValueError,
        match="include every model",
    ):
        fit_temperature_calibration(
            probabilities[selected],
            np.asarray(labels)[selected],
            model_names=MODEL_NAMES,
        )
