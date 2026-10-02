"""Tests for the model-screening validation classifier."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.analysis.model_screening import SCREENING_MODELS
from baymorph.analysis.model_screening_features import (
    SCREENING_FEATURE_NAMES,
    ModelScreeningFeatureVector,
)
from baymorph.validation.model_screening_classifier import (
    fit_linear_discriminant_screening_classifier,
    screening_feature_dataset,
)
from baymorph.validation.model_screening_simulation import (
    simulate_model_screening_case,
)


def make_separated_training_data() -> tuple[
    np.ndarray,
    tuple[str, ...],
]:
    """Return reproducible separated feature clusters."""
    rng = np.random.default_rng(
        31
    )
    feature_rows = []
    labels = []

    for model_index, model_name in enumerate(
        SCREENING_MODELS
    ):
        for _ in range(
            20
        ):
            feature_rows.append(
                rng.normal(
                    loc=3.0
                    * model_index,
                    scale=0.2,
                    size=len(
                        SCREENING_FEATURE_NAMES
                    ),
                )
            )
            labels.append(
                model_name
            )

    return (
        np.asarray(
            feature_rows
        ),
        tuple(
            labels
        ),
    )


def test_classifier_returns_normalized_model_scores() -> None:
    """The classifier must score every screening model."""
    features, labels = make_separated_training_data()

    classifier = (
        fit_linear_discriminant_screening_classifier(
            features,
            labels,
        )
    )

    target_model = SCREENING_MODELS[
        1
    ]

    candidate = ModelScreeningFeatureVector(
        values=np.full(
            len(
                SCREENING_FEATURE_NAMES
            ),
            3.0,
        )
    )

    scores = classifier.predict_scores(
        candidate
    )

    assert set(
        scores
    ) == set(
        SCREENING_MODELS
    )

    assert sum(
        scores.values()
    ) == pytest.approx(
        1.0
    )

    assert max(
        scores,
        key=scores.get,
    ) == target_model


def test_uncalibrated_classifier_forces_ambiguous_status() -> None:
    """An uncalibrated validation model must never recommend a model."""
    features, labels = make_separated_training_data()

    classifier = (
        fit_linear_discriminant_screening_classifier(
            features,
            labels,
        )
    )

    candidate = ModelScreeningFeatureVector(
        values=np.zeros(
            len(
                SCREENING_FEATURE_NAMES
            )
        )
    )

    report = classifier.screen(
        candidate
    )

    assert report.status == "ambiguous"
    assert report.user_decision_required

    assert (
        "not probability-calibrated"
        in report.warnings[
            0
        ]
    )


def test_classifier_requires_every_model_in_training() -> None:
    """Training cannot silently omit a registered candidate model."""
    features, labels = make_separated_training_data()

    excluded_model = SCREENING_MODELS[
        -1
    ]

    keep = np.asarray(
        [
            label
            != excluded_model
            for label in labels
        ]
    )

    with pytest.raises(
        ValueError,
        match="must contain every screening model",
    ):
        fit_linear_discriminant_screening_classifier(
            features[
                keep
            ],
            tuple(
                label
                for label in labels
                if label
                != excluded_model
            ),
        )


def test_classifier_rejects_wrong_feature_contract() -> None:
    """Prediction features must match the training feature names."""
    features, labels = make_separated_training_data()

    classifier = (
        fit_linear_discriminant_screening_classifier(
            features,
            labels,
        )
    )

    wrong_names = (
        "wrong_feature",
        *SCREENING_FEATURE_NAMES[
            1:
        ],
    )

    candidate = ModelScreeningFeatureVector(
        values=np.zeros(
            len(
                wrong_names
            )
        ),
        names=wrong_names,
    )

    with pytest.raises(
        ValueError,
        match="do not match the classifier",
    ):
        classifier.predict_scores(
            candidate
        )


def test_simulation_connects_to_feature_dataset() -> None:
    """A forward simulation must produce one ML feature row."""
    case = simulate_model_screening_case(
        model_name="single_sersic",
        parameters={
            "flux": 1000.0,
            "re": 4.0,
            "n": 2.0,
            "q": 0.8,
            "theta": 0.2,
            "x0": 31.5,
            "y0": 31.5,
            "background_level": 10.0,
        },
        image_shape=(
            64,
            64,
        ),
        psf_fwhm=2.0,
        noise_sigma=1.0,
        random_seed=13,
    )

    features, labels = screening_feature_dataset(
        [
            case,
        ]
    )

    assert features.shape == (
        1,
        len(
            SCREENING_FEATURE_NAMES
        ),
    )

    assert labels == (
        "single_sersic",
    )

    assert np.all(
        np.isfinite(
            features
        )
    )
