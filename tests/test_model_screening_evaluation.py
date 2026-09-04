"""Tests for model-screening holdout evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.analysis.model_screening_features import (
    SCREENING_FEATURE_NAMES,
)
from baymorph.validation.model_screening_classifier import (
    fit_linear_discriminant_screening_classifier,
)
from baymorph.validation.model_screening_evaluation import (
    evaluate_model_screening_classifier,
    fit_and_evaluate_screening_classifier,
    stratified_screening_holdout,
)


MODEL_NAMES = (
    "bulge_disk",
    "double_sersic",
    "sersic_point_source",
    "single_sersic",
)


def separated_dataset(
    *,
    n_per_model: int = 20,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return an easily separable four-class feature dataset."""
    generator = np.random.default_rng(17)
    feature_blocks = []
    labels = []

    for model_index, model_name in enumerate(MODEL_NAMES):
        center = float(model_index * 6)
        feature_blocks.append(
            generator.normal(
                loc=center,
                scale=0.1,
                size=(
                    n_per_model,
                    len(SCREENING_FEATURE_NAMES),
                ),
            )
        )
        labels.extend([model_name] * n_per_model)

    return np.vstack(feature_blocks), tuple(labels)


def test_stratified_holdout_is_deterministic_and_disjoint() -> None:
    """The split must preserve classes without leaking cases."""
    _, labels = separated_dataset(n_per_model=20)

    first_train, first_validation = stratified_screening_holdout(
        labels,
        validation_fraction=0.25,
        random_state=8,
    )
    second_train, second_validation = stratified_screening_holdout(
        labels,
        validation_fraction=0.25,
        random_state=8,
    )

    assert np.array_equal(first_train, second_train)
    assert np.array_equal(first_validation, second_validation)
    assert np.intersect1d(first_train, first_validation).size == 0
    assert first_train.size == 60
    assert first_validation.size == 20

    validation_labels = np.asarray(labels)[first_validation]
    for model_name in MODEL_NAMES:
        assert np.count_nonzero(validation_labels == model_name) == 5


def test_fit_and_evaluate_recovers_separated_classes() -> None:
    """The baseline must recover a deliberately separated dataset."""
    features, labels = separated_dataset()

    result = fit_and_evaluate_screening_classifier(
        features,
        labels,
        feature_names=SCREENING_FEATURE_NAMES,
        validation_fraction=0.25,
        random_state=4,
    )

    evaluation = result.evaluation

    assert evaluation.n_samples == 20
    assert evaluation.accuracy == pytest.approx(1.0)
    assert evaluation.top_two_accuracy == pytest.approx(1.0)
    assert evaluation.mean_log_loss >= 0.0
    assert 0.0 <= evaluation.multiclass_brier_score <= 2.0
    assert np.array_equal(
        evaluation.confusion_matrix,
        np.diag([5, 5, 5, 5]),
    )
    assert set(evaluation.per_class_recall) == set(MODEL_NAMES)
    assert all(
        recall == pytest.approx(1.0)
        for recall in evaluation.per_class_recall.values()
    )


def test_evaluation_summary_preserves_scientific_scope() -> None:
    """The summary must state the simulation-conditioned limitation."""
    features, labels = separated_dataset()
    result = fit_and_evaluate_screening_classifier(
        features,
        labels,
        feature_names=SCREENING_FEATURE_NAMES,
        random_state=3,
    )

    summary = result.evaluation.summary()

    assert "Top-two accuracy" in summary
    assert "Multiclass Brier score" in summary
    assert "conditional on the adopted simulations" in summary
    assert "do not establish real-data model probabilities" in summary


def test_evaluation_rejects_unknown_labels() -> None:
    """Evaluation labels must belong to the fitted model registry."""
    features, labels = separated_dataset()
    classifier = fit_linear_discriminant_screening_classifier(
        features,
        labels,
        feature_names=SCREENING_FEATURE_NAMES,
    )
    invalid_labels = list(labels)
    invalid_labels[0] = "unknown_model"

    with pytest.raises(
        ValueError,
        match="unknown models",
    ):
        evaluate_model_screening_classifier(
            classifier,
            features,
            invalid_labels,
        )


@pytest.mark.parametrize(
    "validation_fraction",
    (
        0.0,
        1.0,
        -0.1,
        1.1,
    ),
)
def test_holdout_rejects_invalid_validation_fraction(
    validation_fraction: float,
) -> None:
    """The validation fraction must define two non-empty partitions."""
    _, labels = separated_dataset()

    with pytest.raises(
        ValueError,
        match="strictly in",
    ):
        stratified_screening_holdout(
            labels,
            validation_fraction=validation_fraction,
        )


def test_holdout_requires_two_cases_per_model() -> None:
    """Every class must contribute to training and validation."""
    labels = (
        "bulge_disk",
        "double_sersic",
        "double_sersic",
    )

    with pytest.raises(
        ValueError,
        match="at least two cases",
    ):
        stratified_screening_holdout(labels)
