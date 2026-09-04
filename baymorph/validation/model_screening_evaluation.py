"""Validation metrics for simulation-trained model screening."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from baymorph.analysis.model_screening_features import (
    ModelScreeningFeatureVector,
)
from baymorph.validation.model_screening_classifier import (
    LinearDiscriminantScreeningClassifier,
    fit_linear_discriminant_screening_classifier,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ModelScreeningEvaluation:
    """Holdout metrics for a model-screening classifier.

    The confusion matrix uses true classes along rows and predicted classes
    along columns. These metrics describe performance under the adopted
    simulation distribution; they do not validate real-data probabilities.
    """

    model_names: tuple[str, ...]
    n_samples: int
    confusion_matrix: IntArray
    accuracy: float
    top_two_accuracy: float
    mean_log_loss: float
    multiclass_brier_score: float
    per_class_recall: Mapping[str, float]

    def __post_init__(self) -> None:
        """Validate and freeze the evaluation payload."""
        if not self.model_names:
            raise ValueError("model_names must not be empty.")

        if len(set(self.model_names)) != len(self.model_names):
            raise ValueError("model_names must be unique.")

        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        confusion = np.asarray(
            self.confusion_matrix,
            dtype=np.int64,
        )

        expected_shape = (
            len(self.model_names),
            len(self.model_names),
        )
        if confusion.shape != expected_shape:
            raise ValueError(
                "confusion_matrix must have shape "
                f"{expected_shape}, got {confusion.shape}."
            )

        if np.any(confusion < 0):
            raise ValueError("confusion_matrix must be non-negative.")

        if int(np.sum(confusion)) != self.n_samples:
            raise ValueError(
                "confusion_matrix counts must sum to n_samples."
            )

        for name, value in (
            ("accuracy", self.accuracy),
            ("top_two_accuracy", self.top_two_accuracy),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")

        if (
            not np.isfinite(self.mean_log_loss)
            or self.mean_log_loss < 0.0
        ):
            raise ValueError(
                "mean_log_loss must be finite and non-negative."
            )

        if (
            not np.isfinite(self.multiclass_brier_score)
            or not 0.0 <= self.multiclass_brier_score <= 2.0
        ):
            raise ValueError(
                "multiclass_brier_score must lie in [0, 2]."
            )

        recall = {
            str(name): float(value)
            for name, value in self.per_class_recall.items()
        }
        if set(recall) != set(self.model_names):
            raise ValueError(
                "per_class_recall must contain every model exactly once."
            )

        for value in recall.values():
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    "Every per-class recall must lie in [0, 1]."
                )

        frozen_confusion = np.array(
            confusion,
            dtype=np.int64,
            copy=True,
        )
        frozen_confusion.setflags(write=False)

        object.__setattr__(
            self,
            "confusion_matrix",
            frozen_confusion,
        )
        object.__setattr__(
            self,
            "per_class_recall",
            MappingProxyType(recall),
        )

    def summary(self) -> str:
        """Return a concise human-readable validation summary."""
        lines = [
            "BayMorph Model Screening Validation",
            "",
            f"Holdout samples: {self.n_samples}",
            f"Accuracy: {self.accuracy:.6f}",
            f"Top-two accuracy: {self.top_two_accuracy:.6f}",
            f"Mean log loss: {self.mean_log_loss:.6f}",
            (
                "Multiclass Brier score: "
                f"{self.multiclass_brier_score:.6f}"
            ),
            "",
            "Per-class recall:",
        ]

        for model_name in self.model_names:
            lines.append(
                f"  {model_name:<22s} "
                f"{self.per_class_recall[model_name]:.6f}"
            )

        lines.extend(
            (
                "",
                "These metrics are conditional on the adopted simulations.",
                "They do not establish real-data model probabilities.",
            )
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class ModelScreeningHoldoutResult:
    """Classifier and metrics from one stratified holdout experiment."""

    classifier: LinearDiscriminantScreeningClassifier
    evaluation: ModelScreeningEvaluation
    train_indices: IntArray
    validation_indices: IntArray

    def __post_init__(self) -> None:
        """Freeze and validate the split indices."""
        train = np.asarray(
            self.train_indices,
            dtype=np.int64,
        )
        validation = np.asarray(
            self.validation_indices,
            dtype=np.int64,
        )

        if train.ndim != 1 or validation.ndim != 1:
            raise ValueError("Split indices must be one-dimensional.")

        if train.size == 0 or validation.size == 0:
            raise ValueError("Both split partitions must be non-empty.")

        if np.intersect1d(train, validation).size:
            raise ValueError(
                "Training and validation indices must be disjoint."
            )

        if validation.size != self.evaluation.n_samples:
            raise ValueError(
                "Validation size must match evaluation.n_samples."
            )

        frozen_train = np.array(train, copy=True)
        frozen_validation = np.array(validation, copy=True)
        frozen_train.setflags(write=False)
        frozen_validation.setflags(write=False)

        object.__setattr__(self, "train_indices", frozen_train)
        object.__setattr__(
            self,
            "validation_indices",
            frozen_validation,
        )


def stratified_screening_holdout(
    labels: Sequence[str],
    *,
    validation_fraction: float = 0.25,
    random_state: int = 0,
) -> tuple[IntArray, IntArray]:
    """Create deterministic stratified training and validation indices."""
    label_array = np.asarray(
        tuple(str(label) for label in labels),
        dtype=str,
    )

    if label_array.ndim != 1 or label_array.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence.")

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly in (0, 1).")

    generator = np.random.default_rng(random_state)
    train_parts: list[IntArray] = []
    validation_parts: list[IntArray] = []

    for model_name in sorted(set(label_array.tolist())):
        indices = np.flatnonzero(label_array == model_name).astype(
            np.int64,
            copy=False,
        )

        if indices.size < 2:
            raise ValueError(
                "Every model requires at least two cases for holdout "
                f"validation; {model_name!r} has {indices.size}."
            )

        shuffled = generator.permutation(indices)
        validation_count = max(
            1,
            int(np.floor(indices.size * validation_fraction)),
        )
        validation_count = min(
            validation_count,
            indices.size - 1,
        )

        validation_parts.append(shuffled[:validation_count])
        train_parts.append(shuffled[validation_count:])

    train_indices = np.concatenate(train_parts)
    validation_indices = np.concatenate(validation_parts)
    generator.shuffle(train_indices)
    generator.shuffle(validation_indices)

    train_indices.setflags(write=False)
    validation_indices.setflags(write=False)

    return train_indices, validation_indices


def evaluate_model_screening_classifier(
    classifier: LinearDiscriminantScreeningClassifier,
    features: NDArray[np.floating],
    labels: Sequence[str],
) -> ModelScreeningEvaluation:
    """Evaluate a screening classifier on an independent labelled sample."""
    feature_array = np.asarray(features, dtype=float)
    label_array = np.asarray(
        tuple(str(label) for label in labels),
        dtype=str,
    )

    expected_shape = (
        label_array.size,
        len(classifier.feature_names),
    )
    if feature_array.shape != expected_shape:
        raise ValueError(
            f"features must have shape {expected_shape}, "
            f"got {feature_array.shape}."
        )

    if label_array.size == 0:
        raise ValueError("The evaluation sample must not be empty.")

    unknown = sorted(
        set(label_array.tolist()).difference(classifier.model_names)
    )
    if unknown:
        raise ValueError(
            "Evaluation labels contain unknown models: "
            + ", ".join(unknown)
        )

    missing = sorted(
        set(classifier.model_names).difference(label_array.tolist())
    )
    if missing:
        raise ValueError(
            "Evaluation labels must include every model: "
            + ", ".join(missing)
        )

    score_matrix = np.empty(
        (
            label_array.size,
            len(classifier.model_names),
        ),
        dtype=float,
    )

    for row_index, row in enumerate(feature_array):
        feature_vector = ModelScreeningFeatureVector(
            values=row,
            names=classifier.feature_names,
        )
        scores = classifier.predict_scores(feature_vector)

        score_matrix[row_index] = [
            scores[model_name]
            for model_name in classifier.model_names
        ]

    model_index = {
        model_name: index
        for index, model_name in enumerate(classifier.model_names)
    }
    true_indices = np.asarray(
        [model_index[label] for label in label_array],
        dtype=np.int64,
    )
    predicted_indices = np.argmax(
        score_matrix,
        axis=1,
    ).astype(np.int64)

    confusion = np.zeros(
        (
            len(classifier.model_names),
            len(classifier.model_names),
        ),
        dtype=np.int64,
    )
    np.add.at(
        confusion,
        (true_indices, predicted_indices),
        1,
    )

    accuracy = float(
        np.mean(predicted_indices == true_indices)
    )

    top_count = min(
        2,
        len(classifier.model_names),
    )
    top_indices = np.argsort(
        score_matrix,
        axis=1,
    )[:, -top_count:]
    top_two_accuracy = float(
        np.mean(
            np.any(
                top_indices == true_indices[:, None],
                axis=1,
            )
        )
    )

    true_scores = score_matrix[
        np.arange(label_array.size),
        true_indices,
    ]
    mean_log_loss = float(
        -np.mean(
            np.log(
                np.clip(
                    true_scores,
                    np.finfo(float).tiny,
                    1.0,
                )
            )
        )
    )

    targets = np.zeros_like(score_matrix)
    targets[
        np.arange(label_array.size),
        true_indices,
    ] = 1.0
    multiclass_brier_score = float(
        np.mean(
            np.sum(
                np.square(score_matrix - targets),
                axis=1,
            )
        )
    )

    per_class_recall = {}
    for index, model_name in enumerate(classifier.model_names):
        class_count = int(np.sum(confusion[index]))
        per_class_recall[model_name] = (
            float(confusion[index, index] / class_count)
        )

    return ModelScreeningEvaluation(
        model_names=classifier.model_names,
        n_samples=label_array.size,
        confusion_matrix=confusion,
        accuracy=accuracy,
        top_two_accuracy=top_two_accuracy,
        mean_log_loss=mean_log_loss,
        multiclass_brier_score=multiclass_brier_score,
        per_class_recall=per_class_recall,
    )


def fit_and_evaluate_screening_classifier(
    features: NDArray[np.floating],
    labels: Sequence[str],
    *,
    feature_names: tuple[str, ...],
    validation_fraction: float = 0.25,
    random_state: int = 0,
    ridge: float = 1e-6,
) -> ModelScreeningHoldoutResult:
    """Fit and evaluate the baseline using a stratified holdout split."""
    feature_array = np.asarray(features, dtype=float)
    label_tuple = tuple(str(label) for label in labels)

    if feature_array.ndim != 2:
        raise ValueError("features must be a two-dimensional array.")

    if feature_array.shape[0] != len(label_tuple):
        raise ValueError(
            "features and labels must contain the same number of cases."
        )

    train_indices, validation_indices = stratified_screening_holdout(
        label_tuple,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )

    classifier = fit_linear_discriminant_screening_classifier(
        feature_array[train_indices],
        [label_tuple[index] for index in train_indices],
        feature_names=feature_names,
        ridge=ridge,
    )

    evaluation = evaluate_model_screening_classifier(
        classifier,
        feature_array[validation_indices],
        [label_tuple[index] for index in validation_indices],
    )

    return ModelScreeningHoldoutResult(
        classifier=classifier,
        evaluation=evaluation,
        train_indices=train_indices,
        validation_indices=validation_indices,
    )
