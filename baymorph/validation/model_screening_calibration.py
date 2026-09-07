"""Probability calibration for simulation-conditioned model screening."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from baymorph.analysis.model_screening_features import (
    ModelScreeningFeatureVector,
)
from baymorph.analysis.model_screening_classifier import (
    LinearDiscriminantScreeningClassifier,
)


FloatArray = NDArray[np.float64]


def _normalized_probability_matrix(
    probabilities: NDArray[np.floating],
) -> FloatArray:
    """Validate and normalize a multiclass probability matrix."""
    values = np.asarray(
        probabilities,
        dtype=float,
    )

    if values.ndim != 2:
        raise ValueError(
            "Probability values must form a two-dimensional matrix."
        )

    if values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError(
            "Probability values require cases and at least two classes."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Probability values must be finite."
        )

    if np.any(values < 0.0):
        raise ValueError(
            "Probability values must be non-negative."
        )

    row_sums = np.sum(
        values,
        axis=1,
        keepdims=True,
    )
    if np.any(row_sums <= 0.0):
        raise ValueError(
            "Every probability row must have positive mass."
        )

    return values / row_sums


def calibrate_probability_matrix(
    probabilities: NDArray[np.floating],
    *,
    temperature: float,
) -> FloatArray:
    """Apply scalar temperature calibration to probability scores."""
    normalized = _normalized_probability_matrix(
        probabilities
    )
    temperature = float(temperature)

    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(
            "Calibration temperature must be finite and positive."
        )

    log_probabilities = np.log(
        np.clip(
            normalized,
            np.finfo(float).tiny,
            1.0,
        )
    )
    scaled = log_probabilities / temperature
    calibrated = np.exp(
        scaled
        - logsumexp(
            scaled,
            axis=1,
            keepdims=True,
        )
    )

    return np.asarray(
        calibrated,
        dtype=float,
    )


def _true_class_indices(
    labels: Sequence[str],
    model_names: tuple[str, ...],
    *,
    require_every_model: bool,
) -> NDArray[np.int64]:
    """Map class labels to ordered integer indices."""
    if not model_names:
        raise ValueError(
            "model_names must not be empty."
        )

    if len(set(model_names)) != len(model_names):
        raise ValueError(
            "model_names must be unique."
        )

    label_tuple = tuple(
        str(label)
        for label in labels
    )
    if not label_tuple:
        raise ValueError(
            "labels must not be empty."
        )

    model_index = {
        model_name: index
        for index, model_name in enumerate(model_names)
    }
    unknown = sorted(
        set(label_tuple).difference(model_names)
    )
    if unknown:
        raise ValueError(
            "Labels contain unknown models: "
            + ", ".join(unknown)
        )

    if require_every_model:
        missing = sorted(
            set(model_names).difference(label_tuple)
        )
        if missing:
            raise ValueError(
                "Calibration labels must include every model: "
                + ", ".join(missing)
            )

    return np.asarray(
        [
            model_index[label]
            for label in label_tuple
        ],
        dtype=np.int64,
    )


@dataclass(frozen=True, slots=True)
class ProbabilityCalibrationMetrics:
    """Calibration-sensitive metrics for one labelled sample."""

    n_cases: int
    accuracy: float
    mean_log_loss: float
    multiclass_brier_score: float
    expected_calibration_error: float

    def __post_init__(self) -> None:
        """Validate metric ranges."""
        if self.n_cases <= 0:
            raise ValueError(
                "n_cases must be positive."
            )

        for name, value in (
            ("accuracy", self.accuracy),
            (
                "expected_calibration_error",
                self.expected_calibration_error,
            ),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must lie in [0, 1]."
                )

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


def evaluate_probability_calibration(
    probabilities: NDArray[np.floating],
    labels: Sequence[str],
    *,
    model_names: tuple[str, ...],
    n_bins: int = 10,
) -> ProbabilityCalibrationMetrics:
    """Evaluate probability calibration on labelled simulations."""
    normalized = _normalized_probability_matrix(
        probabilities
    )
    truth = _true_class_indices(
        labels,
        model_names,
        require_every_model=False,
    )

    if normalized.shape != (
        truth.size,
        len(model_names),
    ):
        raise ValueError(
            "Probability shape must match labels and model_names."
        )

    n_bins = int(n_bins)
    if n_bins < 2:
        raise ValueError(
            "n_bins must be at least two."
        )

    prediction = np.argmax(
        normalized,
        axis=1,
    )
    confidence = np.max(
        normalized,
        axis=1,
    )
    accuracy = float(
        np.mean(
            prediction == truth
        )
    )

    true_probabilities = normalized[
        np.arange(truth.size),
        truth,
    ]
    mean_log_loss = float(
        -np.mean(
            np.log(
                np.clip(
                    true_probabilities,
                    np.finfo(float).tiny,
                    1.0,
                )
            )
        )
    )

    targets = np.zeros_like(
        normalized
    )
    targets[
        np.arange(truth.size),
        truth,
    ] = 1.0
    brier = float(
        np.mean(
            np.sum(
                np.square(
                    normalized - targets
                ),
                axis=1,
            )
        )
    )

    bin_indices = np.minimum(
        (
            confidence
            * n_bins
        ).astype(int),
        n_bins - 1,
    )
    calibration_error = 0.0

    for bin_index in range(n_bins):
        selected = (
            bin_indices == bin_index
        )
        count = int(
            np.count_nonzero(selected)
        )
        if count == 0:
            continue

        bin_confidence = float(
            np.mean(
                confidence[selected]
            )
        )
        bin_accuracy = float(
            np.mean(
                prediction[selected]
                == truth[selected]
            )
        )
        calibration_error += (
            count
            / truth.size
            * abs(
                bin_accuracy
                - bin_confidence
            )
        )

    return ProbabilityCalibrationMetrics(
        n_cases=truth.size,
        accuracy=accuracy,
        mean_log_loss=mean_log_loss,
        multiclass_brier_score=brier,
        expected_calibration_error=float(
            calibration_error
        ),
    )


@dataclass(frozen=True, slots=True)
class TemperatureCalibration:
    """Scalar calibration fitted on an independent simulation sample."""

    model_names: tuple[str, ...]
    temperature: float
    before: ProbabilityCalibrationMetrics
    after: ProbabilityCalibrationMetrics
    method: str = "temperature_scaling_v1"

    def __post_init__(self) -> None:
        """Validate the calibration contract."""
        if not self.model_names:
            raise ValueError(
                "model_names must not be empty."
            )

        if len(set(self.model_names)) != len(self.model_names):
            raise ValueError(
                "model_names must be unique."
            )

        if (
            not np.isfinite(self.temperature)
            or self.temperature <= 0.0
        ):
            raise ValueError(
                "temperature must be finite and positive."
            )

        if self.before.n_cases != self.after.n_cases:
            raise ValueError(
                "Before and after metrics must use the same cases."
            )

    def calibrate_scores(
        self,
        scores: Mapping[str, float],
    ) -> dict[str, float]:
        """Calibrate one ordered model-score mapping."""
        supplied = set(scores)
        expected = set(self.model_names)

        if supplied != expected:
            missing = sorted(
                expected - supplied
            )
            extra = sorted(
                supplied - expected
            )
            raise ValueError(
                "Scores must contain every calibrated model exactly once. "
                f"Missing: {missing}. Extra: {extra}."
            )

        matrix = np.asarray(
            [
                [
                    float(
                        scores[model_name]
                    )
                    for model_name in self.model_names
                ]
            ],
            dtype=float,
        )
        calibrated = calibrate_probability_matrix(
            matrix,
            temperature=self.temperature,
        )[0]

        return {
            model_name: float(
                calibrated[index]
            )
            for index, model_name in enumerate(self.model_names)
        }

    def summary(self) -> str:
        """Return a concise calibration summary."""
        return "\n".join(
            (
                "BayMorph Model Screening Calibration",
                "",
                f"Method: {self.method}",
                f"Calibration cases: {self.before.n_cases}",
                f"Temperature: {self.temperature:.6f}",
                (
                    "Mean log loss: "
                    f"{self.before.mean_log_loss:.6f} -> "
                    f"{self.after.mean_log_loss:.6f}"
                ),
                (
                    "Multiclass Brier score: "
                    f"{self.before.multiclass_brier_score:.6f} -> "
                    f"{self.after.multiclass_brier_score:.6f}"
                ),
                (
                    "Expected calibration error: "
                    f"{self.before.expected_calibration_error:.6f} -> "
                    f"{self.after.expected_calibration_error:.6f}"
                ),
                "",
                "Calibration is conditional on the adopted simulations.",
                "Scores are not real-data posterior model probabilities.",
            )
        )


def fit_temperature_calibration(
    probabilities: NDArray[np.floating],
    labels: Sequence[str],
    *,
    model_names: tuple[str, ...],
    n_bins: int = 10,
    temperature_bounds: tuple[float, float] = (
        0.05,
        20.0,
    ),
) -> TemperatureCalibration:
    """Fit temperature scaling on an independent calibration sample."""
    normalized = _normalized_probability_matrix(
        probabilities
    )
    truth = _true_class_indices(
        labels,
        model_names,
        require_every_model=True,
    )

    if normalized.shape != (
        truth.size,
        len(model_names),
    ):
        raise ValueError(
            "Probability shape must match labels and model_names."
        )

    lower, upper = (
        float(value)
        for value in temperature_bounds
    )
    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or lower <= 0.0
        or upper <= lower
    ):
        raise ValueError(
            "temperature_bounds must define a positive interval."
        )

    log_probabilities = np.log(
        np.clip(
            normalized,
            np.finfo(float).tiny,
            1.0,
        )
    )

    def objective(
        log_temperature: float,
    ) -> float:
        temperature = float(
            np.exp(log_temperature)
        )
        scaled = (
            log_probabilities
            / temperature
        )
        true_log_scores = scaled[
            np.arange(truth.size),
            truth,
        ]
        log_normalization = logsumexp(
            scaled,
            axis=1,
        )

        return float(
            -np.mean(
                true_log_scores
                - log_normalization
            )
        )

    optimization = minimize_scalar(
        objective,
        bounds=(
            float(np.log(lower)),
            float(np.log(upper)),
        ),
        method="bounded",
        options={
            "xatol": 1e-10,
        },
    )

    if not optimization.success:
        raise RuntimeError(
            "Temperature calibration optimization failed: "
            f"{optimization.message}"
        )

    temperature = float(
        np.exp(
            optimization.x
        )
    )
    calibrated = calibrate_probability_matrix(
        normalized,
        temperature=temperature,
    )

    before = evaluate_probability_calibration(
        normalized,
        labels,
        model_names=model_names,
        n_bins=n_bins,
    )
    after = evaluate_probability_calibration(
        calibrated,
        labels,
        model_names=model_names,
        n_bins=n_bins,
    )

    if after.mean_log_loss > before.mean_log_loss:
        temperature = 1.0
        calibrated = normalized.copy()
        after = evaluate_probability_calibration(
            calibrated,
            labels,
            model_names=model_names,
            n_bins=n_bins,
        )

    return TemperatureCalibration(
        model_names=model_names,
        temperature=temperature,
        before=before,
        after=after,
    )


def classifier_probability_matrix(
    classifier: LinearDiscriminantScreeningClassifier,
    features: NDArray[np.floating],
) -> FloatArray:
    """Evaluate one BayMorph screening classifier as a score matrix."""
    feature_array = np.asarray(
        features,
        dtype=float,
    )
    expected_shape = (
        feature_array.shape[0],
        len(classifier.feature_names),
    )

    if (
        feature_array.ndim != 2
        or feature_array.shape != expected_shape
    ):
        raise ValueError(
            "features must be a two-dimensional matrix matching "
            "the classifier feature contract."
        )

    probabilities = np.empty(
        (
            feature_array.shape[0],
            len(classifier.model_names),
        ),
        dtype=float,
    )

    for row_index, row in enumerate(feature_array):
        scores = classifier.predict_scores(
            ModelScreeningFeatureVector(
                values=row,
                names=classifier.feature_names,
            )
        )
        probabilities[row_index] = [
            scores[model_name]
            for model_name in classifier.model_names
        ]

    return probabilities


__all__ = [
    "ProbabilityCalibrationMetrics",
    "TemperatureCalibration",
    "calibrate_probability_matrix",
    "classifier_probability_matrix",
    "evaluate_probability_calibration",
    "fit_temperature_calibration",
]
