"""Transparent baseline classifiers for model-screening validation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from baymorph.analysis.model_screening import ModelScreeningReport
from baymorph.analysis.model_screening_features import (
    SCREENING_FEATURE_NAMES,
    ModelScreeningFeatureVector,
    extract_model_screening_features,
)
from baymorph.models import registered_models
from baymorph.validation.model_screening_simulation import (
    ModelScreeningSimulationCase,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LinearDiscriminantScreeningClassifier:
    """Shared-covariance Gaussian model-screening baseline.

    This classifier belongs to the validation layer. Its normalized
    outputs are simulation-conditioned screening scores, not calibrated
    real-data posterior model probabilities.
    """

    model_names: tuple[str, ...]
    feature_names: tuple[str, ...]

    feature_location: FloatArray
    feature_scale: FloatArray

    class_centroids: FloatArray
    precision: FloatArray
    class_priors: FloatArray

    def __post_init__(self) -> None:
        """Validate and freeze the fitted classifier."""
        model_names = tuple(
            self.model_names
        )
        feature_names = tuple(
            self.feature_names
        )

        expected_models = set(
            registered_models()
        )

        if set(
            model_names
        ) != expected_models:
            raise ValueError(
                "The classifier must contain every registered model."
            )

        if len(
            set(
                model_names
            )
        ) != len(
            model_names
        ):
            raise ValueError(
                "Classifier model names must be unique."
            )

        if len(
            set(
                feature_names
            )
        ) != len(
            feature_names
        ):
            raise ValueError(
                "Classifier feature names must be unique."
            )

        location = np.asarray(
            self.feature_location,
            dtype=float,
        )
        scale = np.asarray(
            self.feature_scale,
            dtype=float,
        )
        centroids = np.asarray(
            self.class_centroids,
            dtype=float,
        )
        precision = np.asarray(
            self.precision,
            dtype=float,
        )
        class_priors = np.asarray(
            self.class_priors,
            dtype=float,
        )

        n_models = len(
            model_names
        )
        n_features = len(
            feature_names
        )

        if location.shape != (
            n_features,
        ):
            raise ValueError(
                "Classifier feature location has invalid shape."
            )

        if scale.shape != (
            n_features,
        ):
            raise ValueError(
                "Classifier feature scale has invalid shape."
            )

        if centroids.shape != (
            n_models,
            n_features,
        ):
            raise ValueError(
                "Classifier centroids have invalid shape."
            )

        if precision.shape != (
            n_features,
            n_features,
        ):
            raise ValueError(
                "Classifier precision matrix has invalid shape."
            )

        if class_priors.shape != (
            n_models,
        ):
            raise ValueError(
                "Classifier class priors have invalid shape."
            )

        arrays = (
            location,
            scale,
            centroids,
            precision,
            class_priors,
        )

        if any(
            not np.all(
                np.isfinite(
                    array
                )
            )
            for array in arrays
        ):
            raise ValueError(
                "Classifier parameters must be finite."
            )

        if np.any(
            scale
            <= 0.0
        ):
            raise ValueError(
                "Classifier feature scales must be positive."
            )

        if np.any(
            class_priors
            <= 0.0
        ):
            raise ValueError(
                "Classifier class priors must be positive."
            )

        if not np.isclose(
            np.sum(
                class_priors
            ),
            1.0,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "Classifier class priors must sum to one."
            )

        for array in arrays:
            array.setflags(
                write=False
            )

        object.__setattr__(
            self,
            "model_names",
            model_names,
        )
        object.__setattr__(
            self,
            "feature_names",
            feature_names,
        )
        object.__setattr__(
            self,
            "feature_location",
            location,
        )
        object.__setattr__(
            self,
            "feature_scale",
            scale,
        )
        object.__setattr__(
            self,
            "class_centroids",
            centroids,
        )
        object.__setattr__(
            self,
            "precision",
            precision,
        )
        object.__setattr__(
            self,
            "class_priors",
            class_priors,
        )

    def predict_scores(
        self,
        features: ModelScreeningFeatureVector,
    ) -> dict[str, float]:
        """Return normalized simulation-conditioned screening scores."""
        if features.names != self.feature_names:
            raise ValueError(
                "Feature-vector names do not match the classifier."
            )

        standardized = (
            features.values
            - self.feature_location
        ) / self.feature_scale

        linear_terms = (
            self.class_centroids
            @ self.precision
        )

        logits = (
            linear_terms
            @ standardized
            - 0.5
            * np.einsum(
                "ij,ij->i",
                linear_terms,
                self.class_centroids,
            )
            + np.log(
                self.class_priors
            )
        )

        logits -= np.max(
            logits
        )

        weights = np.exp(
            logits
        )
        normalized = (
            weights
            / np.sum(
                weights
            )
        )

        return {
            model_name: float(
                score
            )
            for model_name, score in zip(
                self.model_names,
                normalized,
                strict=True,
            )
        }

    def screen(
        self,
        features: ModelScreeningFeatureVector,
    ) -> ModelScreeningReport:
        """Return an ambiguity-safe report from the validation baseline."""
        return ModelScreeningReport.from_scores(
            self.predict_scores(
                features
            ),
            status="ambiguous",
            method="linear_discriminant_validation_baseline_v1",
            warnings=(
                "The validation baseline is not probability-calibrated; "
                "status is forced to ambiguous.",
            ),
        )


def fit_linear_discriminant_screening_classifier(
    features: NDArray[np.floating],
    labels: Sequence[str],
    *,
    feature_names: tuple[str, ...] = SCREENING_FEATURE_NAMES,
    ridge: float = 1.0e-6,
) -> LinearDiscriminantScreeningClassifier:
    """Fit a regularized shared-covariance Gaussian classifier."""
    matrix = np.asarray(
        features,
        dtype=float,
    )
    label_array = np.asarray(
        labels,
        dtype=str,
    )
    model_names = tuple(
        registered_models()
    )
    feature_names = tuple(
        feature_names
    )

    if matrix.ndim != 2:
        raise ValueError(
            "Training features must form a two-dimensional matrix."
        )

    if matrix.shape[0] != label_array.shape[0]:
        raise ValueError(
            "Training features and labels must have equal lengths."
        )

    if matrix.shape[1] != len(
        feature_names
    ):
        raise ValueError(
            "Training feature width does not match the feature names."
        )

    if not np.all(
        np.isfinite(
            matrix
        )
    ):
        raise ValueError(
            "Training features must be finite."
        )

    if set(
        label_array
    ) != set(
        model_names
    ):
        raise ValueError(
            "Training labels must contain every registered model."
        )

    ridge = float(
        ridge
    )

    if not np.isfinite(
        ridge
    ) or ridge <= 0.0:
        raise ValueError(
            "Classifier ridge must be finite and positive."
        )

    feature_location = np.mean(
        matrix,
        axis=0,
    )
    feature_scale = np.std(
        matrix,
        axis=0,
    )

    feature_scale = np.where(
        feature_scale
        > 0.0,
        feature_scale,
        1.0,
    )

    standardized = (
        matrix
        - feature_location
    ) / feature_scale

    class_centroids = np.vstack(
        [
            np.mean(
                standardized[
                    label_array
                    == model_name
                ],
                axis=0,
            )
            for model_name in model_names
        ]
    )

    residuals = np.vstack(
        [
            standardized[
                label_array
                == model_name
            ]
            - class_centroids[
                model_index
            ]
            for model_index, model_name in enumerate(
                model_names
            )
        ]
    )

    denominator = max(
        matrix.shape[0]
        - len(
            model_names
        ),
        1,
    )

    covariance = (
        residuals.T
        @ residuals
        / float(
            denominator
        )
    )

    covariance += (
        ridge
        * np.eye(
            matrix.shape[1]
        )
    )

    precision = np.linalg.pinv(
        covariance,
        hermitian=True,
    )

    class_counts = np.array(
        [
            np.count_nonzero(
                label_array
                == model_name
            )
            for model_name in model_names
        ],
        dtype=float,
    )

    class_priors = (
        class_counts
        / np.sum(
            class_counts
        )
    )

    return LinearDiscriminantScreeningClassifier(
        model_names=model_names,
        feature_names=feature_names,
        feature_location=feature_location,
        feature_scale=feature_scale,
        class_centroids=class_centroids,
        precision=precision,
        class_priors=class_priors,
    )


def screening_feature_dataset(
    cases: Sequence[
        ModelScreeningSimulationCase
    ],
) -> tuple[
    FloatArray,
    tuple[str, ...],
]:
    """Extract a feature matrix and labels from simulated cases."""
    if not cases:
        raise ValueError(
            "At least one screening simulation case is required."
        )

    feature_rows = []
    labels = []

    for case in cases:
        if case.noise_sigma <= 0.0:
            raise ValueError(
                "Classifier training cases require positive noise sigma."
            )

        features = extract_model_screening_features(
            case.image,
            target_x=case.parameters[
                "x0"
            ],
            target_y=case.parameters[
                "y0"
            ],
            psf_fwhm=case.psf_fwhm,
            background=case.parameters[
                "sky"
            ],
            noise_sigma=case.noise_sigma,
        )

        feature_rows.append(
            features.values
        )
        labels.append(
            case.model_name
        )

    return (
        np.vstack(
            feature_rows
        ),
        tuple(
            labels
        ),
    )


__all__ = [
    "LinearDiscriminantScreeningClassifier",
    "fit_linear_discriminant_screening_classifier",
    "screening_feature_dataset",
]
