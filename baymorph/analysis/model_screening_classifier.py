"""Runtime classifier for simulation-conditioned model screening."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from baymorph.analysis.model_screening import (
    SCREENING_MODELS,
    ModelScreeningReport,
)
from baymorph.analysis.model_screening_features import (
    ModelScreeningFeatureVector,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LinearDiscriminantScreeningClassifier:
    """Shared-covariance Gaussian model-screening baseline.

    This transparent classifier is usable by the runtime screening
    workflow. Its normalized outputs are simulation-conditioned screening
    scores, not calibrated real-data posterior model probabilities.
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
            SCREENING_MODELS
        )

        if set(
            model_names
        ) != expected_models:
            raise ValueError(
                "The classifier must contain every screening model."
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
        """Return an ambiguity-safe model-screening report."""
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




__all__ = [
    "LinearDiscriminantScreeningClassifier",
]
