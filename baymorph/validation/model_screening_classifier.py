"""Training utilities for the model-screening validation classifier."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from baymorph.analysis.model_screening_classifier import (
    LinearDiscriminantScreeningClassifier,
)
from baymorph.analysis.model_screening_features import (
    SCREENING_FEATURE_NAMES,
    extract_model_screening_features,
)
from baymorph.models import registered_models
from baymorph.validation.model_screening_simulation import (
    ModelScreeningSimulationCase,
)


FloatArray = NDArray[np.float64]


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
    "fit_linear_discriminant_screening_classifier",
    "screening_feature_dataset",
]
