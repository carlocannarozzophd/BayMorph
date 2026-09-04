"""Tests for model-screening classifier persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from baymorph.analysis.model_screening_features import (
    SCREENING_FEATURE_NAMES,
    ModelScreeningFeatureVector,
)
from baymorph.validation.model_screening_classifier import (
    fit_linear_discriminant_screening_classifier,
)
from baymorph.validation.model_screening_persistence import (
    MODEL_SCREENING_ARTIFACT_VERSION,
    MODEL_SCREENING_CLASSIFIER_TYPE,
    load_model_screening_classifier,
    save_model_screening_classifier,
)


MODEL_NAMES = (
    "bulge_disk",
    "double_sersic",
    "sersic_point_source",
    "single_sersic",
)


def make_classifier():
    """Return one deterministic fitted screening classifier."""
    generator = np.random.default_rng(19)
    feature_blocks = []
    labels = []

    for model_index, model_name in enumerate(MODEL_NAMES):
        feature_blocks.append(
            generator.normal(
                loc=float(model_index * 3),
                scale=0.2,
                size=(
                    20,
                    len(SCREENING_FEATURE_NAMES),
                ),
            )
        )
        labels.extend([model_name] * 20)

    return fit_linear_discriminant_screening_classifier(
        np.vstack(feature_blocks),
        labels,
        feature_names=SCREENING_FEATURE_NAMES,
    )


def test_classifier_round_trip_preserves_scores(
    tmp_path: Path,
) -> None:
    """NPZ persistence must preserve classifier predictions exactly."""
    classifier = make_classifier()
    filename = tmp_path / "screening_classifier.npz"
    metadata = {
        "training_seed": 42,
        "scientific_scope": (
            "Simulation-conditioned validation artifact."
        ),
        "classes": list(MODEL_NAMES),
    }

    saved = save_model_screening_classifier(
        classifier,
        filename,
        metadata=metadata,
    )
    artifact = load_model_screening_classifier(
        saved
    )

    feature_vector = ModelScreeningFeatureVector(
        values=np.linspace(
            0.0,
            1.0,
            len(SCREENING_FEATURE_NAMES),
        )
    )

    assert artifact.path == filename.resolve()
    assert (
        artifact.format_version
        == MODEL_SCREENING_ARTIFACT_VERSION
    )
    assert (
        artifact.classifier_type
        == MODEL_SCREENING_CLASSIFIER_TYPE
    )
    assert artifact.metadata == metadata
    assert artifact.classifier.model_names == (
        classifier.model_names
    )
    assert artifact.classifier.feature_names == (
        classifier.feature_names
    )
    assert artifact.classifier.predict_scores(
        feature_vector
    ) == pytest.approx(
        classifier.predict_scores(
            feature_vector
        )
    )


def test_save_refuses_implicit_overwrite(
    tmp_path: Path,
) -> None:
    """Existing artifacts require explicit overwrite authorization."""
    classifier = make_classifier()
    filename = tmp_path / "screening_classifier.npz"

    save_model_screening_classifier(
        classifier,
        filename,
    )

    with pytest.raises(
        FileExistsError,
        match="artifact exists",
    ):
        save_model_screening_classifier(
            classifier,
            filename,
        )

    save_model_screening_classifier(
        classifier,
        filename,
        overwrite=True,
    )


def test_save_requires_npz_suffix(
    tmp_path: Path,
) -> None:
    """Classifier artifacts must use an explicit safe format."""
    with pytest.raises(
        ValueError,
        match=r"\.npz suffix",
    ):
        save_model_screening_classifier(
            make_classifier(),
            tmp_path / "classifier.bin",
        )


def test_load_rejects_missing_arrays(
    tmp_path: Path,
) -> None:
    """Incomplete archives must fail before classifier construction."""
    filename = tmp_path / "incomplete.npz"
    np.savez_compressed(
        filename,
        format_version=np.asarray(
            MODEL_SCREENING_ARTIFACT_VERSION
        ),
    )

    with pytest.raises(
        ValueError,
        match="Invalid model-screening artifact contents",
    ):
        load_model_screening_classifier(
            filename
        )


def test_load_rejects_unknown_format_version(
    tmp_path: Path,
) -> None:
    """Unknown schema versions must never be interpreted silently."""
    classifier = make_classifier()
    valid = tmp_path / "valid.npz"
    invalid = tmp_path / "invalid.npz"

    save_model_screening_classifier(
        classifier,
        valid,
    )

    with np.load(
        valid,
        allow_pickle=False,
    ) as archive:
        payload = {
            name: np.asarray(
                archive[name]
            )
            for name in archive.files
        }

    payload["format_version"] = np.asarray(
        MODEL_SCREENING_ARTIFACT_VERSION + 1,
        dtype=np.int64,
    )
    np.savez_compressed(
        invalid,
        **payload,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported model-screening artifact version",
    ):
        load_model_screening_classifier(
            invalid
        )
