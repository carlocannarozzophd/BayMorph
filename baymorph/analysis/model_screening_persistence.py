"""Safe persistence for BayMorph model-screening classifiers."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from baymorph.analysis.model_screening_classifier import (
    LinearDiscriminantScreeningClassifier,
)


MODEL_SCREENING_ARTIFACT_VERSION = 1
MODEL_SCREENING_CLASSIFIER_TYPE = "linear_discriminant_v1"

_REQUIRED_ARRAYS = {
    "format_version",
    "classifier_type",
    "model_names",
    "feature_names",
    "feature_location",
    "feature_scale",
    "class_centroids",
    "precision",
    "class_priors",
    "metadata_json",
}


@dataclass(frozen=True, slots=True)
class ModelScreeningClassifierArtifact:
    """Loaded classifier together with its persisted provenance."""

    path: Path
    format_version: int
    classifier_type: str
    classifier: LinearDiscriminantScreeningClassifier
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate and freeze artifact metadata."""
        if self.format_version != MODEL_SCREENING_ARTIFACT_VERSION:
            raise ValueError(
                "Unsupported model-screening artifact version: "
                f"{self.format_version}."
            )

        if self.classifier_type != MODEL_SCREENING_CLASSIFIER_TYPE:
            raise ValueError(
                "Unsupported model-screening classifier type: "
                f"{self.classifier_type!r}."
            )

        object.__setattr__(
            self,
            "path",
            Path(self.path).resolve(),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                dict(self.metadata)
            ),
        )


def save_model_screening_classifier(
    classifier: LinearDiscriminantScreeningClassifier,
    filename: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Persist one classifier as a non-pickle compressed NPZ artifact."""
    destination = Path(filename)

    if destination.suffix.lower() != ".npz":
        raise ValueError(
            "Model-screening classifier artifacts must use the .npz suffix."
        )

    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Model-screening classifier artifact exists: {destination}"
        )

    metadata_payload = dict(
        metadata
        if metadata is not None
        else {}
    )

    try:
        metadata_json = json.dumps(
            metadata_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "Model-screening artifact metadata must be JSON serializable."
        ) from exc

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_name: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f"{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary_name = Path(
                stream.name
            )

            np.savez_compressed(
                stream,
                format_version=np.asarray(
                    MODEL_SCREENING_ARTIFACT_VERSION,
                    dtype=np.int64,
                ),
                classifier_type=np.asarray(
                    MODEL_SCREENING_CLASSIFIER_TYPE,
                    dtype=np.str_,
                ),
                model_names=np.asarray(
                    classifier.model_names,
                    dtype=np.str_,
                ),
                feature_names=np.asarray(
                    classifier.feature_names,
                    dtype=np.str_,
                ),
                feature_location=np.asarray(
                    classifier.feature_location,
                    dtype=float,
                ),
                feature_scale=np.asarray(
                    classifier.feature_scale,
                    dtype=float,
                ),
                class_centroids=np.asarray(
                    classifier.class_centroids,
                    dtype=float,
                ),
                precision=np.asarray(
                    classifier.precision,
                    dtype=float,
                ),
                class_priors=np.asarray(
                    classifier.class_priors,
                    dtype=float,
                ),
                metadata_json=np.asarray(
                    metadata_json,
                    dtype=np.str_,
                ),
            )

        temporary_name.replace(
            destination
        )
    except Exception:
        if (
            temporary_name is not None
            and temporary_name.exists()
        ):
            temporary_name.unlink()
        raise

    return destination.resolve()


def load_model_screening_classifier(
    filename: str | Path,
) -> ModelScreeningClassifierArtifact:
    """Load and validate one non-pickle classifier artifact."""
    source = Path(filename)

    if not source.exists():
        raise FileNotFoundError(
            f"Model-screening classifier artifact not found: {source}"
        )

    try:
        archive_context = np.load(
            source,
            allow_pickle=False,
        )
    except Exception as exc:
        raise ValueError(
            "Could not read the model-screening classifier artifact."
        ) from exc

    with archive_context as archive:
        supplied = set(
            archive.files
        )
        if supplied != _REQUIRED_ARRAYS:
            missing = sorted(
                _REQUIRED_ARRAYS - supplied
            )
            extra = sorted(
                supplied - _REQUIRED_ARRAYS
            )
            raise ValueError(
                "Invalid model-screening artifact contents. "
                f"Missing: {missing}. Extra: {extra}."
            )

        format_version = int(
            np.asarray(
                archive["format_version"]
            ).item()
        )
        classifier_type = str(
            np.asarray(
                archive["classifier_type"]
            ).item()
        )

        if format_version != MODEL_SCREENING_ARTIFACT_VERSION:
            raise ValueError(
                "Unsupported model-screening artifact version: "
                f"{format_version}."
            )

        if classifier_type != MODEL_SCREENING_CLASSIFIER_TYPE:
            raise ValueError(
                "Unsupported model-screening classifier type: "
                f"{classifier_type!r}."
            )

        model_names = tuple(
            str(value)
            for value in np.asarray(
                archive["model_names"]
            ).tolist()
        )
        feature_names = tuple(
            str(value)
            for value in np.asarray(
                archive["feature_names"]
            ).tolist()
        )

        classifier = LinearDiscriminantScreeningClassifier(
            model_names=model_names,
            feature_names=feature_names,
            feature_location=np.asarray(
                archive["feature_location"],
                dtype=float,
            ),
            feature_scale=np.asarray(
                archive["feature_scale"],
                dtype=float,
            ),
            class_centroids=np.asarray(
                archive["class_centroids"],
                dtype=float,
            ),
            precision=np.asarray(
                archive["precision"],
                dtype=float,
            ),
            class_priors=np.asarray(
                archive["class_priors"],
                dtype=float,
            ),
        )

        metadata_text = str(
            np.asarray(
                archive["metadata_json"]
            ).item()
        )

    try:
        metadata = json.loads(
            metadata_text
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Model-screening artifact metadata are not valid JSON."
        ) from exc

    if not isinstance(
        metadata,
        dict,
    ):
        raise ValueError(
            "Model-screening artifact metadata must be a JSON object."
        )

    return ModelScreeningClassifierArtifact(
        path=source,
        format_version=format_version,
        classifier_type=classifier_type,
        classifier=classifier,
        metadata=metadata,
    )


__all__ = [
    "MODEL_SCREENING_ARTIFACT_VERSION",
    "MODEL_SCREENING_CLASSIFIER_TYPE",
    "ModelScreeningClassifierArtifact",
    "load_model_screening_classifier",
    "save_model_screening_classifier",
]
