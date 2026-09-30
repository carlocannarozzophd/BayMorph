"""Backend-neutral inference-result representation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from baymorph.results.evidence import EvidenceEstimate
from baymorph.results.samples import PosteriorSamples


_FORMAT_VERSION = 2


def json_compatible_metadata(
    value: Any,
) -> Any:
    """Convert nested metadata into stable JSON-compatible values.

    JSON-native values are preserved exactly. NumPy arrays and scalar
    values are converted to their corresponding Python representations.
    Runtime Python objects that cannot be represented faithfully in JSON
    are described explicitly rather than converted to unstable ``repr``
    strings.
    """
    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): json_compatible_metadata(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            json_compatible_metadata(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        np.random.Generator,
    ):
        return {
            "__runtime_object__": True,
            "type": "numpy.random.Generator",
            "bit_generator": type(
                value.bit_generator
            ).__name__,
        }

    try:
        json.dumps(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        value_type = type(
            value
        )

        return {
            "__runtime_object__": True,
            "type": (
                f"{value_type.__module__}."
                f"{value_type.__qualname__}"
            ),
        }

    return value


@dataclass
class InferenceResult:
    """Canonical BayMorph result returned by inference backends.

    Parameters
    ----------
    posterior : PosteriorSamples
        Backend-neutral weighted posterior representation.
    evidence : EvidenceEstimate, optional
        Marginal-likelihood estimate produced by the inference backend,
        when available.
    diagnostics : dict, optional
        Backend-specific runtime and convergence diagnostics.
    provenance : dict, optional
        Information identifying the backend and numerical configuration
        that produced the result.

    Notes
    -----
    BayMorph inference backends may use fundamentally different numerical
    algorithms internally. This object standardizes only scientifically
    meaningful outputs and intentionally excludes sampler-specific state.
    """

    posterior: PosteriorSamples
    evidence: EvidenceEstimate | None = None
    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )
    provenance: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """Validate the canonical inference result."""
        if not isinstance(
            self.posterior,
            PosteriorSamples,
        ):
            raise TypeError(
                "posterior must be a PosteriorSamples instance."
            )

        if (
            self.evidence is not None
            and not isinstance(
                self.evidence,
                EvidenceEstimate,
            )
        ):
            raise TypeError(
                "evidence must be an EvidenceEstimate instance or None."
            )

        self.diagnostics = dict(
            self.diagnostics
        )
        self.provenance = dict(
            self.provenance
        )

    @property
    def samples(self) -> np.ndarray:
        """Return posterior samples."""
        return self.posterior.samples

    @property
    def weights(self) -> np.ndarray:
        """Return normalized posterior weights."""
        return self.posterior.weights

    @property
    def equal_weight_samples(self) -> np.ndarray:
        """Return deterministic equal-weight posterior samples.

        Equal-weight samples are a derived product intended for visualization.
        The weighted samples remain the statistical source of truth for every
        summary, credible interval, and evidence quantity.
        """
        return self.posterior.equal_weight_samples

    @property
    def parameter_names(self) -> list[str]:
        """Return parameter names in sampling order."""
        return self.posterior.parameter_names

    @property
    def log_likelihood(self) -> np.ndarray | None:
        """Return posterior-sample log-likelihood values."""
        return self.posterior.log_likelihood

    @property
    def log_prior(self) -> np.ndarray | None:
        """Return posterior-sample log-prior values."""
        return self.posterior.log_prior

    @property
    def effective_sample_size(self) -> float:
        """Return the posterior-weight effective sample size."""
        return self.posterior.effective_sample_size

    def summary(
        self,
        parameter_name: str,
    ) -> dict[str, float]:
        """Return weighted posterior summary statistics."""
        return self.posterior.summary(
            parameter_name
        )

    def mahalanobis_distance(
        self,
        reference: Mapping[str, float],
    ) -> dict[str, Any]:
        """Measure how far a reference parameter vector lies from the posterior.

        Parameters
        ----------
        reference : mapping of str to float
            Reference values by parameter name, typically a simulation truth.
            Names absent from the posterior are ignored, and posterior
            parameters absent from the mapping are excluded from the metric.

        Returns
        -------
        dict
            ``parameters`` used, in sampling order, together with
            ``distance_squared``, ``distance``, ``dof``, and ``p_value``.

        Raises
        ------
        ValueError
            If no parameter name is shared with the posterior, if a reference
            value is not finite, or if the posterior covariance is singular.

        Notes
        -----
        The metric is computed from the **weighted** samples, which are the
        statistical source of truth, not from derived equal-weight samples.

        Correlated parameters must be assessed jointly. Reading marginal
        one-parameter offsets of a correlated posterior double-counts a single
        shared displacement and overstates the tension.

        ``p_value`` is the chi-squared survival probability at ``dof`` degrees
        of freedom. It assumes an approximately Gaussian posterior. The
        distance itself remains defined for a non-Gaussian posterior, but its
        ``p_value`` is then not trustworthy.

        A single realization does not test calibration. One draw yielding a
        small ``p_value`` is expected at the stated rate, so distinguishing an
        unlucky realization from a genuine bias requires repeated experiments
        with independent noise.
        """
        from scipy import stats as scipy_stats

        from baymorph.stats.weighted import (
            mahalanobis_distance_squared,
        )

        names = [
            name
            for name in self.parameter_names
            if name in reference
        ]

        if not names:
            raise ValueError(
                "reference shares no parameter name with the posterior."
            )

        indices = [
            self.parameter_names.index(
                name
            )
            for name in names
        ]

        distance_squared = mahalanobis_distance_squared(
            self.samples[:, indices],
            self.weights,
            np.asarray(
                [
                    float(
                        reference[name]
                    )
                    for name in names
                ],
                dtype=float,
            ),
        )

        dof = len(names)

        return {
            "parameters": names,
            "distance_squared": distance_squared,
            "distance": float(
                np.sqrt(
                    distance_squared
                )
            ),
            "dof": dof,
            "p_value": float(
                scipy_stats.chi2.sf(
                    distance_squared,
                    dof,
                )
            ),
        }

    @property
    def best_index(self) -> int:
        """Return the maximum-likelihood posterior-sample index."""
        return self.posterior.best_index

    @property
    def best_sample(self) -> np.ndarray:
        """Return the maximum-likelihood parameter vector."""
        return self.posterior.best_sample

    @property
    def best_parameters(self) -> dict[str, float]:
        """Return maximum-likelihood parameters by name."""
        return self.posterior.best_parameters

    @property
    def median_parameters(self) -> dict[str, float]:
        """Return posterior-median parameters by name."""
        return self.posterior.median_parameters

    @property
    def logz(self) -> float | None:
        """Return the logarithmic Bayesian evidence when available."""
        if self.evidence is None:
            return None

        return self.evidence.logz

    @property
    def logz_error(self) -> float | None:
        """Return the reported evidence uncertainty when available."""
        if self.evidence is None:
            return None

        return self.evidence.uncertainty

    @property
    def evidence_method(self) -> str | None:
        """Return the evidence-estimation method when available."""
        if self.evidence is None:
            return None

        return self.evidence.method

    def save(
        self,
        path: str | Path,
    ) -> Path:
        """Serialize a canonical BayMorph inference result."""
        output_path = Path(path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        arrays: dict[str, np.ndarray] = {
            "format_version": np.asarray(
                _FORMAT_VERSION,
                dtype=int,
            ),
            "samples": np.asarray(
                self.samples,
                dtype=float,
            ),
            "weights": np.asarray(
                self.weights,
                dtype=float,
            ),
            "parameter_names": np.asarray(
                self.parameter_names,
                dtype=str,
            ),
            "diagnostics_json": np.asarray(
                json.dumps(
                    json_compatible_metadata(
                        self.diagnostics
                    )
                )
            ),
            "provenance_json": np.asarray(
                json.dumps(
                    json_compatible_metadata(
                        self.provenance
                    )
                )
            ),
        }

        if self.log_likelihood is not None:
            arrays[
                "log_likelihood"
            ] = np.asarray(
                self.log_likelihood,
                dtype=float,
            )

        if self.log_prior is not None:
            arrays[
                "log_prior"
            ] = np.asarray(
                self.log_prior,
                dtype=float,
            )

        if self.evidence is not None:
            arrays[
                "evidence_logz"
            ] = np.asarray(
                self.evidence.logz,
                dtype=float,
            )

            arrays[
                "evidence_method"
            ] = np.asarray(
                self.evidence.method
            )

            arrays[
                "evidence_diagnostics_json"
            ] = np.asarray(
                json.dumps(
                    json_compatible_metadata(
                        self.evidence.diagnostics
                    )
                )
            )

            arrays[
                "evidence_provenance_json"
            ] = np.asarray(
                json.dumps(
                    json_compatible_metadata(
                        self.evidence.provenance
                    )
                )
            )

            if self.evidence.uncertainty is not None:
                arrays[
                    "evidence_uncertainty"
                ] = np.asarray(
                    self.evidence.uncertainty,
                    dtype=float,
                )

            if (
                self.evidence.uncertainty_kind
                is not None
            ):
                arrays[
                    "evidence_uncertainty_kind"
                ] = np.asarray(
                    self.evidence.uncertainty_kind
                )

        np.savez_compressed(
            output_path,
            **arrays,
        )

        return output_path

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> InferenceResult:
        """Load a canonical BayMorph inference-result archive."""
        input_path = Path(path)

        with np.load(
            input_path,
            allow_pickle=False,
        ) as archive:
            files = set(
                archive.files
            )

            if "format_version" not in files:
                raise ValueError(
                    f"{input_path} is not a BayMorph inference-result "
                    "archive: it has no 'format_version' entry. Legacy "
                    "posterior archives are no longer readable; rerun the "
                    "fit to produce the canonical format."
                )

            return cls._load_v2(
                archive
            )

    @classmethod
    def _load_v2(
        cls,
        archive,
    ) -> InferenceResult:
        """Load the backend-neutral BayMorph result format."""
        version = int(
            np.asarray(
                archive["format_version"]
            )
        )

        if version != _FORMAT_VERSION:
            raise ValueError(
                "Unsupported BayMorph inference-result "
                f"format version {version}."
            )

        files = set(
            archive.files
        )

        posterior = PosteriorSamples(
            samples=np.asarray(
                archive["samples"],
                dtype=float,
            ),
            weights=np.asarray(
                archive["weights"],
                dtype=float,
            ),
            parameter_names=[
                str(name)
                for name in archive[
                    "parameter_names"
                ]
            ],
            log_likelihood=(
                np.asarray(
                    archive["log_likelihood"],
                    dtype=float,
                )
                if "log_likelihood" in files
                else None
            ),
            log_prior=(
                np.asarray(
                    archive["log_prior"],
                    dtype=float,
                )
                if "log_prior" in files
                else None
            ),
        )

        evidence = None

        if "evidence_logz" in files:
            evidence = EvidenceEstimate(
                logz=float(
                    np.asarray(
                        archive[
                            "evidence_logz"
                        ]
                    )
                ),
                uncertainty=(
                    float(
                        np.asarray(
                            archive[
                                "evidence_uncertainty"
                            ]
                        )
                    )
                    if (
                        "evidence_uncertainty"
                        in files
                    )
                    else None
                ),
                method=str(
                    np.asarray(
                        archive[
                            "evidence_method"
                        ]
                    )
                ),
                uncertainty_kind=(
                    str(
                        np.asarray(
                            archive[
                                "evidence_uncertainty_kind"
                            ]
                        )
                    )
                    if (
                        "evidence_uncertainty_kind"
                        in files
                    )
                    else None
                ),
                diagnostics=json.loads(
                    str(
                        np.asarray(
                            archive[
                                "evidence_diagnostics_json"
                            ]
                        )
                    )
                ),
                provenance=json.loads(
                    str(
                        np.asarray(
                            archive[
                                "evidence_provenance_json"
                            ]
                        )
                    )
                ),
            )

        return cls(
            posterior=posterior,
            evidence=evidence,
            diagnostics=json.loads(
                str(
                    np.asarray(
                        archive[
                            "diagnostics_json"
                        ]
                    )
                )
            ),
            provenance=json.loads(
                str(
                    np.asarray(
                        archive[
                            "provenance_json"
                        ]
                    )
                )
            ),
        )
