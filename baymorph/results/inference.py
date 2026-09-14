"""Backend-neutral inference-result representation."""

from __future__ import annotations

import json
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
    def effective_sample_size(self) -> float | None:
        """Return the backend-appropriate effective sample size.

        Importance-sampled backends use the posterior-weight effective
        sample size. For correlated emcee chains, the autocorrelation-based
        estimate is returned only when it is available.
        """
        backend = str(
            self.provenance.get(
                "backend",
                "",
            )
        ).strip().lower()

        if backend == "emcee":
            value = self.diagnostics.get(
                "mcmc_effective_sample_size"
            )

            if value is None:
                return None

            return float(value)

        return self.posterior.effective_sample_size

    def summary(
        self,
        parameter_name: str,
    ) -> dict[str, float]:
        """Return weighted posterior summary statistics."""
        return self.posterior.summary(
            parameter_name
        )

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
        """Load a canonical or legacy BayMorph inference result."""
        input_path = Path(path)

        with np.load(
            input_path,
            allow_pickle=False,
        ) as archive:
            files = set(
                archive.files
            )

            if "format_version" in files:
                return cls._load_v2(
                    archive
                )

        return cls._load_legacy(
            input_path
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

    @classmethod
    def _load_legacy(
        cls,
        path: Path,
    ) -> InferenceResult:
        """Migrate the historical posterior archive format on load."""
        with np.load(
            path,
            allow_pickle=False,
        ) as archive:
            files = set(
                archive.files
            )

            required = {
                "samples",
                "logwt",
                "logz",
                "parameter_names",
            }
            missing = sorted(
                required - files
            )

            if missing:
                names = ", ".join(
                    missing
                )
                raise ValueError(
                    "Legacy posterior archive is missing required "
                    f"array(s): {names}."
                )

            samples = np.asarray(
                archive["samples"],
                dtype=float,
            )
            logwt = np.asarray(
                archive["logwt"],
                dtype=float,
            ).reshape(-1)
            logz = np.asarray(
                archive["logz"],
                dtype=float,
            ).reshape(-1)

            parameter_names = [
                str(name)
                for name in np.asarray(
                    archive[
                        "parameter_names"
                    ]
                )
            ]

            log_likelihood = (
                np.asarray(
                    archive["logl"],
                    dtype=float,
                )
                if "logl" in files
                else None
            )

            logzerr = (
                np.asarray(
                    archive["logzerr"],
                    dtype=float,
                ).reshape(-1)
                if "logzerr" in files
                else None
            )

        if logz.size == 0:
            raise ValueError(
                "Legacy posterior evidence history must not be empty."
            )

        if logwt.size != samples.shape[0]:
            raise ValueError(
                "Legacy posterior weights must match the number of samples."
            )

        final_logz = float(
            logz[-1]
        )

        weights = np.exp(
            logwt - final_logz
        )

        normalization = float(
            np.sum(
                weights
            )
        )

        if (
            not np.isfinite(
                normalization
            )
            or normalization <= 0.0
        ):
            raise ValueError(
                "Legacy posterior weights cannot be normalized."
            )

        weights = (
            weights
            / normalization
        )

        uncertainty = None

        if logzerr is not None:
            if logzerr.size == 0:
                raise ValueError(
                    "Legacy posterior evidence uncertainty "
                    "history must not be empty."
                )

            uncertainty = float(
                logzerr[-1]
            )

        posterior = PosteriorSamples(
            samples=samples,
            weights=weights,
            parameter_names=parameter_names,
            log_likelihood=log_likelihood,
        )

        evidence = EvidenceEstimate(
            logz=final_logz,
            uncertainty=uncertainty,
            method="legacy_baymorph",
            uncertainty_kind=(
                None
                if uncertainty is None
                else "legacy_backend_internal"
            ),
            provenance={
                "format": "posterior_result_v1",
            },
        )

        return cls(
            posterior=posterior,
            evidence=evidence,
            provenance={
                "source_format": "posterior_result_v1",
            },
        )
