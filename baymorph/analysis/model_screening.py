"""Result contracts for simulation-conditioned model screening."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isclose, isfinite


SCREENING_MODELS = (
    "bulge_disk",
    "double_sersic",
    "sersic_point_source",
    "single_sersic",
)
"""Morphology models the screening was designed and trained to rank.

These are the four models the registry held when the screening was built,
in the order it listed them. Models registered later, such as those of
AD-048, are outside the declared candidates of the screening, so its
simulations, classifier and reports do not change with the registry.
"""

SCREENING_STATUSES = (
    "recommended",
    "ambiguous",
    "out_of_distribution",
    "not_assessable",
)


@dataclass(frozen=True, slots=True)
class ModelScreeningScore:
    """Normalized screening score for one BayMorph model."""

    model_name: str
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class ModelScreeningReport:
    """Human-reviewable model-screening result.

    Screening scores are conditional on the simulation and training
    distributions. They do not automatically select a production model.
    """

    scores: tuple[ModelScreeningScore, ...]
    status: str
    method: str
    warnings: tuple[str, ...] = ()
    score_semantics: str = "simulation_conditioned"
    user_decision_required: bool = True

    def __post_init__(self) -> None:
        """Validate the complete screening-result contract."""
        if self.status not in SCREENING_STATUSES:
            raise ValueError(
                f"Unknown model-screening status: {self.status!r}."
            )

        if not self.method.strip():
            raise ValueError(
                "Model-screening method must be a non-empty string."
            )

        if not self.user_decision_required:
            raise ValueError(
                "BayMorph model screening must require a user decision."
            )

        expected_models = set(SCREENING_MODELS)
        supplied_models = {
            item.model_name
            for item in self.scores
        }

        if supplied_models != expected_models:
            missing = sorted(
                expected_models - supplied_models
            )
            extra = sorted(
                supplied_models - expected_models
            )

            raise ValueError(
                "A screening report must contain every screening model. "
                f"Missing: {missing}. Extra: {extra}."
            )

        if len(supplied_models) != len(self.scores):
            raise ValueError(
                "A screening report cannot contain duplicate models."
            )

        expected_ranks = tuple(
            range(
                1,
                len(self.scores) + 1,
            )
        )
        supplied_ranks = tuple(
            item.rank
            for item in self.scores
        )

        if supplied_ranks != expected_ranks:
            raise ValueError(
                "Screening ranks must be consecutive and start at one."
            )

        score_values = [
            float(item.score)
            for item in self.scores
        ]

        if any(
            not isfinite(score)
            or score < 0.0
            or score > 1.0
            for score in score_values
        ):
            raise ValueError(
                "Screening scores must be finite and lie in [0, 1]."
            )

        if not isclose(
            sum(score_values),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "Screening scores must sum to one."
            )

        if score_values != sorted(
            score_values,
            reverse=True,
        ):
            raise ValueError(
                "Screening scores must be ordered from highest to lowest."
            )

    @classmethod
    def from_scores(
        cls,
        scores: Mapping[str, float],
        *,
        status: str,
        method: str,
        warnings: Sequence[str] = (),
    ) -> ModelScreeningReport:
        """Normalize and rank scores for all screening models."""
        expected_models = set(SCREENING_MODELS)
        supplied_models = set(scores)

        if supplied_models != expected_models:
            missing = sorted(
                expected_models - supplied_models
            )
            extra = sorted(
                supplied_models - expected_models
            )

            raise ValueError(
                "Scores must be supplied for every screening model. "
                f"Missing: {missing}. Extra: {extra}."
            )

        numeric_scores = {
            model_name: float(score)
            for model_name, score in scores.items()
        }

        if any(
            not isfinite(score)
            or score < 0.0
            for score in numeric_scores.values()
        ):
            raise ValueError(
                "Input screening scores must be finite and non-negative."
            )

        total = sum(
            numeric_scores.values()
        )

        if total <= 0.0:
            raise ValueError(
                "At least one screening score must be positive."
            )

        ranked = sorted(
            numeric_scores.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )

        normalized_scores = tuple(
            ModelScreeningScore(
                model_name=model_name,
                score=score / total,
                rank=rank,
            )
            for rank, (
                model_name,
                score,
            ) in enumerate(
                ranked,
                start=1,
            )
        )

        return cls(
            scores=normalized_scores,
            status=status,
            method=method,
            warnings=tuple(
                str(warning)
                for warning in warnings
            ),
        )

    @property
    def top_model(self) -> str:
        """Return the highest-ranked model without selecting it."""
        return self.scores[0].model_name

    def summary(self) -> str:
        """Return a concise human-readable screening report."""
        lines = [
            "BayMorph Model Screening",
            "",
        ]

        for item in self.scores:
            lines.append(
                f"{item.model_name:22s} {item.score:.6f}"
            )

        lines.extend(
            [
                "",
                f"Status: {self.status}",
                "User decision required: YES",
                "",
                (
                    "Scores are conditional on the declared simulation "
                    "and training distributions."
                ),
                (
                    "They are not real-data posterior model "
                    "probabilities or Bayesian evidences."
                ),
                "No production model was selected.",
            ]
        )

        return "\n".join(
            lines
        )


__all__ = [
    "ModelScreeningReport",
    "ModelScreeningScore",
    "SCREENING_STATUSES",
]
