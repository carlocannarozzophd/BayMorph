"""Tests for simulation-conditioned model-screening results."""

from __future__ import annotations

import pytest

from baymorph.analysis.model_screening import ModelScreeningReport


def example_scores() -> dict[str, float]:
    """Return the reference ambiguous screening scores."""
    return {
        "single_sersic": 0.47,
        "bulge_disk": 0.31,
        "double_sersic": 0.19,
        "sersic_point_source": 0.03,
    }


def test_report_ranks_all_registered_models() -> None:
    """The report must rank every implemented BayMorph model."""
    report = ModelScreeningReport.from_scores(
        example_scores(),
        status="ambiguous",
        method="test_classifier_v1",
    )

    assert report.top_model == "single_sersic"
    assert report.status == "ambiguous"
    assert report.user_decision_required

    assert [
        item.model_name
        for item in report.scores
    ] == [
        "single_sersic",
        "bulge_disk",
        "double_sersic",
        "sersic_point_source",
    ]

    assert [
        item.score
        for item in report.scores
    ] == pytest.approx(
        [
            0.47,
            0.31,
            0.19,
            0.03,
        ]
    )


def test_report_summary_preserves_scientific_limits() -> None:
    """The summary must not claim Bayesian model probabilities."""
    report = ModelScreeningReport.from_scores(
        example_scores(),
        status="ambiguous",
        method="test_classifier_v1",
    )

    summary = report.summary()

    assert "single_sersic          0.470000" in summary
    assert "Status: ambiguous" in summary
    assert "User decision required: YES" in summary
    assert "not real-data posterior model probabilities" in summary
    assert "No production model was selected." in summary


def test_report_normalizes_positive_classifier_weights() -> None:
    """Unnormalized positive classifier outputs must sum to one."""
    report = ModelScreeningReport.from_scores(
        {
            model_name: score * 100.0
            for model_name, score in example_scores().items()
        },
        status="ambiguous",
        method="test_classifier_v1",
    )

    assert sum(
        item.score
        for item in report.scores
    ) == pytest.approx(
        1.0
    )


def test_report_requires_every_registered_model() -> None:
    """A supported model must never be silently omitted."""
    scores = example_scores()
    del scores["double_sersic"]

    with pytest.raises(
        ValueError,
        match="every registered model",
    ):
        ModelScreeningReport.from_scores(
            scores,
            status="ambiguous",
            method="test_classifier_v1",
        )


@pytest.mark.parametrize(
    "status",
    [
        "recommended",
        "ambiguous",
        "out_of_distribution",
        "not_assessable",
    ],
)
def test_report_accepts_declared_statuses(
    status: str,
) -> None:
    """Every screening outcome must preserve human ownership."""
    report = ModelScreeningReport.from_scores(
        example_scores(),
        status=status,
        method="test_classifier_v1",
    )

    assert report.status == status
    assert report.user_decision_required


def test_report_rejects_automatic_model_selection() -> None:
    """No screening result may bypass the user decision."""
    report = ModelScreeningReport.from_scores(
        example_scores(),
        status="recommended",
        method="test_classifier_v1",
    )

    with pytest.raises(
        ValueError,
        match="must require a user decision",
    ):
        type(report)(
            scores=report.scores,
            status=report.status,
            method=report.method,
            user_decision_required=False,
        )
