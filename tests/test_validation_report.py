import pytest

from baymorph.validation import RecoveryResult, summarize_recovery


def test_summarize_recovery_reports_bias_scatter_and_coverage():
    results = [
        RecoveryResult(
            case_id="case-00000",
            truth={"re": 10.0},
            estimates={"re": 9.0},
            intervals={"re": (8.0, 11.0)},
        ),
        RecoveryResult(
            case_id="case-00001",
            truth={"re": 10.0},
            estimates={"re": 13.0},
            intervals={"re": (11.0, 14.0)},
        ),
    ]

    summary = summarize_recovery(results)

    assert summary["re"]["count"] == 2
    assert summary["re"]["bias"] == 1.0
    assert summary["re"]["scatter"] == 2.0
    assert summary["re"]["coverage"] == 0.5


def test_summarize_recovery_rejects_empty_input():
    with pytest.raises(ValueError):
        summarize_recovery([])
