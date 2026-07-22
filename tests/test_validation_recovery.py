import pytest

from baymorph.validation import RecoveryResult


def test_recovery_result_exposes_parameter_diagnostics():
    result = RecoveryResult(
        case_id="case-00000",
        truth={"re": 10.0},
        estimates={"re": 11.0},
        intervals={"re": (9.0, 12.0)},
        runtime_seconds=2.5,
        likelihood_evaluations=100,
    )

    recovery = result.parameter("re")

    assert recovery.bias == 1.0
    assert recovery.fractional_error == 0.1
    assert recovery.covered is True


def test_recovery_result_requires_estimate_for_every_truth():
    with pytest.raises(ValueError):
        RecoveryResult(case_id="case-00000", truth={"re": 10.0}, estimates={})


def test_recovery_result_requires_complete_interval():
    result = RecoveryResult(
        case_id="case-00000",
        truth={"re": 10.0},
        estimates={"re": 11.0},
        intervals={"re": (None, 12.0)},
    )

    with pytest.raises(ValueError):
        _ = result.parameter("re").covered
