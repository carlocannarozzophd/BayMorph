import pytest

from baymorph.validation import (
    GroundTruth,
    RecoveryExperiment,
    RecoveryFit,
)


def make_truth():
    return GroundTruth(
        model_name="single_sersic",
        parameters={
            "flux": 1000.0,
            "re": 8.0,
            "n": 2.0,
        },
        image_shape=(64, 64),
        random_seed=42,
    )


def test_recovery_experiment_runs_simulation_and_fit():
    truth = make_truth()
    received = {}

    def simulator(input_truth):
        received["truth"] = input_truth
        return {"image": "synthetic"}

    def fitter(observation):
        received["observation"] = observation
        return RecoveryFit(
            estimates={
                "flux": 990.0,
                "re": 8.2,
                "n": 2.1,
            },
            intervals={
                "flux": (950.0, 1030.0),
                "re": (7.5, 8.8),
                "n": (1.8, 2.4),
            },
            log_evidence=-120.5,
            log_evidence_error=0.3,
            likelihood_evaluations=1250,
            metadata={"backend": "test"},
        )

    experiment = RecoveryExperiment(
        case_id="case-0001",
        truth=truth,
        simulator=simulator,
        fitter=fitter,
        metadata={"campaign": "smoke-test"},
    )

    result = experiment.run()

    assert received["truth"] is truth
    assert received["observation"] == {"image": "synthetic"}

    assert result.case_id == "case-0001"
    assert result.truth == dict(truth.parameters)
    assert result.estimates["re"] == 8.2
    assert result.intervals["re"] == (7.5, 8.8)
    assert result.log_evidence == -120.5
    assert result.log_evidence_error == 0.3
    assert result.likelihood_evaluations == 1250
    assert result.runtime_seconds >= 0.0
    assert result.metadata["campaign"] == "smoke-test"
    assert result.metadata["backend"] == "test"
    assert result.metadata["model_name"] == "single_sersic"


def test_recovery_experiment_rejects_missing_estimates():
    truth = make_truth()

    experiment = RecoveryExperiment(
        case_id="case-0001",
        truth=truth,
        simulator=lambda input_truth: object(),
        fitter=lambda observation: RecoveryFit(
            estimates={
                "flux": 990.0,
                "re": 8.2,
            }
        ),
    )

    with pytest.raises(ValueError, match="Missing recovered estimates for: n"):
        experiment.run()


def test_recovery_experiment_requires_recovery_fit():
    experiment = RecoveryExperiment(
        case_id="case-0001",
        truth=make_truth(),
        simulator=lambda input_truth: object(),
        fitter=lambda observation: {"re": 8.2},
    )

    with pytest.raises(TypeError, match="RecoveryFit"):
        experiment.run()


def test_recovery_fit_validates_intervals():
    with pytest.raises(ValueError, match="lower bounds"):
        RecoveryFit(
            estimates={"re": 8.2},
            intervals={"re": (9.0, 7.0)},
        )


def test_recovery_fit_is_isolated_from_mutable_inputs():
    estimates = {"re": 8.2}
    metadata = {"backend": "test"}

    fit = RecoveryFit(
        estimates=estimates,
        metadata=metadata,
    )

    estimates["re"] = 10.0
    metadata["backend"] = "changed"

    assert fit.estimates["re"] == 8.2
    assert fit.metadata["backend"] == "test"

    with pytest.raises(TypeError):
        fit.estimates["re"] = 10.0


def test_recovery_experiment_validates_configuration():
    truth = make_truth()

    with pytest.raises(ValueError, match="Case ID"):
        RecoveryExperiment(
            case_id=" ",
            truth=truth,
            simulator=lambda input_truth: object(),
            fitter=lambda observation: RecoveryFit(estimates={"re": 8.0}),
        )

    with pytest.raises(TypeError, match="Simulator"):
        RecoveryExperiment(
            case_id="case-0001",
            truth=truth,
            simulator=None,
            fitter=lambda observation: RecoveryFit(estimates={"re": 8.0}),
        )