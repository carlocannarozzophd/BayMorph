"""Tests for the stable BayMorph public API."""

import baymorph


EXPECTED_PUBLIC_API = {
    "PriorGuidance",
    "ObservationInspection",
    "ObservationAdvisor",
    "BayMorphRunner",
    "Observation",
    "ObservationMetadata",
    "ImageData",
    "SersicModel",
    "DoubleSersicModel",
    "CompositeModel",
    "ConvolvedModel",
    "GaussianPSF",
    "EmpiricalPSF",
    "SyntheticObservation",
    "SyntheticSersic",
    "EvidenceEstimate",
    "InferenceResult",
    "PosteriorSamples",
    "RealFitProducts",
    "ModelScreeningClassifierArtifact",
    "ModelScreeningReport",
    "ModelScreeningScore",
    "SingleObservationModelAdvisor",
    "SingleObservationScreeningResult",
    "load_model_screening_classifier",
    "make_config_from_fits",
    "make_observation",
    "make_multi_observation_config",
    "make_multi_parameters",
}


def test_public_api_is_explicit_and_stable():
    assert set(baymorph.__all__) == EXPECTED_PUBLIC_API


def test_public_api_symbols_are_available():
    for name in EXPECTED_PUBLIC_API:
        assert hasattr(
            baymorph,
            name,
        )
