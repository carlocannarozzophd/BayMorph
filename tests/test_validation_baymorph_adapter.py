import numpy as np
import pytest

from baymorph.results import (
    EvidenceEstimate,
    InferenceResult,
    PosteriorSamples,
    RealFitProducts,
)
from baymorph.validation import recovery_fit_from_baymorph


def make_inference_result(
    *,
    uncertainty: float | None = 0.3,
) -> InferenceResult:
    return InferenceResult(
        posterior=PosteriorSamples(
            samples=np.asarray(
                [
                    [960.0, 7.7],
                    [995.0, 8.2],
                    [1030.0, 8.8],
                ],
                dtype=float,
            ),
            weights=np.asarray(
                [
                    0.25,
                    0.50,
                    0.25,
                ],
                dtype=float,
            ),
            parameter_names=[
                "flux",
                "re",
            ],
            log_likelihood=np.asarray(
                [
                    -2.0,
                    -1.0,
                    -2.0,
                ],
                dtype=float,
            ),
        ),
        evidence=EvidenceEstimate(
            logz=-12.5,
            uncertainty=uncertainty,
            method="test",
        ),
        diagnostics={
            "calls": 123,
        },
        provenance={
            "backend": "test",
        },
    )


def make_fit_products(
    posterior: InferenceResult | None = None,
) -> RealFitProducts:
    if posterior is None:
        posterior = make_inference_result()

    image = np.zeros(
        (
            8,
            8,
        )
    )

    return RealFitProducts(
        posterior=posterior,
        best_parameters={
            "flux": 990.0,
            "re": 8.1,
        },
        best_model_image=image,
        residual=image,
        norm_residual=image,
        chi2=120.0,
        ndof=100,
        reduced_chi2=1.2,
        model_name="single_sersic",
    )


def test_recovery_fit_from_baymorph_uses_canonical_inference_result():
    inference = make_inference_result()

    result = recovery_fit_from_baymorph(
        make_fit_products(
            inference
        ),
        metadata={
            "campaign": "smoke-test",
        },
    )

    flux_summary = inference.summary(
        "flux"
    )
    re_summary = inference.summary(
        "re"
    )

    assert result.estimates == pytest.approx(
        {
            "flux": flux_summary[
                "median"
            ],
            "re": re_summary[
                "median"
            ],
        }
    )
    assert result.intervals[
        "flux"
    ] == pytest.approx(
        (
            flux_summary["q16"],
            flux_summary["q84"],
        )
    )
    assert result.intervals[
        "re"
    ] == pytest.approx(
        (
            re_summary["q16"],
            re_summary["q84"],
        )
    )

    assert result.log_evidence == pytest.approx(
        -12.5
    )
    assert result.log_evidence_error == pytest.approx(
        0.3
    )
    assert result.likelihood_evaluations == 123

    assert result.metadata[
        "adapter"
    ] == "baymorph"
    assert result.metadata[
        "backend"
    ] == "test"
    assert result.metadata[
        "model_name"
    ] == "single_sersic"
    assert result.metadata[
        "chi2"
    ] == pytest.approx(
        120.0
    )
    assert result.metadata[
        "reduced_chi2"
    ] == pytest.approx(
        1.2
    )
    assert result.metadata[
        "ndof"
    ] == 100
    assert result.metadata[
        "campaign"
    ] == "smoke-test"


def test_recovery_fit_from_baymorph_allows_missing_evidence_uncertainty():
    result = recovery_fit_from_baymorph(
        make_fit_products(
            make_inference_result(
                uncertainty=None
            )
        )
    )

    assert result.log_evidence == pytest.approx(
        -12.5
    )
    assert result.log_evidence_error is None


def test_recovery_fit_from_baymorph_requires_posterior():
    fit_products = make_fit_products()
    fit_products.posterior = None

    with pytest.raises(
        ValueError,
        match="posterior results",
    ):
        recovery_fit_from_baymorph(
            fit_products
        )
