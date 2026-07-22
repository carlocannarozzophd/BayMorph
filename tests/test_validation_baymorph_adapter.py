import numpy as np
import pytest

from baymorph.results import RealFitProducts
from baymorph.validation import recovery_fit_from_baymorph


class FakePosterior:
    parameter_names = ["flux", "re"]
    logz = np.array([-15.0, -12.5])
    logzerr = np.array([0.6, 0.3])

    def summary(self, parameter_name):
        summaries = {
            "flux": {
                "median": 995.0,
                "q16": 960.0,
                "q84": 1030.0,
            },
            "re": {
                "median": 8.2,
                "q16": 7.7,
                "q84": 8.8,
            },
        }
        return summaries[parameter_name]


def make_fit_products(posterior=None):
    if posterior is None:
        posterior = FakePosterior()

    image = np.zeros((8, 8))

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


def test_recovery_fit_from_baymorph_uses_posterior_summaries():
    result = recovery_fit_from_baymorph(
        make_fit_products(),
        metadata={"campaign": "smoke-test"},
    )

    assert result.estimates == {
        "flux": 995.0,
        "re": 8.2,
    }
    assert result.intervals == {
        "flux": (960.0, 1030.0),
        "re": (7.7, 8.8),
    }
    assert result.log_evidence == -12.5
    assert result.log_evidence_error == 0.3
    assert result.likelihood_evaluations is None

    assert result.metadata["adapter"] == "baymorph"
    assert result.metadata["model_name"] == "single_sersic"
    assert result.metadata["chi2"] == 120.0
    assert result.metadata["reduced_chi2"] == 1.2
    assert result.metadata["ndof"] == 100
    assert result.metadata["campaign"] == "smoke-test"


def test_recovery_fit_from_baymorph_allows_missing_logzerr():
    posterior = FakePosterior()
    posterior.logzerr = None

    result = recovery_fit_from_baymorph(
        make_fit_products(posterior),
    )

    assert result.log_evidence == -12.5
    assert result.log_evidence_error is None


def test_recovery_fit_from_baymorph_requires_posterior():
    fit_products = make_fit_products()
    fit_products.posterior = None

    with pytest.raises(ValueError, match="posterior results"):
        recovery_fit_from_baymorph(fit_products)


def test_recovery_fit_from_baymorph_requires_parameters():
    posterior = FakePosterior()
    posterior.parameter_names = []

    with pytest.raises(ValueError, match="at least one parameter"):
        recovery_fit_from_baymorph(
            make_fit_products(posterior),
        )


def test_recovery_fit_from_baymorph_requires_evidence():
    posterior = FakePosterior()
    posterior.logz = np.array([])

    with pytest.raises(ValueError, match="evidence values"):
        recovery_fit_from_baymorph(
            make_fit_products(posterior),
        )