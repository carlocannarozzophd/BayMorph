"""Tests for Dynesty canonical inference results."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting import (
    Parameter,
    ParameterSet,
)
from baymorph.fitting.dynesty import DynestyFitter
from baymorph.fitting.image_problem import ImageInferenceProblem
from baymorph.likelihood import GaussianImageLikelihood
from baymorph.results import (
    InferenceResult,
    PosteriorSamples,
)


class ConstantImageModel:
    """Minimal constant-valued image model."""

    def __init__(
        self,
        x: float,
    ) -> None:
        self.x = float(x)

    def render(
        self,
        shape,
    ) -> np.ndarray:
        """Render a constant image."""
        return np.full(
            shape,
            self.x,
            dtype=float,
        )


def _make_problem() -> ImageInferenceProblem:
    """Construct a minimal image inference problem."""
    parameters = ParameterSet(
        [
            Parameter(
                "x",
                -2.0,
                2.0,
            )
        ]
    )

    likelihood = GaussianImageLikelihood(
        data=np.zeros(
            (2, 2),
            dtype=float,
        ),
        noise=np.ones(
            (2, 2),
            dtype=float,
        ),
    )

    return ImageInferenceProblem(
        model_factory=lambda x: ConstantImageModel(
            x
        ),
        parameters=parameters,
        likelihood=likelihood,
    )


def test_dynesty_returns_canonical_inference_result():
    fitter = DynestyFitter(
        problem=_make_problem()
    )

    result = fitter.run(
        nlive=20,
        dlogz=1.0,
        print_progress=False,
        seed=123,
    )

    assert isinstance(
        result,
        InferenceResult,
    )

    assert isinstance(
        result.posterior,
        PosteriorSamples,
    )

    assert result.evidence is not None
    assert result.evidence.method == "dynesty"

    assert (
        result.evidence.uncertainty_kind
        == "nested_sampling_internal"
    )

    assert np.isfinite(
        result.logz
    )

    assert result.logz_error is not None
    assert result.logz_error >= 0.0

    assert result.parameter_names == [
        "x",
    ]

    assert result.samples.ndim == 2

    assert np.sum(
        result.weights
    ) == pytest.approx(
        1.0
    )

    assert result.log_likelihood is not None

    assert result.log_prior is None

    assert result.diagnostics[
        "backend"
    ] == "dynesty"

    assert result.diagnostics[
        "calls"
    ] is not None

    assert result.provenance[
        "backend"
    ] == "dynesty"

    assert result.provenance[
        "seed"
    ] == 123


def test_dynesty_result_has_no_nested_sampling_public_state():
    fitter = DynestyFitter(
        problem=_make_problem()
    )

    result = fitter.run(
        nlive=20,
        dlogz=1.0,
        print_progress=False,
        seed=123,
    )

    assert not hasattr(
        result,
        "logwt",
    )

    assert not hasattr(
        result,
        "logzerr",
    )
