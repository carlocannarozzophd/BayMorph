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
        mode="static",
        init_kwargs={
            "nlive": 20,
            "rstate": np.random.default_rng(
                123
            ),
        },
        run_kwargs={
            "dlogz": 1.0,
            "print_progress": False,
        },
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
        "sampler"
    ]["mode"] == "static"

    assert result.provenance[
        "sampler"
    ]["init"]["nlive"] == 20


def test_dynesty_result_has_no_nested_sampling_public_state():
    fitter = DynestyFitter(
        problem=_make_problem()
    )

    result = fitter.run(
        mode="static",
        init_kwargs={
            "nlive": 20,
            "rstate": np.random.default_rng(
                123
            ),
        },
        run_kwargs={
            "dlogz": 1.0,
            "print_progress": False,
        },
    )

    assert not hasattr(
        result,
        "logwt",
    )

    assert not hasattr(
        result,
        "logzerr",
    )


def test_dynesty_dynamic_returns_canonical_inference_result():
    """Dynamic nested sampling should use the canonical result contract."""
    fitter = DynestyFitter(
        problem=_make_problem()
    )

    result = fitter.run(
        mode="dynamic",
        init_kwargs={
            "nlive": 20,
            "sample": "rwalk",
            "rstate": np.random.default_rng(
                321
            ),
        },
        run_kwargs={
            "nlive_init": 20,
            "nlive_batch": 20,
            "maxbatch": 1,
            "dlogz_init": 5.0,
            "print_progress": False,
        },
    )

    assert isinstance(
        result,
        InferenceResult,
    )

    assert np.isfinite(
        result.logz
    )

    assert result.logz_error is not None
    assert np.isfinite(
        result.logz_error
    )

    assert result.diagnostics[
        "mode"
    ] == "dynamic"

    assert result.provenance[
        "sampler"
    ]["mode"] == "dynamic"
