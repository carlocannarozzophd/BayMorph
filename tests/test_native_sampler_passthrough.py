"""Tests for native sampler keyword pass-through."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

import baymorph.fitting.dynesty as dynesty_module
from baymorph.fitting.dynesty import DynestyFitter
from baymorph.fitting.image_problem import ImageInferenceProblem
from baymorph.fitting.parameters import (
    Parameter,
    ParameterSet,
)
from baymorph.fitting.pocomc import PocoMCFitter
from baymorph.likelihood import GaussianImageLikelihood


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
    ):
        """Render a constant image."""
        return np.full(
            shape,
            self.x,
            dtype=float,
        )


def _make_problem() -> ImageInferenceProblem:
    """Construct a minimal image inference problem."""
    return ImageInferenceProblem(
        model_factory=lambda x: ConstantImageModel(
            x
        ),
        parameters=ParameterSet(
            [
                Parameter(
                    "x",
                    -1.0,
                    1.0,
                )
            ]
        ),
        likelihood=GaussianImageLikelihood(
            data=np.zeros(
                (2, 2),
                dtype=float,
            ),
            noise=np.ones(
                (2, 2),
                dtype=float,
            ),
        ),
    )


def _fake_dynesty_results():
    """Return minimal Dynesty-like results."""
    return SimpleNamespace(
        samples=np.asarray(
            [
                [-0.1],
                [0.2],
            ]
        ),
        logwt=np.log(
            np.asarray(
                [
                    0.4,
                    0.6,
                ]
            )
        ),
        logz=np.asarray(
            [
                0.0,
            ]
        ),
        logl=np.asarray(
            [
                -1.0,
                -0.5,
            ]
        ),
        logzerr=np.asarray(
            [
                0.1,
            ]
        ),
        ncall=np.asarray(
            [
                10,
                20,
            ]
        ),
    )


def test_dynesty_static_forwards_native_kwargs(
    monkeypatch,
):
    captured = {}

    class FakeNestedSampler:
        def __init__(
            self,
            loglikelihood,
            prior_transform,
            ndim,
            **kwargs,
        ):
            captured["loglikelihood"] = loglikelihood
            captured["prior_transform"] = prior_transform
            captured["ndim"] = ndim
            captured["init"] = kwargs
            self.results = _fake_dynesty_results()

        def run_nested(
            self,
            **kwargs,
        ):
            captured["run"] = kwargs

    monkeypatch.setattr(
        dynesty_module.dynesty,
        "NestedSampler",
        FakeNestedSampler,
    )

    result = DynestyFitter(
        problem=_make_problem()
    ).run(
        mode="static",
        init_kwargs={
            "nlive": 321,
            "sample": "rslice",
            "future_option": 17,
        },
        run_kwargs={
            "dlogz": 0.123,
            "future_run_option": "value",
        },
    )

    assert captured["ndim"] == 1
    assert captured["init"] == {
        "nlive": 321,
        "sample": "rslice",
        "future_option": 17,
    }
    assert captured["run"] == {
        "dlogz": 0.123,
        "future_run_option": "value",
    }

    assert result.provenance[
        "sampler"
    ] == {
        "name": "dynesty",
        "mode": "static",
        "init": captured["init"],
        "run": captured["run"],
    }


def test_dynesty_dynamic_uses_dynamic_sampler(
    monkeypatch,
):
    captured = {}

    class FakeDynamicNestedSampler:
        def __init__(
            self,
            loglikelihood,
            prior_transform,
            ndim,
            **kwargs,
        ):
            captured["ndim"] = ndim
            captured["init"] = kwargs
            self.results = _fake_dynesty_results()

        def run_nested(
            self,
            **kwargs,
        ):
            captured["run"] = kwargs

    monkeypatch.setattr(
        dynesty_module.dynesty,
        "DynamicNestedSampler",
        FakeDynamicNestedSampler,
    )

    result = DynestyFitter(
        problem=_make_problem()
    ).run(
        mode="dynamic",
        init_kwargs={
            "sample": "rwalk",
        },
        run_kwargs={
            "nlive_init": 500,
            "nlive_batch": 250,
            "dlogz_init": 0.01,
        },
    )

    assert captured["init"] == {
        "sample": "rwalk",
    }
    assert captured["run"] == {
        "nlive_init": 500,
        "nlive_batch": 250,
        "dlogz_init": 0.01,
    }

    assert result.diagnostics[
        "mode"
    ] == "dynamic"


@pytest.mark.parametrize(
    "reserved",
    [
        "loglikelihood",
        "prior_transform",
        "ndim",
    ],
)
def test_dynesty_rejects_baymorph_owned_constructor_kwargs(
    reserved,
):
    with pytest.raises(
        ValueError,
        match="owned by BayMorph",
    ):
        DynestyFitter(
            problem=_make_problem()
        ).run(
            init_kwargs={
                reserved: object(),
            }
        )


def test_pocomc_forwards_native_kwargs(
    monkeypatch,
):
    captured = {}

    class FakePrior:
        def __init__(
            self,
            distributions,
        ):
            self.distributions = distributions

    class FakeSampler:
        def __init__(
            self,
            prior,
            likelihood,
            n_dim,
            **kwargs,
        ):
            captured["prior"] = prior
            captured["likelihood"] = likelihood
            captured["n_dim"] = n_dim
            captured["init"] = kwargs

            self.results = {
                "calls": np.asarray(
                    [100]
                ),
            }

        def run(
            self,
            **kwargs,
        ):
            captured["run"] = kwargs

        def posterior(self):
            return (
                np.asarray(
                    [
                        [-0.2],
                        [0.3],
                    ]
                ),
                np.asarray(
                    [
                        0.4,
                        0.6,
                    ]
                ),
                np.asarray(
                    [
                        -1.0,
                        -0.5,
                    ]
                ),
                np.asarray(
                    [
                        -0.7,
                        -0.6,
                    ]
                ),
            )

        def evidence(self):
            return (
                -2.5,
                0.15,
            )

    monkeypatch.setitem(
        sys.modules,
        "pocomc",
        SimpleNamespace(
            Prior=FakePrior,
            Sampler=FakeSampler,
        ),
    )

    result = PocoMCFitter(
        problem=_make_problem()
    ).run(
        init_kwargs={
            "n_effective": 700,
            "sample": "tpcn",
            "random_state": 42,
            "future_option": 17,
        },
        run_kwargs={
            "n_total": 8000,
            "n_evidence": 4000,
            "future_run_option": "value",
        },
    )

    assert captured["n_dim"] == 1

    assert captured["init"] == {
        "n_effective": 700,
        "sample": "tpcn",
        "random_state": 42,
        "future_option": 17,
    }

    assert captured["run"] == {
        "n_total": 8000,
        "n_evidence": 4000,
        "future_run_option": "value",
    }

    assert result.provenance[
        "sampler"
    ] == {
        "name": "pocomc",
        "init": captured["init"],
        "run": captured["run"],
    }


@pytest.mark.parametrize(
    "reserved",
    [
        "prior",
        "likelihood",
        "n_dim",
    ],
)
def test_pocomc_rejects_baymorph_owned_constructor_kwargs(
    reserved,
):
    with pytest.raises(
        ValueError,
        match="owned by BayMorph",
    ):
        PocoMCFitter(
            problem=_make_problem()
        ).run(
            init_kwargs={
                reserved: object(),
            }
        )
