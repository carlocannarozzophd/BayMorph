"""Tests for persisted multi-observation fit products."""

from __future__ import annotations

import json

import numpy as np

from baymorph.pipeline.multi_observation_persistence import (
    save_multi_observation_fit_products,
)
from baymorph.pipeline.multi_observation_outputs import (
    multi_observation_paths,
)
from baymorph.results import (
    InferenceResult,
    MultiObservationFitProducts,
    ObservationFitProducts,
    PosteriorSamples,
)


def _inference_result() -> InferenceResult:
    samples = PosteriorSamples(
        samples=np.asarray(
            [
                [1.0, 2.0],
                [1.5, 2.5],
            ]
        ),
        weights=np.asarray(
            [
                0.5,
                0.5,
            ]
        ),
        parameter_names=[
            "shared_parameter",
            "flux__observation_a",
        ],
        log_likelihood=np.asarray(
            [
                -2.0,
                -1.0,
            ]
        ),
    )

    return InferenceResult(
        posterior=samples,
        diagnostics={},
        provenance={
            "backend": "test",
        },
    )


def _products() -> MultiObservationFitProducts:
    inference_result = _inference_result()

    observation = ObservationFitProducts(
        parameters={
            "flux": 2.5,
            "x0": 4.0,
        },
        best_model_image=np.ones(
            (5, 7),
            dtype=float,
        ),
        residual=np.full(
            (5, 7),
            2.0,
            dtype=float,
        ),
        norm_residual=np.full(
            (5, 7),
            0.5,
            dtype=float,
        ),
        chi2=8.75,
        n_valid_pixels=35,
        noise=np.full(
            (5, 7),
            4.0,
            dtype=float,
        ),
    )

    return MultiObservationFitProducts(
        model_name="single_sersic",
        posterior=inference_result,
        best_parameters=dict(
            inference_result.best_parameters
        ),
        observations={
            "observation_a": observation,
        },
        chi2=8.75,
        ndof=33,
        reduced_chi2=8.75 / 33.0,
    )


def _config(
    tmp_path,
) -> dict:
    return {
        "observations": {
            "observation_a": {
                "config_path": "observation_a.yaml",
            },
        },
        "outputs": {
            "root": str(
                tmp_path
            ),
            "analysis_name": "joint_case",
        },
        "fit": {
            "models": [
                "single_sersic",
            ],
        },
    }


def test_save_multi_observation_fit_products_persists_joint_result(
    tmp_path,
):
    config = _config(
        tmp_path
    )
    products = _products()

    save_multi_observation_fit_products(
        config,
        products,
    )

    paths = multi_observation_paths(
        config
    )["fits"]["single_sersic"]

    assert paths[
        "posterior"
    ].exists()

    loaded = InferenceResult.load(
        paths["posterior"]
    )

    assert loaded.parameter_names == (
        products.posterior.parameter_names
    )

    with paths[
        "best_parameters"
    ].open() as stream:
        best_parameters = json.load(
            stream
        )

    assert best_parameters == (
        products.best_parameters
    )


def test_save_multi_observation_fit_products_persists_observation_arrays(
    tmp_path,
):
    config = _config(
        tmp_path
    )
    products = _products()

    save_multi_observation_fit_products(
        config,
        products,
    )

    paths = multi_observation_paths(
        config
    )["fits"]["single_sersic"][
        "observations"
    ]["observation_a"]

    np.testing.assert_allclose(
        np.load(
            paths["best_model_image"]
        ),
        products.observations[
            "observation_a"
        ].best_model_image,
    )

    np.testing.assert_allclose(
        np.load(
            paths["residual"]
        ),
        products.observations[
            "observation_a"
        ].residual,
    )

    np.testing.assert_allclose(
        np.load(
            paths["norm_residual"]
        ),
        products.observations[
            "observation_a"
        ].norm_residual,
    )


def test_save_multi_observation_fit_products_persists_statistics(
    tmp_path,
):
    config = _config(
        tmp_path
    )
    products = _products()

    save_multi_observation_fit_products(
        config,
        products,
    )

    paths = multi_observation_paths(
        config
    )["fits"]["single_sersic"]

    with paths[
        "fit_summary"
    ].open() as stream:
        joint_summary = json.load(
            stream
        )

    assert joint_summary[
        "chi2"
    ] == products.chi2
    assert joint_summary[
        "ndof"
    ] == products.ndof

    observation_path = paths[
        "observations"
    ]["observation_a"][
        "summary"
    ]

    with observation_path.open() as stream:
        observation_summary = json.load(
            stream
        )

    assert observation_summary[
        "chi2"
    ] == products.observations[
        "observation_a"
    ].chi2

    assert observation_summary[
        "n_valid_pixels"
    ] == 35

    assert observation_summary[
        "parameters"
    ] == {
        "flux": 2.5,
        "x0": 4.0,
    }
