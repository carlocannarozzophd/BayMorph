import numpy as np
import pytest

from baymorph import PosteriorResult



def test_posterior_result_summary():
    samples = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
        ]
    )

    logwt = np.log(
        np.ones(5) / 5
    )

    logz = np.array(
        [0.0]
    )

    result = PosteriorResult(
        samples=samples,
        logwt=logwt,
        logz=logz,
        parameter_names=["flux"],
    )

    summary = result.summary(
        "flux"
    )

    assert np.isclose(
        summary["median"],
        3.0,
    )


def test_posterior_result_dynamic_attributes():
    samples = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
        ]
    )

    logwt = np.log(
        np.ones(5) / 5
    )

    logz = np.array(
        [0.0]
    )

    result = PosteriorResult(
        samples=samples,
        logwt=logwt,
        logz=logz,
        parameter_names=["flux"],
    )

    assert np.isclose(
        result.flux_median,
        3.0,
    )


def test_posterior_result_best_parameters():
    samples = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ]
    )

    logwt = np.log(
        np.ones(3) / 3
    )

    logz = np.array(
        [0.0]
    )

    logl = np.array(
        [
            -10.0,
            -1.0,
            -5.0,
        ]
    )

    result = PosteriorResult(
        samples=samples,
        logwt=logwt,
        logz=logz,
        logl=logl,
        parameter_names=[
            "flux",
            "re",
        ],
    )

    assert result.best_index == 1
    assert np.allclose(
        result.best_sample,
        np.array(
            [
                2.0,
                20.0,
            ]
        ),
    )
    assert result.best_parameters == {
        "flux": 2.0,
        "re": 20.0,
    }


def test_posterior_result_best_parameters_requires_logl():
    samples = np.array(
        [
            [1.0],
            [2.0],
        ]
    )

    logwt = np.log(
        np.ones(2) / 2
    )

    logz = np.array(
        [0.0]
    )

    result = PosteriorResult(
        samples=samples,
        logwt=logwt,
        logz=logz,
        parameter_names=["flux"],
    )

    with pytest.raises(ValueError):
        _ = result.best_parameters