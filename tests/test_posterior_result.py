import numpy as np

from baymorph import (
    PosteriorResult,
)


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

    logz = np.array([0.0])

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

    logz = np.array([0.0])

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