import numpy as np
import pytest

from baymorph.results import PosteriorResult



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


def test_posterior_result_save_load_round_trip(tmp_path):
    result = PosteriorResult(
        samples=np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
            ]
        ),
        logwt=np.array([-2.0, -1.0]),
        logz=np.array([-3.0, -0.5]),
        parameter_names=["flux", "re"],
        logl=np.array([-10.0, -5.0]),
        logzerr=np.array([0.4, 0.2]),
    )

    path = result.save(
        tmp_path / "posterior.npz"
    )
    loaded = PosteriorResult.load(path)

    assert path.exists()
    assert np.array_equal(loaded.samples, result.samples)
    assert np.array_equal(loaded.logwt, result.logwt)
    assert np.array_equal(loaded.logz, result.logz)
    assert np.array_equal(loaded.logl, result.logl)
    assert np.array_equal(loaded.logzerr, result.logzerr)
    assert loaded.parameter_names == result.parameter_names
    assert loaded.best_parameters == result.best_parameters


def test_posterior_result_save_load_preserves_missing_optional_arrays(
    tmp_path,
):
    result = PosteriorResult(
        samples=np.array(
            [
                [1.0],
                [2.0],
            ]
        ),
        logwt=np.array([-1.0, -1.0]),
        logz=np.array([0.0]),
        parameter_names=["flux"],
    )

    path = result.save(
        tmp_path / "posterior.npz"
    )
    loaded = PosteriorResult.load(path)

    assert loaded.logl is None
    assert loaded.logzerr is None
    assert loaded.parameter_names == ["flux"]
