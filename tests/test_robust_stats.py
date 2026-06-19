import numpy as np

from baymorph import (
    robust_mad,
    robust_rms,
)


def test_robust_rms_ignores_non_finite_values():
    values = np.array(
        [
            3.0,
            4.0,
            np.nan,
            np.inf,
        ]
    )

    np.testing.assert_allclose(
        robust_rms(values),
        np.sqrt(12.5),
)

def test_robust_rms_returns_nan_for_no_finite_values():
    values = np.array(
        [
            np.nan,
            np.inf,
            -np.inf,
        ]
    )

    assert np.isnan(
        robust_rms(values)
    )


def test_robust_mad_ignores_non_finite_values():
    values = np.array(
        [
            1.0,
            2.0,
            3.0,
            np.nan,
            np.inf,
        ]
    )

    assert robust_mad(values) == 1.0


def test_robust_mad_returns_nan_for_no_finite_values():
    values = np.array(
        [
            np.nan,
            np.inf,
            -np.inf,
        ]
    )

    assert np.isnan(
        robust_mad(values)
    )