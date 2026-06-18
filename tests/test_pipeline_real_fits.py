import numpy as np
import pytest

from baymorph import robust_rms, robust_mad


def test_robust_mad_uses_median_absolute_deviation():
    values = np.array([1.0, 2.0, 3.0, 100.0])

    assert robust_mad(values) == pytest.approx(1.0)


def test_robust_rms_ignores_nan_values():
    values = np.array([1.0, 2.0, np.nan, 3.0])

    assert robust_rms(values) == pytest.approx(np.sqrt(14.0 / 3.0))