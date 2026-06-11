import numpy as np

from baymorph.stats import (
    posterior_summary,
)


def test_posterior_summary():

    values = np.array(
        [1, 2, 3, 4, 5]
    )

    weights = np.ones(5) / 5

    summary = posterior_summary(
        values,
        weights,
    )

    assert np.isclose(
        summary["median"],
        3.0,
    )