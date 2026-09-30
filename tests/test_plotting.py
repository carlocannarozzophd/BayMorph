import matplotlib.pyplot as plt
import numpy as np

from baymorph.plotting import (
    corner_plot,
    make_plot,
)


def test_make_plot():
    fig, ax = make_plot(
        figsize=(4, 4),
        fontsize=12,
    )

    assert fig is not None
    assert ax is not None

    plt.close(fig)


def test_corner_plot():
    samples = np.vstack(
        [
            np.random.normal(0, 1, 200),
            np.random.normal(1, 1, 200),
        ]
    )

    fig, ax_grid, params = corner_plot(
        samples,
        parameter_labels=["flux", "re"],
        bins=20,
    )

    assert fig is not None
    assert ax_grid.shape == (2, 2)
    assert params.shape == (1, 2, 200)

    plt.close(fig)
