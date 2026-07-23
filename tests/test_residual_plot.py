import matplotlib.pyplot as plt
import numpy as np
import pytest

from baymorph.plotting import plot_data_model_residual



def test_residual_plot():
    data = np.ones((20, 20))
    model = np.ones((20, 20))

    fig, axes = plot_data_model_residual(
        data,
        model,
    )

    assert fig is not None
    assert len(axes) == 3

    plt.close(fig)


def test_residual_plot_rejects_shape_mismatch():
    data = np.ones((20, 20))
    model = np.ones((10, 10))

    with pytest.raises(ValueError):
        plot_data_model_residual(
            data,
            model,
        )