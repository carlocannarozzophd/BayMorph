import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from baymorph import (
    plot_log_data_model_residual,
    show_log_image,
)


def test_show_log_image_runs():
    image = np.ones(
        (
            10,
            10,
        )
    )

    show_log_image(
        image,
        title="Test",
    )

    plt.close(
        "all"
    )


def test_plot_log_data_model_residual_runs():
    data = np.ones(
        (
            10,
            10,
        )
    )

    model = np.ones(
        (
            10,
            10,
        )
    )

    plot_log_data_model_residual(
        data,
        model,
    )

    plt.close(
        "all"
    )