import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from baymorph.plotting import (
    plot_log_data_model_residual,
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
