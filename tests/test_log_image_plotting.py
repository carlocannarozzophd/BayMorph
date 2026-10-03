import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

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


def test_plot_log_data_model_residual_draws_given_norm_residual():
    """A standardized residual from fit products is drawn as given."""
    data = np.full(
        (
            6,
            6,
        ),
        2.0,
    )

    model = np.ones(
        (
            6,
            6,
        )
    )

    norm_residual = np.linspace(
        -3.0,
        3.0,
        36,
    ).reshape(
        (
            6,
            6,
        )
    )

    _, axes = plot_log_data_model_residual(
        data,
        model,
        norm_residual=norm_residual,
    )

    np.testing.assert_array_equal(
        np.ma.filled(
            np.ma.asarray(
                axes[3].images[0].get_array(),
                dtype=float,
            ),
            np.nan,
        ),
        norm_residual,
    )

    plt.close(
        "all"
    )


def test_plot_log_data_model_residual_rejects_ambiguous_standardization():
    """Noise and a standardized residual cannot both define the panel."""
    image = np.ones(
        (
            4,
            4,
        )
    )

    with pytest.raises(
        ValueError,
        match="either noise or norm_residual",
    ):
        plot_log_data_model_residual(
            image,
            image,
            noise=image,
            norm_residual=image,
        )

    with pytest.raises(
        ValueError,
        match="shape of the data",
    ):
        plot_log_data_model_residual(
            image,
            image,
            norm_residual=np.ones(
                (
                    3,
                    4,
                )
            ),
        )

    plt.close(
        "all"
    )
