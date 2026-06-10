import matplotlib.pyplot as plt
import numpy as np

from baymorph import plot_1d_posterior, plot_2d_samples


def test_plot_1d_posterior():
    samples = np.linspace(0, 1, 100)

    fig, ax = plot_1d_posterior(
        samples,
        true_value=0.5,
        parameter_name="flux",
    )

    assert fig is not None
    assert ax is not None

    plt.close(fig)


def test_plot_2d_samples():
    samples_x = np.linspace(0, 1, 100)
    samples_y = np.linspace(1, 2, 100)

    fig, ax = plot_2d_samples(
        samples_x,
        samples_y,
        true_x=0.5,
        true_y=1.5,
        x_name="flux",
        y_name="re",
    )

    assert fig is not None
    assert ax is not None

    plt.close(fig)