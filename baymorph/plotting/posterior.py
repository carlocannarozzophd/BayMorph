"""Posterior plotting utilities."""

import numpy as np

from baymorph.plotting.style import finalize_figure, make_plot


def plot_1d_posterior(
    samples: np.ndarray,
    true_value: float | None = None,
    parameter_name: str = "Parameter",
    bins: int = 30,
):
    """Plot a one-dimensional posterior histogram.

    Parameters
    ----------
    samples : ndarray
        Posterior samples for one parameter.
    true_value : float, optional
        Reference value shown as a vertical dashed line.
    parameter_name : str, default="Parameter"
        Label for the x-axis.
    bins : int, default=30
        Number of histogram bins.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure.
    ax : matplotlib.axes.Axes
        Posterior histogram axis.
    """
    fig, ax = make_plot(
        figsize=(6, 4),
        fontsize=16,
    )

    ax.hist(
        samples,
        bins=bins,
    )

    if true_value is not None:
        ax.axvline(
            true_value,
            linestyle="--",
        )

    ax.set_xlabel(parameter_name)
    ax.set_ylabel("Samples")

    finalize_figure(fig)

    return fig, ax


def plot_2d_samples(
    samples_x: np.ndarray,
    samples_y: np.ndarray,
    true_x: float | None = None,
    true_y: float | None = None,
    x_name: str = "x",
    y_name: str = "y",
):
    """Plot two-dimensional posterior samples.

    Parameters
    ----------
    samples_x : ndarray
        Posterior samples for the x-axis parameter.
    samples_y : ndarray
        Posterior samples for the y-axis parameter.
    true_x : float, optional
        Reference x value shown as a vertical dashed line.
    true_y : float, optional
        Reference y value shown as a horizontal dashed line.
    x_name : str, default="x"
        Label for the x-axis.
    y_name : str, default="y"
        Label for the y-axis.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure.
    ax : matplotlib.axes.Axes
        Scatter-plot axis.
    """
    fig, ax = make_plot(
        figsize=(6, 5),
        fontsize=16,
    )

    ax.scatter(
        samples_x,
        samples_y,
        s=5,
        alpha=0.4,
    )

    if true_x is not None:
        ax.axvline(
            true_x,
            linestyle="--",
        )

    if true_y is not None:
        ax.axhline(
            true_y,
            linestyle="--",
        )

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)

    finalize_figure(fig)

    return fig, ax