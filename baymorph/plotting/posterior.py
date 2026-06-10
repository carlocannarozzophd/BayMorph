"""Posterior plotting utilities."""

import numpy as np

from .style import finalize_figure, make_plot


def plot_1d_posterior(
    samples: np.ndarray,
    true_value: float | None = None,
    parameter_name: str = "Parameter",
    bins: int = 30,
):
    """Plot a one-dimensional posterior histogram."""
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
    """Plot two-dimensional posterior samples."""
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