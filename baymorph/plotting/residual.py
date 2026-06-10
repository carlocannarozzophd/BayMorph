"""Residual plotting utilities."""

import numpy as np

from .style import (
    finalize_figure,
    make_plot,
)


def plot_data_model_residual(
    data: np.ndarray,
    model: np.ndarray,
    cmap: str = "viridis",
):
    """
    Plot data, model and residual images.
    """

    residual = data - model

    fig, axes = make_plot(
        nrows=1,
        ncols=3,
        figsize=(18, 6),
        fontsize=16,
        flatten=True,
    )

    panels = [
        (data, "Data"),
        (model, "Model"),
        (residual, "Residual"),
    ]

    for ax, (image, title) in zip(
        axes,
        panels,
    ):
        im = ax.imshow(
            image,
            origin="lower",
            cmap=cmap,
        )

        ax.set_title(title)

        fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )

    finalize_figure(fig)

    return fig, axes