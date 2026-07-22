"""Residual plotting utilities."""

import numpy as np

from baymorph.plotting.style import finalize_figure, make_plot


def plot_data_model_residual(
    data: np.ndarray,
    model: np.ndarray,
    image_cmap: str = "viridis",
    residual_cmap: str = "RdBu_r",
):
    """Plot data, model, and residual images with consistent scaling.

    Parameters
    ----------
    data : ndarray
        Observed image.
    model : ndarray
        Model image evaluated on the same pixel grid as ``data``.
    image_cmap : str, default="viridis"
        Matplotlib colormap used for the data and model panels.
    residual_cmap : str, default="RdBu_r"
        Matplotlib colormap used for the residual panel.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure.
    axes : ndarray
        Array of Matplotlib axes for the data, model, and residual panels.

    Raises
    ------
    ValueError
        If ``data`` and ``model`` do not have the same shape.
    """
    data = np.asarray(data, dtype=float)
    model = np.asarray(model, dtype=float)

    if data.shape != model.shape:
        raise ValueError("Data and model must have the same shape.")

    residual = data - model

    image_vmin = min(
        float(np.nanmin(data)),
        float(np.nanmin(model)),
    )
    image_vmax = max(
        float(np.nanmax(data)),
        float(np.nanmax(model)),
    )

    residual_vmax = float(np.nanmax(np.abs(residual)))

    if residual_vmax == 0.0:
        residual_vmax = 1.0

    fig, axes = make_plot(
        nrows=1,
        ncols=3,
        figsize=(18, 6),
        fontsize=16,
        flatten=True,
    )

    panels = [
        (
            data,
            "Data",
            image_cmap,
            image_vmin,
            image_vmax,
        ),
        (
            model,
            "Model",
            image_cmap,
            image_vmin,
            image_vmax,
        ),
        (
            residual,
            "Residual",
            residual_cmap,
            -residual_vmax,
            residual_vmax,
        ),
    ]

    for ax, (image, title, cmap, vmin, vmax) in zip(
        axes,
        panels,
    ):
        im = ax.imshow(
            image,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
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