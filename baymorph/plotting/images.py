"""Image plotting utilities."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from .style import set_plot_style, style_axes

BAYMORPH_IMAGE_CMAP = "turbo"


def _log_limits(
    *images,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.9,
    floor: float = 1e-3,
) -> tuple[float, float]:
    """Return robust positive limits for logarithmic image display."""
    positive_values = []

    for image in images:
        values = np.asarray(
            image,
            dtype=float,
        )

        values = values[
            np.isfinite(values)
            & (values > 0)
        ]

        if values.size > 0:
            positive_values.append(
                values
            )

    if len(positive_values) == 0:
        return floor, 1.0

    positive_values = np.concatenate(
        positive_values,
    )

    vmin = np.percentile(
        positive_values,
        lower_percentile,
    )

    vmax = np.percentile(
        positive_values,
        upper_percentile,
    )

    vmin = max(
        float(vmin),
        floor,
    )

    vmax = max(
        float(vmax),
        vmin * 10,
    )

    return vmin, vmax


def show_log_image(
    image,
    title: str,
    cmap: str = BAYMORPH_IMAGE_CMAP,
    colorbar_label: str = "Flux",
    show: bool = False,
):
    """Display an image using robust logarithmic normalization."""
    vmin, vmax = _log_limits(
        image,
    )

    display_image = np.clip(
        np.asarray(
            image,
            dtype=float,
        ),
        vmin,
        vmax,
    )

    plt.figure(
        figsize=(6, 6),
    )

    plt.imshow(
        display_image,
        origin="lower",
        cmap=cmap,
        norm=LogNorm(
            vmin=vmin,
            vmax=vmax,
        ),
    )

    plt.colorbar(
        label=colorbar_label,
    )

    plt.title(
        f"{title} (log scale)",
    )

    if show:
        plt.show()


def plot_log_data_model_residual(
    data,
    model,
    title: str = "",
    cmap: str = BAYMORPH_IMAGE_CMAP,
    show: bool = False,
):
    """Plot data and model on log scale, with linear residual panel."""
    set_plot_style()

    data = np.asarray(
        data,
        dtype=float,
    )

    model = np.asarray(
        model,
        dtype=float,
    )

    residual = data - model

    vmin, vmax = _log_limits(
        data,
        model,
    )

    data_display = np.clip(
        data,
        vmin,
        vmax,
    )

    model_display = np.clip(
        model,
        vmin,
        vmax,
    )

    residual_limit = np.nanpercentile(
        np.abs(residual),
        99,
    )

    if (
        not np.isfinite(
            residual_limit
        )
        or residual_limit == 0
    ):
        residual_limit = 1.0

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(45,15),
    )

    style_axes(
        axes,
    )

    data_image = axes[0].imshow(
        data_display,
        origin="lower",
        cmap=cmap,
        norm=LogNorm(
            vmin=vmin,
            vmax=vmax,
        ),
    )

    axes[0].set_title(
        "Data (log)"
    )

    fig.colorbar(
        data_image,
        ax=axes[0],
    )

    model_image = axes[1].imshow(
        model_display,
        origin="lower",
        cmap=cmap,
        norm=LogNorm(
            vmin=vmin,
            vmax=vmax,
        ),
    )

    axes[1].set_title(
        "Model (log)"
    )

    fig.colorbar(
        model_image,
        ax=axes[1],
    )

    residual_image = axes[2].imshow(
        residual,
        origin="lower",
        cmap="RdBu_r",
        vmin=-residual_limit,
        vmax=residual_limit,
    )

    axes[2].set_title(
        "Residual"
    )

    fig.colorbar(
        residual_image,
        ax=axes[2],
    )

    if title:
        fig.suptitle(
            title,
            y=1.02,
        )

    plt.tight_layout()

    if show:
        plt.show()
    
    return fig, axes