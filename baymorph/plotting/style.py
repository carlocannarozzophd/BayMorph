"""Centralized plotting style for BayMorph."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator


def set_plot_style(
    fontsize: int = 35,
    usetex: bool = False,
) -> None:
    """Set a robust global matplotlib style."""
    plt.rc("text", usetex=usetex)
    plt.rc("font", family="serif", size=fontsize)
    plt.rc("axes", linewidth=3)
    plt.rc("mathtext", fontset="cm")
    plt.rc("xtick", labelsize=fontsize * 0.8)
    plt.rc("ytick", labelsize=fontsize * 0.8)
    plt.rc("legend", fontsize=fontsize)
    plt.rc("axes", titlesize=fontsize, labelsize=fontsize)


def _style_single_axis(
    ax,
    fontsize: int = 35,
) -> None:
    ax.tick_params(
        which="major",
        length=10,
        width=3,
        direction="out",
        labelsize=fontsize * 0.8,
    )
    ax.tick_params(
        which="minor",
        length=5,
        width=3,
        direction="out",
        labelsize=fontsize * 0.6,
    )
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.title.set_fontsize(fontsize)
    ax.xaxis.label.set_fontsize(fontsize)
    ax.yaxis.label.set_fontsize(fontsize)


def style_axes(
    ax,
    fontsize: int = 35,
) -> None:
    """Apply BayMorph axis styling."""
    if isinstance(ax, np.ndarray):
        for single_ax in ax.flatten():
            _style_single_axis(
                single_ax,
                fontsize=fontsize,
            )
    elif isinstance(ax, (list, tuple)):
        for single_ax in ax:
            _style_single_axis(
                single_ax,
                fontsize=fontsize,
            )
    else:
        _style_single_axis(
            ax,
            fontsize=fontsize,
        )


def make_plot(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] = (15, 15),
    fontsize: int = 35,
    usetex: bool = False,
    flatten: bool = True,
    **kwargs,
):
    """Create a matplotlib figure using the BayMorph style."""
    set_plot_style(
        fontsize=fontsize,
        usetex=usetex,
    )

    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        **kwargs,
    )

    if isinstance(ax, np.ndarray):
        if flatten:
            ax = ax.flatten()

        style_axes(
            ax,
            fontsize=fontsize,
        )
    else:
        style_axes(
            ax,
            fontsize=fontsize,
        )

    return fig, ax


def finalize_figure(fig):
    """Apply final layout adjustments."""
    fig.tight_layout()
    return fig