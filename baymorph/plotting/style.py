"""Centralized plotting style for BayMorph."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator


def set_plot_style(
    fontsize: int = 35,
    usetex: bool = False,
) -> None:
    """Set the global Matplotlib style used by BayMorph plots.

    Parameters
    ----------
    fontsize : int, default=35
        Base font size used for labels, titles, ticks, and legends.
    usetex : bool, default=False
        If True, enable LaTeX text rendering through Matplotlib.
    """
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
    """Apply BayMorph styling to a single Matplotlib axis."""
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
    """Apply BayMorph axis styling to one or more axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or iterable of matplotlib.axes.Axes
        Axis, array of axes, or iterable of axes to style.
    fontsize : int, default=35
        Base font size used for ticks, labels, and titles.
    """
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
    """Create a Matplotlib figure using the BayMorph plotting style.

    Parameters
    ----------
    nrows : int, default=1
        Number of subplot rows.
    ncols : int, default=1
        Number of subplot columns.
    figsize : tuple of float, default=(15, 15)
        Figure size in inches.
    fontsize : int, default=35
        Base font size used by the BayMorph plotting style.
    usetex : bool, default=False
        If True, enable LaTeX text rendering through Matplotlib.
    flatten : bool, default=True
        If True and multiple axes are created, return a flattened axes array.
    **kwargs
        Additional keyword arguments passed to :func:`matplotlib.pyplot.subplots`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Created figure.
    ax : matplotlib.axes.Axes or ndarray
        Created axis or axes.
    """
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
    """Apply final layout adjustments to a figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to finalize.

    Returns
    -------
    matplotlib.figure.Figure
        Finalized figure.
    """
    fig.tight_layout()
    return fig