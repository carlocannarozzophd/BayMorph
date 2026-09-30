"""Canonical plotting style for BayMorph."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator


PANEL_SIZE = 15.0
FONT_SIZE = 35
AXES_LINEWIDTH = 3
MAJOR_TICK_LENGTH = 10
MINOR_TICK_LENGTH = 5
TICK_WIDTH = 3


def computer_modern_text(
    value: str,
) -> str:
    """Return text compatible with bundled Computer Modern fonts.

    Matplotlib's bundled ``cmr10`` font does not contain precomposed
    accented Unicode glyphs. Relevant accents are therefore rendered
    through Computer Modern MathText without requiring external LaTeX.
    """
    text = str(
        value
    )

    replacements = {
        "é": r"$\acute{\mathrm{e}}$",
        "É": r"$\acute{\mathrm{E}}$",
    }

    for character, replacement in replacements.items():
        text = text.replace(
            character,
            replacement,
        )

    return text


def canonical_figsize(
    nrows: int = 1,
    ncols: int = 1,
) -> tuple[float, float]:
    """Return the canonical BayMorph figure size.

    Every subplot occupies one nominal 15-by-15 inch panel.

    Parameters
    ----------
    nrows : int, default=1
        Number of subplot rows.
    ncols : int, default=1
        Number of subplot columns.

    Returns
    -------
    tuple of float
        Figure size as ``(15 * ncols, 15 * nrows)``.

    Raises
    ------
    ValueError
        If either grid dimension is not strictly positive.
    """
    if nrows <= 0 or ncols <= 0:
        raise ValueError(
            "nrows and ncols must be positive."
        )

    return (
        PANEL_SIZE * ncols,
        PANEL_SIZE * nrows,
    )


def set_plot_style(
    fontsize: int | None = None,
    usetex: bool | None = None,
) -> None:
    """Apply the canonical BayMorph Matplotlib style.

    ``fontsize`` and ``usetex`` are retained temporarily for compatibility
    with existing call sites. The canonical style uses Matplotlib mathtext
    rather than an external LaTeX installation so that plotting behaviour
    remains portable and reproducible across supported environments.
    """
    del fontsize, usetex

    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": [
                "cmr10",
            ],
            "font.size": FONT_SIZE,
            "mathtext.fontset": "cm",
            "axes.linewidth": AXES_LINEWIDTH,
            "axes.formatter.use_mathtext": True,
        }
    )


def _style_single_axis(
    ax,
) -> None:
    """Apply the canonical BayMorph style to one Matplotlib axis."""
    ax.tick_params(
        which="major",
        length=MAJOR_TICK_LENGTH,
        width=TICK_WIDTH,
        direction="out",
    )
    ax.tick_params(
        which="minor",
        length=MINOR_TICK_LENGTH,
        width=TICK_WIDTH,
        direction="out",
    )

    ax.xaxis.set_minor_locator(
        AutoMinorLocator()
    )
    ax.yaxis.set_minor_locator(
        AutoMinorLocator()
    )


def style_axes(
    ax,
    fontsize: int | None = None,
) -> None:
    """Apply the canonical BayMorph style to one or more axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or iterable of matplotlib.axes.Axes
        Axis, array of axes, or iterable of axes to style.
    fontsize : int, optional
        Retained temporarily for compatibility. The canonical font size is
        always 35.
    """
    del fontsize

    if isinstance(
        ax,
        np.ndarray,
    ):
        axes = ax.flatten()
    elif isinstance(
        ax,
        (list, tuple),
    ):
        axes = ax
    else:
        axes = [ax]

    for single_ax in axes:
        _style_single_axis(
            single_ax
        )


def style_colorbar(
    colorbar,
) -> None:
    """Match colorbar ticks and outline to the canonical axis style."""
    colorbar.ax.tick_params(
        which="major",
        length=MAJOR_TICK_LENGTH,
        width=TICK_WIDTH,
        direction="out",
    )
    colorbar.ax.tick_params(
        which="minor",
        length=MINOR_TICK_LENGTH,
        width=TICK_WIDTH,
        direction="out",
    )
    colorbar.outline.set_linewidth(
        AXES_LINEWIDTH
    )


def make_plot(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    fontsize: int | None = None,
    usetex: bool | None = None,
    flatten: bool = True,
    **kwargs,
):
    """Create a figure using the canonical BayMorph plotting contract.

    By default, each subplot occupies one 15-by-15 inch panel. Therefore
    a grid with ``nrows`` rows and ``ncols`` columns has figure size
    ``(15 * ncols, 15 * nrows)``.

    Parameters
    ----------
    nrows : int, default=1
        Number of subplot rows.
    ncols : int, default=1
        Number of subplot columns.
    figsize : tuple of float, optional
        Explicit figure size. Retained temporarily while existing plotting
        call sites are migrated to the canonical automatic geometry.
    fontsize : int, optional
        Retained temporarily for compatibility. Ignored.
    usetex : bool, optional
        Retained temporarily for compatibility. Ignored.
    flatten : bool, default=True
        If True, flatten arrays of axes before returning them.
    **kwargs
        Additional keyword arguments passed to
        :func:`matplotlib.pyplot.subplots`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Created figure.
    ax : matplotlib.axes.Axes or numpy.ndarray
        Created axis or axes.
    """
    del fontsize, usetex

    set_plot_style()

    if figsize is None:
        figsize = canonical_figsize(
            nrows=nrows,
            ncols=ncols,
        )

    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        **kwargs,
    )

    style_axes(
        ax
    )

    if (
        flatten
        and isinstance(
            ax,
            np.ndarray,
        )
    ):
        ax = ax.flatten()

    return fig, ax
