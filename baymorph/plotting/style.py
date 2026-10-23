"""Canonical plotting style for BayMorph."""

from __future__ import annotations

import os
import re
import shutil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator


PANEL_SIZE = 15.0
FONT_SIZE = 35
AXES_LINEWIDTH = 3
MAJOR_TICK_LENGTH = 10
MINOR_TICK_LENGTH = 5
TICK_WIDTH = 3

USETEX_ENVIRONMENT_VARIABLE = "BAYMORPH_USETEX"
LATEX_EXECUTABLES = (
    "latex",
    "dvipng",
)


def latex_enabled(
    usetex: bool | None = None,
) -> bool:
    """Return whether BayMorph figures are typeset with LaTeX.

    LaTeX is the canonical typesetting. It is turned off only explicitly,
    with ``usetex=False`` or with the environment variable
    ``BAYMORPH_USETEX=0``.
    """
    if usetex is not None:
        return bool(
            usetex
        )

    return (
        os.environ.get(
            USETEX_ENVIRONMENT_VARIABLE,
            "1",
        )
        != "0"
    )


def require_latex() -> None:
    """Raise if Matplotlib cannot typeset text with LaTeX.

    Raises
    ------
    RuntimeError
        If ``latex`` or ``dvipng`` is not on the ``PATH``.
    """
    missing = [
        name
        for name in LATEX_EXECUTABLES
        if shutil.which(
            name
        )
        is None
    ]

    if missing:
        raise RuntimeError(
            "BayMorph figures are typeset with LaTeX, but "
            + " and ".join(
                missing
            )
            + " is not on the PATH. Install a TeX distribution, or set "
            "the environment variable BAYMORPH_USETEX=0 to draw the "
            "figures with Matplotlib's Computer Modern mathtext instead."
        )


def computer_modern_text(
    value: str,
) -> str:
    """Return text that the active typesetting renders as written.

    Matplotlib's bundled ``cmr10`` font does not contain precomposed
    accented Unicode glyphs. Relevant accents are therefore rendered as
    Computer Modern math accents, which LaTeX and Matplotlib's mathtext
    both typeset. When the active style typesets with LaTeX
    (``text.usetex``), the text outside ``$...$`` is adapted to it:
    ``%``, ``&`` and ``#`` are escaped (otherwise LaTeX drops the text
    after ``%`` and fails on the other two), and ``|``, ``<`` and ``>``
    are set in math mode (in text mode the default LaTeX fonts print
    other glyphs for them).
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

    if plt.rcParams["text.usetex"]:
        segments = re.split(
            r"(?<!\\)(\$[^$]*\$)",
            text,
        )
        text = "".join(
            segment
            if segment.startswith("$")
            else re.sub(
                r"(?<!\\)([|<>])",
                r"$\1$",
                re.sub(
                    r"(?<!\\)([%&#])",
                    r"\\\1",
                    segment,
                ),
            )
            for segment in segments
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

    Text is typeset with LaTeX in Computer Modern; pass free text through
    :func:`computer_modern_text` so that it prints as written.
    ``usetex=False`` or the environment variable
    ``BAYMORPH_USETEX=0`` select Matplotlib's bundled Computer Modern
    mathtext instead, for environments without LaTeX.
    ``fontsize`` is retained temporarily for compatibility with existing
    call sites; the canonical size is always 35.

    Parameters
    ----------
    fontsize : int, optional
        Ignored.
    usetex : bool, optional
        Typeset with LaTeX. By default, True unless ``BAYMORPH_USETEX=0``.

    Raises
    ------
    RuntimeError
        If LaTeX is selected and ``latex`` or ``dvipng`` is not on the
        ``PATH``.
    """
    del fontsize

    tex = latex_enabled(
        usetex
    )

    if tex:
        require_latex()

    plt.rcParams.update(
        {
            "text.usetex": tex,
            "font.family": "serif",
            "font.serif": [
                "Computer Modern Roman" if tex else "cmr10",
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
        Typeset with LaTeX. By default, True unless ``BAYMORPH_USETEX=0``;
        see :func:`set_plot_style`.
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
    del fontsize

    set_plot_style(
        usetex=usetex,
    )

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
