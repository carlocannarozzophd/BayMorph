"""Tests for canonical Computer Modern plotting typography."""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

from baymorph.plotting import corner_plot
from baymorph.plotting.images import plot_log_image
from baymorph.plotting.style import (
    computer_modern_text,
    set_plot_style,
)


DISPLAY_TITLE = "Simulated single-Sérsic observation"
RENDERED_TITLE = (
    r"Simulated single-S$\acute{\mathrm{e}}$rsic observation"
)


def test_plot_style_uses_bundled_computer_modern() -> None:
    """All canonical text and mathematical symbols must use CM fonts."""
    set_plot_style()

    assert plt.rcParams["font.family"] == [
        "serif",
    ]
    assert plt.rcParams["font.serif"][0] == "cmr10"
    assert plt.rcParams["mathtext.fontset"] == "cm"
    assert plt.rcParams["axes.formatter.use_mathtext"]

    resolved_font = Path(
        font_manager.findfont(
            FontProperties(
                family="serif",
            )
        )
    ).name

    assert resolved_font == "cmr10.ttf"


def test_computer_modern_text_preserves_accent_visually() -> None:
    """Unsupported Unicode accents must become roman CM MathText."""
    assert (
        computer_modern_text(
            DISPLAY_TITLE
        )
        == RENDERED_TITLE
    )


def test_log_image_renders_accent_without_font_warning() -> None:
    """Natural Unicode titles must render without missing CM glyphs."""
    with warnings.catch_warnings():
        warnings.simplefilter(
            "error",
            UserWarning,
        )

        fig, ax = plot_log_image(
            np.ones(
                (
                    5,
                    5,
                )
            ),
            title=DISPLAY_TITLE,
        )

        fig.canvas.draw()

    assert ax.get_title() == RENDERED_TITLE

    plt.close(
        fig
    )


def test_corner_renders_accent_without_font_warning() -> None:
    """Corner titles must use the same Computer Modern conversion."""
    rng = np.random.default_rng(
        42
    )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "error",
            UserWarning,
        )

        fig, _, _ = corner_plot(
            rng.normal(
                size=(
                    3,
                    200,
                )
            ),
            parameter_labels=[
                r"$F$",
                r"$R_e$",
                r"$n$",
            ],
            title=DISPLAY_TITLE,
        )

        fig.canvas.draw()

    rendered_titles = [
        text.get_text()
        for text in fig.texts
    ]

    assert RENDERED_TITLE in rendered_titles

    plt.close(
        fig
    )
