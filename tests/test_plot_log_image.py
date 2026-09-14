"""Tests for canonical single-image plotting."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from baymorph.plotting.images import plot_log_image
from baymorph.plotting.style import (
    MAJOR_TICK_LENGTH,
    TICK_WIDTH,
)
from baymorph.plotting.style import computer_modern_text


def test_plot_log_image_aligns_image_and_colorbar() -> None:
    """The colorbar must match the image-axis height and tick style."""
    image = np.arange(
        1.0,
        101.0,
    ).reshape(
        10,
        10,
    )

    fig, ax = plot_log_image(
        image,
        title="Simulated single-Sérsic observation",
        colorbar_label="Counts",
    )

    try:
        fig.canvas.draw()

        assert ax.get_xlabel() == r"$X$ [pix]"
        assert ax.get_ylabel() == r"$Y$ [pix]"
        assert (
            ax.get_title()
            == computer_modern_text(
                    "Simulated single-Sérsic observation"
                )
        )

        assert len(
            fig.axes
        ) == 2

        colorbar_ax = fig.axes[1]

        image_bounds = ax.get_position()
        colorbar_bounds = colorbar_ax.get_position()

        assert colorbar_bounds.y0 == pytest.approx(
            image_bounds.y0
        )
        assert colorbar_bounds.y1 == pytest.approx(
            image_bounds.y1
        )
        assert colorbar_bounds.height == pytest.approx(
            image_bounds.height
        )

        major_tick = (
            colorbar_ax.yaxis.majorTicks[0]
        )

        assert (
            major_tick.tick1line.get_markersize()
            == pytest.approx(
                MAJOR_TICK_LENGTH
            )
        )
        assert (
            major_tick.tick1line.get_markeredgewidth()
            == pytest.approx(
                TICK_WIDTH
            )
        )

        assert (
            colorbar_ax.get_yticklabels()[0].get_fontsize()
            == pytest.approx(
                ax.get_xticklabels()[0].get_fontsize()
            )
        )
    finally:
        plt.close(
            fig
        )
