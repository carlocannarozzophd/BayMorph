"""Tests for canonical corner-plot density smoothing."""

from __future__ import annotations

import inspect

import matplotlib.pyplot as plt
import numpy as np

import baymorph.plotting.corner as corner_module
from baymorph.plotting.corner import corner_plot


def test_corner_defaults_are_publication_smooth() -> None:
    """Canonical defaults must expose reproducible density smoothing."""
    parameters = inspect.signature(
        corner_plot
    ).parameters

    assert parameters["bins"].default == 50
    assert parameters["density1d"].default == "kde"
    assert parameters["kde_bw_method"].default == "scott"
    assert parameters["smooth_2d"].default == 3.0


def test_corner_defaults_apply_kde_and_2d_smoothing(
    monkeypatch,
) -> None:
    """Default rendering must use KDE marginals and smoothed 2D bins."""
    captured = {}

    original_histogram = (
        corner_module._hist2d_smoothed
    )

    def capture_histogram(
        *args,
        **kwargs,
    ):
        captured.update(
            kwargs
        )

        return original_histogram(
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        corner_module,
        "_hist2d_smoothed",
        capture_histogram,
    )

    rng = np.random.default_rng(
        42
    )

    samples = rng.multivariate_normal(
        mean=[
            0.0,
            0.0,
        ],
        cov=[
            [
                1.0,
                0.6,
            ],
            [
                0.6,
                1.0,
            ],
        ],
        size=400,
    ).T

    fig, axes, _ = corner_plot(
        samples,
        parameter_labels=[
            "flux",
            "re",
        ],
    )

    assert captured["bins"] == 50
    assert captured["smooth_sigma"] == 3.0

    density_lines = [
        line
        for line in axes[
            0,
            0,
        ].lines
        if len(
            line.get_xdata()
        ) == 1200
    ]

    assert len(density_lines) == 1

    plt.close(
        fig
    )
