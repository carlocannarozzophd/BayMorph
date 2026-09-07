"""Tests for diagnostic titles, labels, and posterior weighting."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from baymorph.models.labels import latex_parameter_label
from baymorph.pipeline.corner import _posterior_samples
from baymorph.pipeline.real_fit import (
    _fit_diagnostic_title,
    _object_label,
)
from baymorph.plotting.corner import corner_plot
from baymorph.plotting.radial import (
    plot_radial_profile_diagnostics,
)


def test_object_label_is_human_readable():
    """Object identifiers should become readable diagnostic labels."""
    config = {
        "object": {
            "name": "example_galaxy_g",
        }
    }

    assert _object_label(
        config
    ) == "example galaxy g"

    assert (
        _fit_diagnostic_title(
            config,
            "single_sersic",
        )
        == "example galaxy g | Single Sérsic"
    )


def test_corner_pipeline_prefers_equal_weight_samples():
    """Pipeline corner plots must use the equal-weight posterior."""
    raw = np.asarray(
        [
            [1.0],
            [2.0],
            [100.0],
        ]
    )

    equal_weight = np.asarray(
        [
            [1.0],
            [1.0],
            [2.0],
        ]
    )

    result = SimpleNamespace(
        samples=raw,
        posterior=SimpleNamespace(
            equal_weight_samples=equal_weight,
        ),
    )

    assert np.array_equal(
        _posterior_samples(
            result
        ),
        equal_weight,
    )


def test_corner_displays_object_title():
    """Standalone corner plots should use the upper-right white region."""
    rng = np.random.default_rng(
        42
    )

    fig, ax_grid, _ = corner_plot(
        rng.normal(
            size=(
                3,
                500,
            )
        ),
        parameter_labels=[
            "x",
            "y",
            "z",
        ],
        title="example galaxy g",
    )

    titles = [
        text
        for text in fig.texts
        if text.get_text() == "example galaxy g"
    ]

    assert len(
        titles
    ) == 1

    upper_triangle_centres = []

    for i in range(3):
        for j in range(
            i + 1,
            3,
        ):
            bbox = ax_grid[
                i,
                j,
            ].get_position()

            upper_triangle_centres.append(
                (
                    0.5 * (
                        bbox.x0
                        + bbox.x1
                    ),
                    0.5 * (
                        bbox.y0
                        + bbox.y1
                    ),
                )
            )

    expected_position = np.mean(
        np.asarray(
            upper_triangle_centres
        ),
        axis=0,
    )

    np.testing.assert_allclose(
        titles[
            0
        ].get_position(),
        expected_position,
    )

    plt.close(
        fig
    )


def test_background_level_uses_compact_latex_label():
    """Background-level parameters should be displayed as B."""
    assert (
        latex_parameter_label(
            "background_level"
        )
        == r"$B$"
    )


def test_radial_plot_uses_latex_labels_and_object_title():
    """Radial diagnostics should use publication-style labels."""
    radius = np.asarray(
        [
            1.0,
            2.0,
            3.0,
        ]
    )

    fig, axes = plot_radial_profile_diagnostics(
        radius=radius,
        profile_data=np.asarray(
            [
                10.0,
                5.0,
                2.0,
            ]
        ),
        profile_model=np.asarray(
            [
                9.5,
                5.2,
                2.1,
            ]
        ),
        title="example galaxy g",
    )

    assert (
        axes[
            1
        ].get_xlabel()
        == r"$R_{\rm ell}\,[\mathrm{pix}]$"
    )

    assert (
        axes[
            0
        ].get_ylabel()
        == r"$\log_{10} I$"
    )

    assert (
        axes[
            1
        ].get_ylabel()
        == r"$\Delta \log_{10} I$"
    )

    assert fig._suptitle is not None
    assert (
        fig._suptitle.get_text()
        == "example galaxy g"
    )

    plt.close(
        fig
    )


def test_corner_coordinate_axes_disable_additive_offsets():
    """Pixel-centre coordinates should use direct numerical tick labels."""
    rng = np.random.default_rng(
        123
    )

    samples = np.vstack(
        (
            rng.normal(
                10.0,
                0.1,
                1000,
            ),
            rng.normal(
                378.25,
                0.02,
                1000,
            ),
            rng.normal(
                381.68,
                0.02,
                1000,
            ),
        )
    )

    fig, ax_grid, _ = corner_plot(
        samples,
        parameter_labels=[
            "flux",
            "x0",
            "y0",
        ],
    )

    x0_formatter = (
        ax_grid[
            2,
            1,
        ]
        .xaxis
        .get_major_formatter()
    )

    y0_formatter = (
        ax_grid[
            2,
            0,
        ]
        .yaxis
        .get_major_formatter()
    )

    assert (
        x0_formatter.get_useOffset()
        is False
    )

    assert (
        y0_formatter.get_useOffset()
        is False
    )

    plt.close(
        fig
    )
