"""Tests for fit-diagnostic masking and radial-profile labels."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from baymorph.analysis.posterior_profiles import (
    PosteriorProfileResult,
    ProfileBand,
)
from baymorph.plotting.images import (
    plot_pipeline_fit_maps,
)
from baymorph.plotting.radial import (
    plot_radial_profile_diagnostics,
)
from baymorph.plotting.style import (
    computer_modern_text,
)


def test_fit_maps_overlay_excluded_pixels_on_residual_panels():
    """Excluded pixels should be gray in both residual diagnostics."""
    shape = (
        5,
        5,
    )

    model = np.full(
        shape,
        10.0,
    )

    prepared = model + 1.0
    prepared[
        2,
        2,
    ] = 1010.0

    mask = np.zeros(
        shape,
        dtype=bool,
    )
    mask[
        2,
        2,
    ] = True

    fig, axes = plot_pipeline_fit_maps(
        original_cutout=prepared,
        prepared_cutout=prepared,
        model=model,
        norm_residual=prepared - model,
        contaminant_mask=mask,
        target_mask=np.zeros(
            shape,
            dtype=bool,
        ),
        psf=np.ones(
            (
                3,
                3,
            )
        ),
    )

    residual_overlay = np.asarray(
        axes[
            1,
            2,
        ].images[
            -1
        ].get_array()
    )

    norm_overlay = np.asarray(
        axes[
            1,
            3,
        ].images[
            -1
        ].get_array()
    )

    expected_gray = np.asarray(
        [
            0.82,
            0.82,
            0.82,
            1.0,
        ]
    )

    assert residual_overlay[
        2,
        2,
    ] == pytest.approx(
        expected_gray
    )

    assert norm_overlay[
        2,
        2,
    ] == pytest.approx(
        expected_gray
    )

    assert residual_overlay[
        0,
        0,
        3,
    ] == pytest.approx(
        0.0
    )

    assert norm_overlay[
        0,
        0,
        3,
    ] == pytest.approx(
        0.0
    )

    plt.close(
        fig
    )


def test_masked_residuals_do_not_set_display_scale():
    """Masked outliers should not determine residual color limits."""
    shape = (
        5,
        5,
    )

    model = np.full(
        shape,
        10.0,
    )

    prepared = model + 1.0
    prepared[
        2,
        2,
    ] = 1010.0

    mask = np.zeros(
        shape,
        dtype=bool,
    )
    mask[
        2,
        2,
    ] = True

    fig, axes = plot_pipeline_fit_maps(
        original_cutout=prepared,
        prepared_cutout=prepared,
        model=model,
        norm_residual=prepared - model,
        contaminant_mask=mask,
        target_mask=np.zeros(
            shape,
            dtype=bool,
        ),
        psf=np.ones(
            (
                3,
                3,
            )
        ),
    )

    residual_clim = axes[
        1,
        2,
    ].images[
        0
    ].get_clim()

    norm_clim = axes[
        1,
        3,
    ].images[
        0
    ].get_clim()

    assert residual_clim == pytest.approx(
        (
            -1.0,
            1.0,
        )
    )

    assert norm_clim == pytest.approx(
        (
            -1.0,
            1.0,
        )
    )

    plt.close(
        fig
    )


def test_fit_maps_draw_the_given_standardized_residual():
    """The normalized panel must draw the fit's standardized residual.

    Under AD-047 the scale of the noise model exceeds the prepared sky scale
    where the source is bright, so the panel must not divide by a noise map.
    """
    shape = (
        5,
        5,
    )

    model = np.full(
        shape,
        10.0,
    )

    prepared = model + 1.0

    norm_residual = np.linspace(
        -2.0,
        2.0,
        25,
    ).reshape(
        shape
    )
    norm_residual[
        0,
        0,
    ] = np.nan

    fig, axes = plot_pipeline_fit_maps(
        original_cutout=prepared,
        prepared_cutout=prepared,
        model=model,
        norm_residual=norm_residual,
        psf=np.ones(
            (
                3,
                3,
            )
        ),
    )

    drawn = np.ma.filled(
        np.ma.asarray(
            axes[
                1,
                3,
            ].images[
                0
            ].get_array(),
            dtype=float,
        ),
        np.nan,
    )

    np.testing.assert_array_equal(
        drawn,
        norm_residual,
    )

    plt.close(
        fig
    )

    with pytest.raises(
        ValueError,
        match="shape of the prepared cutout",
    ):
        plot_pipeline_fit_maps(
            original_cutout=prepared,
            prepared_cutout=prepared,
            model=model,
            norm_residual=np.zeros(
                (
                    4,
                    5,
                )
            ),
        )

    plt.close(
        "all"
    )


def test_radial_legend_describes_pixel_range_and_posterior_interval():
    """Radial diagnostics should distinguish scatter from posterior width."""
    radius = np.asarray(
        [
            1.0,
            2.0,
            3.0,
        ]
    )

    observed = np.asarray(
        [
            10.0,
            5.0,
            2.0,
        ]
    )

    model = np.asarray(
        [
            9.5,
            5.2,
            2.1,
        ]
    )

    posterior = PosteriorProfileResult(
        radius=radius,
        total=ProfileBand(
            p16=np.asarray(
                [
                    9.0,
                    4.8,
                    1.9,
                ]
            ),
            p50=np.asarray(
                [
                    9.5,
                    5.1,
                    2.0,
                ]
            ),
            p84=np.asarray(
                [
                    10.0,
                    5.4,
                    2.2,
                ]
            ),
        ),
        components={},
    )

    fig, axes = plot_radial_profile_diagnostics(
        radius=radius,
        profile_data=observed,
        profile_model=model,
        profile_data_p16=np.asarray(
            [
                8.0,
                4.0,
                1.5,
            ]
        ),
        profile_data_p84=np.asarray(
            [
                12.0,
                6.0,
                2.7,
            ]
        ),
        posterior_profiles=posterior,
    )

    labels = axes[
        0
    ].get_legend_handles_labels()[
        1
    ]

    assert (
        computer_modern_text(
            "Observed 16--84% pixel range"
        )
        in labels
    )

    assert (
        computer_modern_text(
            "68% posterior interval"
        )
        in labels
    )

    plt.close(
        fig
    )
