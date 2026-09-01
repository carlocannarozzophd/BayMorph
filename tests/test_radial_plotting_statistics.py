"""Tests for observed radial-profile summary plotting."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from baymorph.plotting.radial import (
    plot_radial_profile_diagnostics,
)


def test_radial_plot_accepts_observed_percentiles_and_mean():
    radius = np.asarray(
        [
            1.0,
            2.0,
            3.0,
        ]
    )

    median = np.asarray(
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

    p16 = np.asarray(
        [
            8.0,
            4.0,
            1.5,
        ]
    )

    p84 = np.asarray(
        [
            12.0,
            6.0,
            2.7,
        ]
    )

    mean = np.asarray(
        [
            10.5,
            5.1,
            2.2,
        ]
    )

    fig, axes = plot_radial_profile_diagnostics(
        radius=radius,
        profile_data=median,
        profile_model=model,
        profile_data_p16=p16,
        profile_data_p84=p84,
        profile_data_mean=mean,
    )

    assert len(
        axes
    ) == 2

    assert (
        axes[
            1
        ].get_xlabel()
        == r"$R_{\rm ell}\,[\mathrm{pix}]$"
    )

    plt.close(
        fig
    )
