"""Radial-profile plotting utilities."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .style import style_axes


def _safe_log10(
    values,
):
    """Return base-10 logarithms with invalid values masked as NaN."""
    values = np.asarray(
        values,
        dtype=float,
    )

    logged = np.full_like(
        values,
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(values)
        & (values > 0.0)
    )

    logged[valid] = np.log10(
        values[valid]
    )

    return logged


def _profile_band_values(
    profile_band,
):
    """Return log-scaled p16, p50, and p84 arrays from a profile band."""
    return (
        _safe_log10(profile_band.p16),
        _safe_log10(profile_band.p50),
        _safe_log10(profile_band.p84),
    )


def _style_radial_axis(
    ax,
):
    """Apply BayMorph radial-profile axis styling."""
    style_axes(ax)

    ax.tick_params(
        which="both",
        direction="out",
        top=False,
        right=False,
    )

    ax.tick_params(
        which="major",
        length=8,
        width=1.2,
    )

    ax.tick_params(
        which="minor",
        length=4,
        width=1.0,
    )

    ax.minorticks_on()


def plot_radial_profile_diagnostics(
    radius,
    profile_data,
    profile_model,
    profile_noise=None,
    posterior_profiles=None,
    ax=None,
):
    """Plot radial-profile diagnostics.

    Parameters
    ----------
    radius : array-like
        Radial-bin centers.
    profile_data : array-like
        Observed radial profile.
    profile_model : array-like
        Maximum-likelihood model radial profile.
    profile_noise : array-like, optional
        Radial noise profile. Kept for backwards compatibility.
    posterior_profiles : PosteriorProfileResult, optional
        Posterior radial-profile summaries returned by
        ``compute_posterior_profile_bands``.
    ax : matplotlib.axes.Axes, optional
        Existing axis whose position is replaced by a two-panel diagnostic
        layout.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure.
    axes : tuple of matplotlib.axes.Axes
        Upper profile axis and lower residual axis.
    """
    del profile_noise

    radius = np.asarray(
        radius,
        dtype=float,
    )
    profile_data = np.asarray(
        profile_data,
        dtype=float,
    )
    profile_model = np.asarray(
        profile_model,
        dtype=float,
    )

    log_data = _safe_log10(
        profile_data
    )
    log_model = _safe_log10(
        profile_model
    )

    if ax is None:
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(30, 15),
            sharex=True,
            gridspec_kw={
                "height_ratios": [3, 1],
                "hspace": 0.05,
            },
        )
    else:
        fig = ax.figure
        gs = ax.get_subplotspec().subgridspec(
            2,
            1,
            height_ratios=[3, 1],
            hspace=0.05,
        )
        ax.remove()
        ax1 = fig.add_subplot(
            gs[0]
        )
        ax2 = fig.add_subplot(
            gs[1],
            sharex=ax1,
        )

    ax1.plot(
        radius,
        log_data,
        marker="o",
        linestyle="none",
        color="0.75",
        ms=4,
        label="Observed",
    )

    if posterior_profiles is not None:
        p16, p50, p84 = _profile_band_values(
            posterior_profiles.total
        )

        ax1.fill_between(
            posterior_profiles.radius,
            p16,
            p84,
            color="0.85",
            alpha=0.8,
            linewidth=0,
            label="68% posterior band",
        )

        ax1.plot(
            posterior_profiles.radius,
            p50,
            color="k",
            lw=3,
            label="Posterior median",
        )

        for name, profile_band in posterior_profiles.components.items():
            _, component_p50, _ = _profile_band_values(
                profile_band
            )

            ax1.plot(
                posterior_profiles.radius,
                component_p50,
                lw=2,
                alpha=0.8,
                label=name,
            )

    ax1.plot(
        radius,
        log_model,
        color="r",
        linestyle="--",
        lw=3,
        label="Maximum likelihood",
    )

    residual = log_data - log_model

    ax2.axhline(
        0.0,
        color="k",
        lw=1,
    )

    ax2.plot(
        radius,
        residual,
        marker="o",
        linestyle="none",
        color="0.75",
        ms=4,
    )

    ax1.set_ylabel(
        "log10 I"
    )
    ax2.set_ylabel(
        "Delta log10 I"
    )
    ax2.set_xlabel(
        "Radius [pix]"
    )

    ax1.tick_params(
        axis="x",
        which="both",
        labelbottom=False,
    )

    ax1.legend(
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        frameon=False,
    )

    _style_radial_axis(
        ax1
    )
    _style_radial_axis(
        ax2
    )

    return fig, (ax1, ax2)