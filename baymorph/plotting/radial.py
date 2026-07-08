"""Radial-profile plotting utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .style import style_axes


def _safe_log10(values):
    """Return base-10 logarithms with invalid values masked as NaN."""
    values = np.asarray(values, dtype=float)

    logged = np.full_like(
        values,
        np.nan,
        dtype=float,
    )

    valid = np.isfinite(values) & (values > 0.0)

    logged[valid] = np.log10(values[valid])

    return logged


def _profile_band_values(profile_band):
    """Return log-scaled p16, p50, and p84 arrays from a profile band."""
    return (
        _safe_log10(profile_band.p16),
        _safe_log10(profile_band.p50),
        _safe_log10(profile_band.p84),
    )


def _same_radius_grid(radius_a, radius_b) -> bool:
    """Return True if two radius grids are compatible."""
    radius_a = np.asarray(radius_a, dtype=float)
    radius_b = np.asarray(radius_b, dtype=float)

    return (
        radius_a.shape == radius_b.shape
        and np.allclose(
            radius_a,
            radius_b,
            equal_nan=True,
        )
    )


def _style_radial_axis(ax):
    """Apply BayMorph radial-profile axis styling."""
    style_axes(ax)

    for spine in ax.spines.values():
        spine.set_linewidth(3)

    ax.tick_params(
        which="both",
        direction="out",
        top=False,
        right=False,
        width=3,
    )

    ax.tick_params(
        which="major",
        length=10,
        width=3,
    )

    ax.tick_params(
        which="minor",
        length=5,
        width=3,
    )

    ax.minorticks_on()


def plot_radial_profile_diagnostics(
    radius,
    profile_data,
    profile_model,
    profile_noise=None,
    posterior_profiles=None,
    *,
    legend_title=None,
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
    legend_title : str, optional
        Title displayed at the top of the legend. Typically corresponds to
        the fitted morphology model.
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

    label_size = 35
    tick_size = 35
    legend_size = 35

    observed_marker_size = 25
    observed_alpha = 0.5

    ml_color = "mediumvioletred"
    ml_linestyle = "--"
    ml_linewidth = 3

    posterior_color = "black"
    posterior_linewidth = 3

    band_color = "0.75"
    band_alpha = 0.75

    radius = np.asarray(radius, dtype=float)
    profile_data = np.asarray(profile_data, dtype=float)
    profile_model = np.asarray(profile_model, dtype=float)

    log_data = _safe_log10(profile_data)
    log_model = _safe_log10(profile_model)

    valid = (
        np.isfinite(radius)
        & np.isfinite(log_data)
        & np.isfinite(log_model)
    )

    posterior_total = None
    posterior_components = None

    if posterior_profiles is not None:
        posterior_radius = np.asarray(
            posterior_profiles.radius,
            dtype=float,
        )

        p16, p50, p84 = _profile_band_values(
            posterior_profiles.total
        )

        posterior_total = {
            "radius": posterior_radius,
            "p16": p16,
            "p50": p50,
            "p84": p84,
        }

        posterior_valid = (
            np.isfinite(posterior_radius)
            & np.isfinite(p16)
            & np.isfinite(p50)
            & np.isfinite(p84)
        )

        if _same_radius_grid(posterior_radius, radius):
            valid &= posterior_valid

        component_items = list(
            posterior_profiles.components.items()
        )

        if len(component_items) > 1:
            posterior_components = []

            for name, profile_band in component_items:
                _, component_p50, _ = _profile_band_values(
                    profile_band
                )

                component = {
                    "name": name,
                    "radius": posterior_radius,
                    "p50": component_p50,
                }

                posterior_components.append(component)

                component_valid = (
                    np.isfinite(posterior_radius)
                    & np.isfinite(component_p50)
                )

                if _same_radius_grid(posterior_radius, radius):
                    valid &= component_valid

    if not np.any(valid):
        valid = (
            np.isfinite(radius)
            & np.isfinite(log_data)
            & np.isfinite(log_model)
        )

    radius_plot = radius[valid]
    log_data_plot = log_data[valid]
    log_model_plot = log_model[valid]

    if ax is None:
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(30, 15),
            sharex=True,
            gridspec_kw={
                "height_ratios": [3, 1],
                "hspace": 0.0,
            },
        )
    else:
        fig = ax.figure
        gs = ax.get_subplotspec().subgridspec(
            2,
            1,
            height_ratios=[3, 1],
            hspace=0.0,
        )
        ax.remove()
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)

    ax1.plot(
        radius_plot,
        log_data_plot,
        marker="o",
        linestyle="none",
        markerfacecolor="royalblue",
        markeredgecolor="white",
        markeredgewidth=1.5,
        alpha=observed_alpha,
        ms=observed_marker_size,
        label="Observed",
    )

    if posterior_total is not None:
        if _same_radius_grid(posterior_total["radius"], radius):
            posterior_plot_valid = valid
        else:
            posterior_plot_valid = (
                np.isfinite(posterior_total["radius"])
                & np.isfinite(posterior_total["p16"])
                & np.isfinite(posterior_total["p50"])
                & np.isfinite(posterior_total["p84"])
            )

        posterior_radius_plot = posterior_total["radius"][posterior_plot_valid]
        posterior_p16_plot = posterior_total["p16"][posterior_plot_valid]
        posterior_p50_plot = posterior_total["p50"][posterior_plot_valid]
        posterior_p84_plot = posterior_total["p84"][posterior_plot_valid]

        ax1.fill_between(
            posterior_radius_plot,
            posterior_p16_plot,
            posterior_p84_plot,
            color=band_color,
            alpha=band_alpha,
            linewidth=0,
            label="68% posterior band",
        )

        ax1.plot(
            posterior_radius_plot,
            posterior_p50_plot,
            color=posterior_color,
            lw=posterior_linewidth,
            label="Posterior median",
        )

    if posterior_components is not None:
        for component in posterior_components:
            if _same_radius_grid(component["radius"], radius):
                component_plot_valid = valid
            else:
                component_plot_valid = (
                    np.isfinite(component["radius"])
                    & np.isfinite(component["p50"])
                )

            ax1.plot(
                component["radius"][component_plot_valid],
                component["p50"][component_plot_valid],
                lw=3,
                alpha=0.8,
                label=component["name"],
            )

    ax1.plot(
        radius_plot,
        log_model_plot,
        color=ml_color,
        linestyle=ml_linestyle,
        lw=ml_linewidth,
        label="Maximum likelihood",
    )

    data_residual = log_data_plot - log_model_plot

    ax2.axhline(
        0.0,
        color=ml_color,
        linestyle=ml_linestyle,
        lw=ml_linewidth,
    )

    residual_values = [data_residual]

    if posterior_total is not None:
        if _same_radius_grid(posterior_total["radius"], radius):
            posterior_residual_valid = valid
            posterior_model_reference = log_model[posterior_residual_valid]
        else:
            posterior_residual_valid = (
                np.isfinite(posterior_total["radius"])
                & np.isfinite(posterior_total["p16"])
                & np.isfinite(posterior_total["p50"])
                & np.isfinite(posterior_total["p84"])
            )
            posterior_model_reference = np.interp(
                posterior_total["radius"][posterior_residual_valid],
                radius_plot,
                log_model_plot,
            )

        posterior_radius_res = posterior_total["radius"][posterior_residual_valid]
        posterior_p16_res = posterior_total["p16"][posterior_residual_valid]
        posterior_p50_res = posterior_total["p50"][posterior_residual_valid]
        posterior_p84_res = posterior_total["p84"][posterior_residual_valid]

        posterior_low_residual = posterior_p16_res - posterior_model_reference
        posterior_med_residual = posterior_p50_res - posterior_model_reference
        posterior_high_residual = posterior_p84_res - posterior_model_reference

        ax2.fill_between(
            posterior_radius_res,
            posterior_low_residual,
            posterior_high_residual,
            color=band_color,
            alpha=band_alpha,
            linewidth=0,
        )

        ax2.plot(
            posterior_radius_res,
            posterior_med_residual,
            color=posterior_color,
            lw=posterior_linewidth,
        )

        residual_values.extend(
            [
                posterior_low_residual,
                posterior_med_residual,
                posterior_high_residual,
            ]
        )

    if posterior_components is not None:
        for component in posterior_components:
            if _same_radius_grid(component["radius"], radius):
                component_residual_valid = valid
                component_model_reference = log_model[component_residual_valid]
            else:
                component_residual_valid = (
                    np.isfinite(component["radius"])
                    & np.isfinite(component["p50"])
                )
                component_model_reference = np.interp(
                    component["radius"][component_residual_valid],
                    radius_plot,
                    log_model_plot,
                )

            component_radius_res = component["radius"][component_residual_valid]
            component_p50_res = component["p50"][component_residual_valid]
            component_residual = component_p50_res - component_model_reference

            ax2.plot(
                component_radius_res,
                component_residual,
                lw=3,
                alpha=0.8,
            )

            residual_values.append(component_residual)

    ax2.plot(
            radius_plot,
            data_residual,
            marker="o",
            linestyle="none",
            markerfacecolor="royalblue",
            markeredgecolor="white",
            markeredgewidth=1.5,
            alpha=observed_alpha,
            ms=observed_marker_size,
            label="Observed",
        )

    finite_residuals = [
        values[np.isfinite(values)]
        for values in residual_values
        if np.any(np.isfinite(values))
    ]

    if finite_residuals:
        ymax = np.nanmax(
            np.abs(
                np.concatenate(finite_residuals)
            )
        )

        if np.isfinite(ymax) and ymax > 0.0:
            ax2.set_ylim(
                -1.05 * ymax,
                1.05 * ymax,
            )

    ax1.set_ylabel(
        r"$\log I$",
        fontsize=label_size,
    )

    ax2.set_ylabel(
        r"$\Delta \log I$",
        fontsize=label_size,
    )

    ax2.set_xlabel(
        "Radius [pix]",
        fontsize=label_size,
    )

    ax1.tick_params(
        axis="x",
        which="both",
        labelbottom=False,
    )

    _style_radial_axis(ax1)
    _style_radial_axis(ax2)

    ax1.tick_params(labelsize=tick_size)
    ax2.tick_params(labelsize=tick_size)

    ax1.legend(
        loc="upper right",
        fontsize=legend_size,
        frameon=False,
        title=legend_title,
        title_fontsize=legend_size+5,
    )

    return fig, (ax1, ax2)