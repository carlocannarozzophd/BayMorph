"""Radial-profile plotting utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from baymorph.plotting.style import (
    computer_modern_text,
    style_axes,
)


def _safe_log10(values):
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
        np.isfinite(
            values
        )
        & (
            values > 0.0
        )
    )

    logged[
        valid
    ] = np.log10(
        values[
            valid
        ]
    )

    return logged


def _profile_band_values(
    profile_band,
):
    """Return log-scaled p16, p50, and p84 arrays from a profile band."""
    return (
        _safe_log10(
            profile_band.p16
        ),
        _safe_log10(
            profile_band.p50
        ),
        _safe_log10(
            profile_band.p84
        ),
    )


def _same_radius_grid(
    radius_a,
    radius_b,
) -> bool:
    """Return True if two radius grids are compatible."""
    radius_a = np.asarray(
        radius_a,
        dtype=float,
    )
    radius_b = np.asarray(
        radius_b,
        dtype=float,
    )

    return (
        radius_a.shape
        == radius_b.shape
        and np.allclose(
            radius_a,
            radius_b,
            equal_nan=True,
        )
    )


def _style_radial_axis(
    ax,
):
    """Apply BayMorph radial-profile axis styling."""
    style_axes(
        ax
    )

    for spine in ax.spines.values():
        spine.set_linewidth(
            3
        )

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
    posterior_profiles=None,
    component_profiles=None,
    *,
    profile_data_p16=None,
    profile_data_p84=None,
    profile_data_mean=None,
    legend_title=None,
    title=None,
    ax=None,
):
    """Plot elliptical radial-profile diagnostics.

    Parameters
    ----------
    radius : array-like
        Semi-major-axis radial-bin centres.
    profile_data : array-like
        Median observed intensity in each annulus.
    profile_model : array-like
        Maximum-likelihood model radial profile.
    posterior_profiles : PosteriorProfileResult, optional
        Posterior radial-profile summaries.
    component_profiles : dict, optional
        Maximum-likelihood component radial profiles.
    profile_data_p16, profile_data_p84 : array-like, optional
        Sixteenth and eighty-fourth percentiles of observed pixel values in
        each annulus. When both are supplied, they are shown as asymmetric
        error bars around the median.
    profile_data_mean : array-like, optional
        Mean observed intensity in each annulus.
    legend_title : str, optional
        Legend title.
    title : str, optional
        Object label displayed above standalone radial diagnostics.
    ax : matplotlib.axes.Axes, optional
        Existing axis replaced by the two-panel diagnostic layout.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure.
    axes : tuple of matplotlib.axes.Axes
        Profile and residual axes.

    Notes
    -----
    The observed p16--p84 range describes pixel scatter within each annulus.
    It should not generally be interpreted as the sampling uncertainty of the
    median estimator.
    """

    label_size = 35
    tick_size = 35
    legend_size = 35

    observed_marker_size = 25
    observed_alpha = 0.65

    ml_color = "mediumvioletred"
    ml_linestyle = "--"
    ml_linewidth = 3

    posterior_color = "black"
    posterior_linewidth = 3

    band_color = "0.75"
    band_alpha = 0.75

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

    if (
        radius.shape
        != profile_data.shape
        or radius.shape
        != profile_model.shape
    ):
        raise ValueError(
            "radius, profile_data, and profile_model must have matching "
            "shapes."
        )

    log_data = _safe_log10(
        profile_data
    )
    log_model = _safe_log10(
        profile_model
    )

    log_data_p16 = None
    log_data_p84 = None

    if (
        profile_data_p16 is None
    ) != (
        profile_data_p84 is None
    ):
        raise ValueError(
            "profile_data_p16 and profile_data_p84 must be supplied together."
        )

    if profile_data_p16 is not None:
        profile_data_p16 = np.asarray(
            profile_data_p16,
            dtype=float,
        )
        profile_data_p84 = np.asarray(
            profile_data_p84,
            dtype=float,
        )

        if (
            profile_data_p16.shape
            != radius.shape
            or profile_data_p84.shape
            != radius.shape
        ):
            raise ValueError(
                "Observed percentile profiles must match the radius shape."
            )

        log_data_p16 = _safe_log10(
            profile_data_p16
        )
        log_data_p84 = _safe_log10(
            profile_data_p84
        )

    log_data_mean = None

    if profile_data_mean is not None:
        profile_data_mean = np.asarray(
            profile_data_mean,
            dtype=float,
        )

        if profile_data_mean.shape != radius.shape:
            raise ValueError(
                "profile_data_mean must match the radius shape."
            )

        log_data_mean = _safe_log10(
            profile_data_mean
        )

    valid = (
        np.isfinite(
            radius
        )
        & np.isfinite(
            log_data
        )
        & np.isfinite(
            log_model
        )
    )

    posterior_total = None

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
            np.isfinite(
                posterior_radius
            )
            & np.isfinite(
                p16
            )
            & np.isfinite(
                p50
            )
            & np.isfinite(
                p84
            )
        )

        if _same_radius_grid(
            posterior_radius,
            radius,
        ):
            valid &= posterior_valid

    if not np.any(
        valid
    ):
        valid = (
            np.isfinite(
                radius
            )
            & np.isfinite(
                log_data
            )
            & np.isfinite(
                log_model
            )
        )

    radius_plot = radius[
        valid
    ]
    log_data_plot = log_data[
        valid
    ]
    log_model_plot = log_model[
        valid
    ]

    if ax is None:
        fig, (
            ax1,
            ax2,
        ) = plt.subplots(
            2,
            1,
            figsize=(
                30,
                15,
            ),
            sharex=True,
            gridspec_kw={
                "height_ratios": [
                    3,
                    1,
                ],
                "hspace": 0.0,
            },
        )

    else:
        fig = ax.figure

        gs = (
            ax
            .get_subplotspec()
            .subgridspec(
                2,
                1,
                height_ratios=[
                    3,
                    1,
                ],
                hspace=0.0,
            )
        )

        ax.remove()

        ax1 = fig.add_subplot(
            gs[
                0
            ]
        )

        ax2 = fig.add_subplot(
            gs[
                1
            ],
            sharex=ax1,
        )

    if title is not None:
        title = computer_modern_text(
            title
        )

    if title and ax is None:
        fig.suptitle(
            str(title),
            fontsize=42,
            y=0.985,
        )

        fig.subplots_adjust(
            top=0.90,
        )

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
        label="Observed median",
    )

    if (
        log_data_p16 is not None
        and log_data_p84 is not None
    ):
        interval_valid = (
            valid
            & np.isfinite(
                log_data_p16
            )
            & np.isfinite(
                log_data_p84
            )
            & (
                log_data_p16
                <= log_data
            )
            & (
                log_data
                <= log_data_p84
            )
        )

        if np.any(
            interval_valid
        ):
            lower = (
                log_data[
                    interval_valid
                ]
                - log_data_p16[
                    interval_valid
                ]
            )

            upper = (
                log_data_p84[
                    interval_valid
                ]
                - log_data[
                    interval_valid
                ]
            )

            ax1.errorbar(
                radius[
                    interval_valid
                ],
                log_data[
                    interval_valid
                ],
                yerr=np.vstack(
                    (
                        lower,
                        upper,
                    )
                ),
                fmt="none",
                ecolor="royalblue",
                elinewidth=2,
                capsize=4,
                alpha=0.65,
                label=computer_modern_text(
                    "Observed 16--84% pixel range"
                ),
            )

    if log_data_mean is not None:
        mean_valid = (
            valid
            & np.isfinite(
                log_data_mean
            )
        )

        ax1.plot(
            radius[
                mean_valid
            ],
            log_data_mean[
                mean_valid
            ],
            marker="D",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="darkorange",
            markeredgewidth=2,
            ms=14,
            label="Observed mean",
        )

    if posterior_total is not None:
        if _same_radius_grid(
            posterior_total[
                "radius"
            ],
            radius,
        ):
            posterior_plot_valid = valid
        else:
            posterior_plot_valid = (
                np.isfinite(
                    posterior_total[
                        "radius"
                    ]
                )
                & np.isfinite(
                    posterior_total[
                        "p16"
                    ]
                )
                & np.isfinite(
                    posterior_total[
                        "p50"
                    ]
                )
                & np.isfinite(
                    posterior_total[
                        "p84"
                    ]
                )
            )

        posterior_radius_plot = posterior_total[
            "radius"
        ][
            posterior_plot_valid
        ]

        posterior_p16_plot = posterior_total[
            "p16"
        ][
            posterior_plot_valid
        ]

        posterior_p50_plot = posterior_total[
            "p50"
        ][
            posterior_plot_valid
        ]

        posterior_p84_plot = posterior_total[
            "p84"
        ][
            posterior_plot_valid
        ]

        ax1.fill_between(
            posterior_radius_plot,
            posterior_p16_plot,
            posterior_p84_plot,
            color=band_color,
            alpha=band_alpha,
            linewidth=0,
            label=computer_modern_text(
                "68% posterior interval"
            ),
        )

        ax1.plot(
            posterior_radius_plot,
            posterior_p50_plot,
            color=posterior_color,
            lw=posterior_linewidth,
            label="Posterior median",
        )

    if component_profiles is not None:
        for (
            name,
            component_profile,
        ) in component_profiles.items():
            component_profile = np.asarray(
                component_profile,
                dtype=float,
            )

            if component_profile.shape != radius.shape:
                continue

            component_log = _safe_log10(
                component_profile
            )

            component_valid = (
                valid
                & np.isfinite(
                    component_log
                )
            )

            ax1.plot(
                radius[
                    component_valid
                ],
                component_log[
                    component_valid
                ],
                lw=3,
                alpha=0.85,
                label=name,
            )

    ax1.plot(
        radius_plot,
        log_model_plot,
        color=ml_color,
        linestyle=ml_linestyle,
        lw=ml_linewidth,
        label="Maximum likelihood",
    )

    data_residual = (
        log_data_plot
        - log_model_plot
    )

    ax2.axhline(
        0.0,
        color=ml_color,
        linestyle=ml_linestyle,
        lw=ml_linewidth,
    )

    residual_values = [
        data_residual,
    ]

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
    )

    if (
        log_data_p16 is not None
        and log_data_p84 is not None
    ):
        interval_valid = (
            valid
            & np.isfinite(
                log_data_p16
            )
            & np.isfinite(
                log_data_p84
            )
            & (
                log_data_p16
                <= log_data
            )
            & (
                log_data
                <= log_data_p84
            )
        )

        if np.any(
            interval_valid
        ):
            median_residual = (
                log_data[
                    interval_valid
                ]
                - log_model[
                    interval_valid
                ]
            )

            lower = (
                log_data[
                    interval_valid
                ]
                - log_data_p16[
                    interval_valid
                ]
            )

            upper = (
                log_data_p84[
                    interval_valid
                ]
                - log_data[
                    interval_valid
                ]
            )

            ax2.errorbar(
                radius[
                    interval_valid
                ],
                median_residual,
                yerr=np.vstack(
                    (
                        lower,
                        upper,
                    )
                ),
                fmt="none",
                ecolor="royalblue",
                elinewidth=2,
                capsize=4,
                alpha=0.65,
            )

    if log_data_mean is not None:
        mean_valid = (
            valid
            & np.isfinite(
                log_data_mean
            )
        )

        mean_residual = (
            log_data_mean[
                mean_valid
            ]
            - log_model[
                mean_valid
            ]
        )

        ax2.plot(
            radius[
                mean_valid
            ],
            mean_residual,
            marker="D",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="darkorange",
            markeredgewidth=2,
            ms=14,
        )

        residual_values.append(
            mean_residual
        )

    if posterior_total is not None:
        if _same_radius_grid(
            posterior_total[
                "radius"
            ],
            radius,
        ):
            posterior_residual_valid = valid

            posterior_model_reference = log_model[
                posterior_residual_valid
            ]

        else:
            posterior_residual_valid = (
                np.isfinite(
                    posterior_total[
                        "radius"
                    ]
                )
                & np.isfinite(
                    posterior_total[
                        "p16"
                    ]
                )
                & np.isfinite(
                    posterior_total[
                        "p50"
                    ]
                )
                & np.isfinite(
                    posterior_total[
                        "p84"
                    ]
                )
            )

            posterior_model_reference = np.interp(
                posterior_total[
                    "radius"
                ][
                    posterior_residual_valid
                ],
                radius_plot,
                log_model_plot,
            )

        posterior_radius_res = posterior_total[
            "radius"
        ][
            posterior_residual_valid
        ]

        posterior_p16_res = posterior_total[
            "p16"
        ][
            posterior_residual_valid
        ]

        posterior_p50_res = posterior_total[
            "p50"
        ][
            posterior_residual_valid
        ]

        posterior_p84_res = posterior_total[
            "p84"
        ][
            posterior_residual_valid
        ]

        posterior_low_residual = (
            posterior_p16_res
            - posterior_model_reference
        )

        posterior_med_residual = (
            posterior_p50_res
            - posterior_model_reference
        )

        posterior_high_residual = (
            posterior_p84_res
            - posterior_model_reference
        )

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

    finite_residuals = [
        values[
            np.isfinite(
                values
            )
        ]
        for values in residual_values
        if np.any(
            np.isfinite(
                values
            )
        )
    ]

    if finite_residuals:
        ymax = np.nanmax(
            np.abs(
                np.concatenate(
                    finite_residuals
                )
            )
        )

        if (
            np.isfinite(
                ymax
            )
            and ymax > 0.0
        ):
            ax2.set_ylim(
                -1.05 * ymax,
                1.05 * ymax,
            )

    ax1.set_ylabel(
        r"$\log_{10} I$",
        fontsize=label_size,
    )

    ax2.set_ylabel(
        r"$\Delta \log_{10} I$",
        fontsize=label_size,
    )

    ax2.set_xlabel(
        r"$R_{\rm ell}\,[\mathrm{pix}]$",
        fontsize=label_size,
    )

    ax1.tick_params(
        axis="x",
        which="both",
        labelbottom=False,
    )

    _style_radial_axis(
        ax1
    )
    _style_radial_axis(
        ax2
    )

    ax1.tick_params(
        labelsize=tick_size
    )

    ax2.tick_params(
        labelsize=tick_size
    )

    ax1.legend(
        loc="upper right",
        fontsize=legend_size,
        frameon=False,
        title=legend_title,
        title_fontsize=legend_size + 5,
    )

    return (
        fig,
        (
            ax1,
            ax2,
        ),
    )
