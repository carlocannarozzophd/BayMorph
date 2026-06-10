"""Corner-plot utilities for BayMorph."""

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as ss
from matplotlib import colormaps
from matplotlib.colors import Colormap
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, NullLocator
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.stats import gaussian_kde


def erf_sigma_to_percentiles(
    sigma,
):
    """Convert Gaussian sigma levels to percentile bounds."""
    sigma = np.asarray(
        sigma,
        dtype=float,
    )
    p_hi = 0.5 * (
        1.0 + ss.erf(sigma / np.sqrt(2.0))
    )
    p_lo = 1.0 - p_hi

    return p_lo, p_hi


def _as_cmap(
    cmap,
):
    if cmap is None:
        return colormaps.get_cmap("bone_r")

    if isinstance(cmap, Colormap):
        return cmap

    return colormaps.get_cmap(cmap)


def _sigma_colors(
    cmap,
    n,
    vmin=0.20,
    vmax=0.90,
):
    vals = np.linspace(
        vmin,
        vmax,
        n,
    )

    return [
        cmap(value)
        for value in vals
    ]


def _hist2d_smoothed(
    x,
    y,
    bins=20,
    smooth_sigma=0.0,
    range_xy=None,
):
    counts, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=bins,
        range=range_xy,
    )

    if smooth_sigma and smooth_sigma > 0:
        counts = gaussian_filter(
            counts,
            smooth_sigma,
        )

    counts[:, 0] = 0
    counts[:, -1] = 0
    counts[0, :] = 0
    counts[-1, :] = 0

    return counts, xedges, yedges


def corner_plot(
    parameters,
    sigma=None,
    parameter_labels=None,
    bins=30,
    fill_contours=True,
    cmap="bone_r",
    density1d="hist",
    kde_bw_method=None,
    smooth_1d=0.0,
    smooth_2d=1.0,
    show_medians=True,
    write_medians=False,
    hist_color=None,
    max_n_ticks=4,
    fig=None,
    range_xy=None,
    alpha_2d=0.95,
    alpha_1d=0.90,
    alpha_lines=1.0,
    cmap_min=0.20,
    cmap_max=0.90,
):
    """
    Create a publication-style corner plot.

    Parameters should have shape:

    - (n_params, n_samples)
    - or (n_sets, n_params, n_samples)
    """
    params = np.asarray(
        parameters,
        dtype=float,
    )

    if params.ndim == 2:
        params = np.expand_dims(
            params,
            axis=0,
        )

    if params.ndim != 3:
        raise ValueError(
            "`parameters` must have shape "
            "(n_params, n_samples) or "
            "(n_sets, n_params, n_samples)."
        )

    n_sets, n_params, _ = params.shape

    if sigma is None:
        sigma = [1, 2]

    sigma_sorted = np.sort(
        np.asarray(
            sigma,
            dtype=float,
        )
    )

    base_levels = list(
        np.exp(
            -(sigma_sorted**2) / 2.0
        )
    )
    base_levels.sort()
    levels_filled = base_levels + [1.0]

    cmap = _as_cmap(cmap)

    band_colors = _sigma_colors(
        cmap,
        len(base_levels),
        vmin=cmap_min,
        vmax=cmap_max,
    )

    inner_color = band_colors[-1]

    if parameter_labels is None:
        parameter_labels = [
            f"par{i + 1}"
            for i in range(n_params)
        ]

    if len(parameter_labels) != n_params:
        raise ValueError(
            "`parameter_labels` length must match "
            "the number of parameters."
        )

    if fig is None:
        fig, ax_grid = plt.subplots(
            n_params,
            n_params,
            figsize=(15, 15),
        )
    else:
        axes = np.array(fig.axes)

        if axes.size != n_params * n_params:
            raise ValueError(
                f"Provided figure has {axes.size} axes, "
                f"but n_params={n_params} requires "
                f"{n_params * n_params}."
            )

        ax_grid = axes.reshape(
            (n_params, n_params)
        )

    def _style_ticks(ax):
        ax.tick_params(
            axis="x",
            rotation=45,
            labelsize=20,
        )
        ax.tick_params(
            axis="y",
            rotation=45,
            labelsize=20,
        )
        ax.tick_params(
            which="major",
            length=10,
            width=3,
            direction="out",
        )
        ax.tick_params(
            which="minor",
            length=5,
            width=3,
            direction="out",
        )

        if max_n_ticks == 0:
            ax.xaxis.set_major_locator(
                NullLocator()
            )
            ax.yaxis.set_major_locator(
                NullLocator()
            )
        else:
            ax.xaxis.set_major_locator(
                MaxNLocator(
                    max_n_ticks,
                    prune="lower",
                )
            )
            ax.yaxis.set_major_locator(
                MaxNLocator(
                    max_n_ticks,
                    prune="lower",
                )
            )
            ax.xaxis.set_minor_locator(
                AutoMinorLocator()
            )
            ax.yaxis.set_minor_locator(
                AutoMinorLocator()
            )

    for i in range(n_params):
        for j in range(n_params):
            ax = ax_grid[i, j]
            _style_ticks(ax)

            if j > i:
                ax.axis("off")
                continue

            if i < n_params - 1:
                ax.set_xticklabels([])

            if j > 0:
                ax.set_yticklabels([])

            if j < i:
                for s in range(n_sets):
                    x = params[s, j]
                    y = params[s, i]

                    counts, xedges, yedges = _hist2d_smoothed(
                        x,
                        y,
                        bins=bins,
                        smooth_sigma=smooth_2d,
                        range_xy=range_xy,
                    )

                    norm = counts / (
                        counts.max()
                        if counts.max() > 0
                        else 1.0
                    )

                    extent = [
                        xedges[0],
                        xedges[-1],
                        yedges[0],
                        yedges[-1],
                    ]

                    if fill_contours:
                        ax.contourf(
                            norm.T,
                            extent=extent,
                            levels=levels_filled,
                            colors=band_colors,
                            alpha=alpha_2d,
                            antialiased=True,
                        )

                    ax.contour(
                        norm.T,
                        extent=extent,
                        levels=base_levels,
                        colors=[inner_color],
                        linewidths=1.5,
                        alpha=alpha_lines,
                    )

                    if show_medians:
                        xm = np.median(x)
                        ym = np.median(y)

                        ax.axvline(
                            xm,
                            color=inner_color,
                            lw=1,
                        )
                        ax.axhline(
                            ym,
                            color=inner_color,
                            lw=1,
                        )
                        ax.scatter(
                            xm,
                            ym,
                            marker="X",
                            color=inner_color,
                            s=60,
                            lw=0.8,
                            edgecolor="white",
                            zorder=5,
                        )

                if j == 0:
                    ax.set_ylabel(
                        parameter_labels[i],
                        fontsize=25,
                    )

            if j == i:
                xmin = np.min(
                    [
                        np.min(params[s, j])
                        for s in range(n_sets)
                    ]
                )
                xmax = np.max(
                    [
                        np.max(params[s, j])
                        for s in range(n_sets)
                    ]
                )

                for s in range(n_sets):
                    x = params[s, j]

                    if density1d.lower() == "kde":
                        kde = gaussian_kde(
                            x,
                            bw_method=kde_bw_method,
                        )
                        xgrid = np.linspace(
                            xmin,
                            xmax,
                            1200,
                        )
                        ygrid = kde.evaluate(xgrid)

                        ax.plot(
                            xgrid,
                            ygrid,
                            lw=1.5,
                            c=inner_color,
                            alpha=alpha_lines,
                        )

                        for k, sig in enumerate(
                            sigma_sorted[::-1]
                        ):
                            plo, phi = erf_sigma_to_percentiles(
                                sig
                            )
                            xl = np.percentile(
                                x,
                                100 * plo,
                            )
                            xr = np.percentile(
                                x,
                                100 * phi,
                            )
                            sel = (
                                (xgrid >= xl)
                                & (xgrid <= xr)
                            )

                            if np.any(sel):
                                ax.fill_between(
                                    xgrid[sel],
                                    ygrid[sel],
                                    0.0,
                                    facecolor=band_colors[k],
                                    alpha=alpha_1d,
                                    edgecolor=None,
                                )

                    else:
                        counts, edges = np.histogram(
                            x,
                            bins=bins,
                            range=(xmin, xmax),
                        )
                        centers = 0.5 * (
                            edges[:-1] + edges[1:]
                        )

                        if smooth_1d and smooth_1d > 0:
                            counts = gaussian_filter1d(
                                counts.astype(float),
                                smooth_1d,
                            )

                        line_color = (
                            hist_color
                            if hist_color is not None
                            else inner_color
                        )

                        ax.plot(
                            centers,
                            counts,
                            lw=1.5,
                            c=line_color,
                            alpha=alpha_lines,
                        )

                        interp = interp1d(
                            centers,
                            counts,
                            kind="linear",
                            bounds_error=False,
                            fill_value=0.0,
                        )
                        xgrid = np.linspace(
                            xmin,
                            xmax,
                            1000,
                        )
                        ygrid = interp(xgrid)

                        for k, sig in enumerate(
                            sigma_sorted[::-1]
                        ):
                            plo, phi = erf_sigma_to_percentiles(
                                sig
                            )
                            xl = np.percentile(
                                x,
                                100 * plo,
                            )
                            xr = np.percentile(
                                x,
                                100 * phi,
                            )
                            sel = (
                                (xgrid >= xl)
                                & (xgrid <= xr)
                            )

                            if np.any(sel):
                                ax.fill_between(
                                    xgrid[sel],
                                    ygrid[sel],
                                    0.0,
                                    facecolor=band_colors[k],
                                    alpha=alpha_1d,
                                    edgecolor=None,
                                )

                    med = float(np.median(x))
                    plo1, phi1 = erf_sigma_to_percentiles(
                        1.0
                    )
                    q16 = med - float(
                        np.percentile(
                            x,
                            100 * plo1,
                        )
                    )
                    q84 = float(
                        np.percentile(
                            x,
                            100 * phi1,
                        )
                    ) - med

                    if show_medians:
                        ax.axvline(
                            med,
                            color=inner_color,
                            lw=1.2,
                            ls="--",
                        )

                    if write_medians:
                        def fmt(val):
                            return f"{val:.3f}"

                        title = (
                            rf"${fmt(med)}_"
                            rf"{{-{fmt(q16)}}}"
                            rf"^{{+{fmt(q84)}}}$"
                        )

                        ax.set_title(
                            f"{parameter_labels[j]} = {title}",
                            fontsize=20,
                        )

                ax.set_ylim(bottom=0)
                ax.set_xlim(
                    left=xmin,
                    right=xmax,
                )
                ax_grid[-1, j].set_xlabel(
                    parameter_labels[j],
                    fontsize=25,
                )

    for k in range(n_params):
        ax = ax_grid[k, k]
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.yaxis.set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    plt.subplots_adjust(
        left=0.10,
        bottom=0.10,
    )
    fig.align_labels(ax_grid)

    return fig, ax_grid, params