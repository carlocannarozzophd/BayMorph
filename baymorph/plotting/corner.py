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
    """Convert Gaussian sigma levels to enclosed percentile bounds."""
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
    """Return a Matplotlib colormap instance."""
    if cmap is None:
        return colormaps.get_cmap("bone_r")

    if isinstance(cmap, Colormap):
        return cmap

    return colormaps.get_cmap(cmap)


_PARAMETER_LABELS_LATEX = {
    "flux": r"$F$",
    "re": r"$R_{\rm e}$",
    "n": r"$n$",
    "q": r"$q$",
    "theta": r"$\theta$",
    "x0": r"$x_0$",
    "y0": r"$y_0$",
    "sky": r"$S$",
    "flux_bulge": r"$F_{\rm bulge}$",
    "re_bulge": r"$R_{{\rm e},{\rm bulge}}$",
    "n_bulge": r"$n_{\rm bulge}$",
    "q_bulge": r"$q_{\rm bulge}$",
    "theta_bulge": r"$\theta_{\rm bulge}$",
    "flux_disk": r"$F_{\rm disk}$",
    "re_disk": r"$R_{{\rm e},{\rm disk}}$",
    "n_disk": r"$n_{\rm disk}$",
    "q_disk": r"$q_{\rm disk}$",
    "theta_disk": r"$\theta_{\rm disk}$",
}


def _latex_parameter_label(label):
    """Return a LaTeX label for a BayMorph parameter name."""
    if label is None:
        return label

    label = str(label)

    if label.startswith("$") and label.endswith("$"):
        return label

    return _PARAMETER_LABELS_LATEX.get(
        label,
        label,
    )


def _sigma_colors(
    cmap,
    n,
    vmin=0.20,
    vmax=0.90,
):
    """Sample contour colors from a colormap."""
    vals = np.linspace(
        vmin,
        vmax,
        n,
    )

    return [
        cmap(value)
        for value in vals
    ]


def _style_corner_axis(
    ax,
    spine_width=3,
    major_length=10,
    minor_length=5,
    max_n_ticks=4,
):
    """Apply BayMorph corner-plot styling to one axis."""
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)

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
        length=major_length,
        width=spine_width,
        direction="out",
    )
    ax.tick_params(
        which="minor",
        length=minor_length,
        width=spine_width,
        direction="out",
    )

    if max_n_ticks == 0:
        ax.xaxis.set_major_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())
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
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())


def _hist2d_smoothed(
    x,
    y,
    bins=20,
    smooth_sigma=0.0,
    range_xy=None,
):
    """Compute a smoothed two-dimensional histogram."""
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


def _resolve_corner_grid(
    n_params,
    fig=None,
    ax=None,
):
    """Resolve or create the axes grid used by a corner plot."""
    if ax is not None and fig is not None:
        raise ValueError("Pass either `fig` or `ax`, not both.")

    if ax is not None:
        ax.axis("off")
        subfig = ax.figure
        bbox = ax.get_position()
        grid_spec = subfig.add_gridspec(
            n_params,
            n_params,
            left=bbox.x0,
            right=bbox.x1,
            bottom=bbox.y0,
            top=bbox.y1,
            wspace=0.05,
            hspace=0.05,
        )
        ax_grid = np.empty(
            (n_params, n_params),
            dtype=object,
        )

        for i in range(n_params):
            for j in range(n_params):
                ax_grid[i, j] = subfig.add_subplot(
                    grid_spec[i, j]
                )

        return subfig, ax_grid

    if fig is None:
        fig, ax_grid = plt.subplots(
            n_params,
            n_params,
            figsize=(15, 15),
        )
        return fig, ax_grid

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

    return fig, ax_grid


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
    ax=None,
    range_xy=None,
    alpha_2d=0.95,
    alpha_1d=0.90,
    alpha_lines=1.0,
    cmap_min=0.20,
    cmap_max=0.90,
):
    """Create a publication-style corner plot.

    Parameters
    ----------
    parameters : array-like
        Posterior samples with shape ``(n_params, n_samples)`` or
        ``(n_sets, n_params, n_samples)``.
    sigma : sequence of float, optional
        Gaussian-equivalent sigma levels used for contours and one-dimensional
        credible intervals. If None, ``[1, 2]`` is used.
    parameter_labels : sequence of str, optional
        Labels displayed on the plot axes.
    bins : int, default=30
        Number of histogram bins.
    fill_contours : bool, default=True
        If True, draw filled two-dimensional contour regions.
    cmap : str or matplotlib.colors.Colormap, default="bone_r"
        Colormap used for credible regions.
    density1d : {"hist", "kde"}, default="hist"
        Method used for one-dimensional marginal distributions.
    kde_bw_method : str, scalar, or callable, optional
        Bandwidth method passed to :class:`scipy.stats.gaussian_kde`.
    smooth_1d : float, default=0.0
        Gaussian smoothing scale applied to one-dimensional histograms.
    smooth_2d : float, default=1.0
        Gaussian smoothing scale applied to two-dimensional histograms.
    show_medians : bool, default=True
        If True, draw median markers and median lines.
    write_medians : bool, default=False
        If True, write median and one-sigma interval values in diagonal panel
        titles.
    hist_color : color-like, optional
        Color used for one-dimensional histogram lines. If None, the inner
        contour color is used.
    max_n_ticks : int, default=4
        Maximum number of major ticks per axis. If zero, major ticks are
        hidden.
    fig : matplotlib.figure.Figure, optional
        Existing figure containing an ``n_params`` by ``n_params`` axes grid.
    ax : matplotlib.axes.Axes, optional
        Existing axis whose bounding box is replaced by an embedded corner
        grid.
    range_xy : sequence, optional
        Range passed to two-dimensional histograms.
    alpha_2d : float, default=0.95
        Alpha value for filled two-dimensional contours.
    alpha_1d : float, default=0.90
        Alpha value for one-dimensional credible-region fills.
    alpha_lines : float, default=1.0
        Alpha value for contour and marginal-distribution lines.
    cmap_min : float, default=0.20
        Lower colormap sampling value.
    cmap_max : float, default=0.90
        Upper colormap sampling value.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the corner plot.
    ax_grid : ndarray
        Two-dimensional array of Matplotlib axes.
    params : ndarray
        Parameter array used internally, always with shape
        ``(n_sets, n_params, n_samples)``.

    Raises
    ------
    ValueError
        If the input parameter array has an invalid shape, if the number of
        labels does not match the number of parameters, or if incompatible
        figure/axis arguments are supplied.
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

    parameter_labels = [
        _latex_parameter_label(label)
        for label in parameter_labels
    ]

    fig, ax_grid = _resolve_corner_grid(
        n_params=n_params,
        fig=fig,
        ax=ax,
    )

    for i in range(n_params):
        for j in range(n_params):
            axis = ax_grid[i, j]
            _style_corner_axis(
                axis,
                max_n_ticks=max_n_ticks,
            )

            if j > i:
                axis.axis("off")
                continue

            if i < n_params - 1:
                axis.set_xticklabels([])

            if j > 0:
                axis.set_yticklabels([])

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
                        axis.contourf(
                            norm.T,
                            extent=extent,
                            levels=levels_filled,
                            colors=band_colors,
                            alpha=alpha_2d,
                            antialiased=True,
                        )

                    axis.contour(
                        norm.T,
                        extent=extent,
                        levels=base_levels,
                        colors=[inner_color],
                        linewidths=3,
                        alpha=alpha_lines,
                    )

                    if show_medians:
                        xm = np.median(x)
                        ym = np.median(y)

                        axis.axvline(
                            xm,
                            color=inner_color,
                            lw=3,
                        )
                        axis.axhline(
                            ym,
                            color=inner_color,
                            lw=3,
                        )
                        axis.scatter(
                            xm,
                            ym,
                            marker="X",
                            color=inner_color,
                            s=60,
                            lw=3,
                            edgecolor="white",
                            zorder=5,
                        )

                if j == 0:
                    axis.set_ylabel(
                        parameter_labels[i],
                        fontsize=35,
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

                        axis.plot(
                            xgrid,
                            ygrid,
                            lw=3,
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
                                axis.fill_between(
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

                        axis.plot(
                            centers,
                            counts,
                            lw=3,
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
                                axis.fill_between(
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
                        axis.axvline(
                            med,
                            color=inner_color,
                            lw=3,
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

                        axis.set_title(
                            f"{parameter_labels[j]} = {title}",
                            fontsize=25,
                        )

                axis.set_ylim(bottom=0)
                axis.set_xlim(
                    left=xmin,
                    right=xmax,
                )
                ax_grid[-1, j].set_xlabel(
                    parameter_labels[j],
                    fontsize=35,
                )

    for k in range(n_params):
        axis = ax_grid[k, k]
        axis.set_yticks([])
        axis.set_yticklabels([])
        axis.yaxis.set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["top"].set_visible(False)

    if ax is None:
        plt.subplots_adjust(
            left=0.10,
            bottom=0.10,
        )

    fig.align_labels(ax_grid)

    return fig, ax_grid, params