"""High-level report figures for BayMorph."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from .images import BAYMORPH_IMAGE_CMAP, _log_limits


def _style_axes(
    ax,
    spine_width=3.0,
    major_length=10,
    minor_length=5,
):
    """Apply BayMorph-style axis spines and ticks."""
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)

    ax.minorticks_on()

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=major_length,
        width=spine_width,
        top=True,
        right=True,
    )

    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        length=minor_length,
        width=spine_width,
        top=True,
        right=True,
    )


def _show_log(ax, image, title, cmap=BAYMORPH_IMAGE_CMAP):
    """Display an image with robust logarithmic normalization."""
    vmin, vmax = _log_limits(image)
    image = np.clip(np.asarray(image, dtype=float), vmin, vmax)

    im = ax.imshow(
        image,
        origin="lower",
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )

    ax.set_title(title, fontsize=30)
    ax.set_xticks([])
    ax.set_yticks([])

    _style_axes(ax)

    return im


def _show_linear(
    ax,
    image,
    title,
    cmap="RdBu_r",
    symmetric=True,
):
    """Display an image with linear normalization."""
    image = np.asarray(image, dtype=float)

    if symmetric:
        vmax = np.nanpercentile(np.abs(image), 99)
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        vmin = -vmax
    else:
        vmin = np.nanpercentile(image, 1)
        vmax = np.nanpercentile(image, 99)

    im = ax.imshow(
        image,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(title, fontsize=30)
    ax.set_xticks([])
    ax.set_yticks([])

    _style_axes(ax)

    return im


def _format_stat_label(key):
    """Format a statistic name for display in report figures."""
    key_lower = key.lower()

    if key_lower.startswith("reduced"):
        return r"Reduced $\chi^2$"

    if key_lower in {"chi2", "chi^2"}:
        return r"$\chi^2$"

    if key_lower == "logz":
        return r"$\log Z$"

    if key_lower == "logzerr":
        return r"$\log Z_{\rm err}$"

    return key


def _draw_statistics_panel(ax, statistics):
    """Draw a text panel summarizing scalar fit statistics."""
    ax.axis("off")

    y = 0.95

    ax.text(
        0.02,
        y,
        "BayMorph Statistics",
        fontsize=30,
        fontweight="bold",
        va="top",
    )

    y -= 0.14

    if not statistics:
        ax.text(
            0.02,
            y,
            "No statistics provided.",
            fontsize=24,
            ha="left",
            va="top",
        )
        return

    for key, value in statistics.items():
        label = _format_stat_label(key)

        if isinstance(value, float):
            value = f"{value:.4g}"

        ax.text(
            0.02,
            y,
            label,
            fontsize=26,
            ha="left",
            va="top",
        )

        ax.text(
            0.98,
            y,
            str(value),
            fontsize=26,
            ha="right",
            va="top",
        )

        y -= 0.10


def plot_mock_fit_report(
    observation,
    fit,
    noise_map,
    statistics=None,
    corner_plotter=None,
    radial_profile_plotter=None,
    output_path=None,
    show=True,
):
    """Plot a mock-observation and BayMorph reconstruction report.

    Parameters
    ----------
    observation : SyntheticObservation
        Synthetic observation containing intrinsic, convolved, sky, noise, and
        observed images.
    fit : object
        Fit result object exposing at least ``best_model_image``.
    noise_map : array-like
        Noise map used in the likelihood.
    statistics : dict, optional
        Scalar statistics displayed in the report panel.
    corner_plotter : callable, optional
        Callable receiving a Matplotlib axis-like target and drawing a corner
        diagnostic.
    radial_profile_plotter : callable, optional
        Callable receiving a Matplotlib axis-like target and drawing a radial
        profile diagnostic.
    output_path : str or pathlib.Path, optional
        Path where the figure is saved. If None, the figure is not written to
        disk.
    show : bool, default=True
        If True, call :func:`matplotlib.pyplot.show`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output report figure.
    axes : ndarray
        Array of axes for the two main report rows.
    """
    intrinsic_image = observation.intrinsic_image
    psf = observation.psf_kernel
    convolved_image = observation.convolved_image
    sky_image = observation.sky_image
    noise_realization = observation.noise_realization
    observed_image = observation.image

    best_model_image = fit.best_model_image

    residual = observed_image - best_model_image

    norm_residual = np.full_like(residual, np.nan, dtype=float)
    valid = np.isfinite(noise_map) & (noise_map > 0)
    norm_residual[valid] = residual[valid] / noise_map[valid]

    has_diagnostic_row = (
        corner_plotter is not None
        or radial_profile_plotter is not None
    )

    if has_diagnostic_row:
        fig = plt.figure(figsize=(40, 24), constrained_layout=False)

        gs = fig.add_gridspec(
            3,
            6,
            height_ratios=[1.0, 1.0, 2.2],
        )

        axes = np.empty((2, 6), dtype=object)

        for row in range(2):
            for col in range(6):
                axes[row, col] = fig.add_subplot(gs[row, col])

        corner_ax = fig.add_subplot(gs[2, 0:3], box_aspect=1)
        radial_ax = fig.add_subplot(gs[2, 3:6], box_aspect=1)

    else:
        fig, axes = plt.subplots(
            2,
            6,
            figsize=(40, 15),
            constrained_layout=False,
        )

        corner_ax = None
        radial_ax = None

    fig.suptitle(
        "BayMorph Mock Observation and Bayesian Reconstruction",
        fontsize=30,
        y=0.98,
    )

    fig.text(
        0.5,
        0.90,
        "Mock Observation",
        ha="center",
        fontsize=30,
    )

    fig.text(
        0.5,
        0.70,
        "BayMorph Reconstruction",
        ha="center",
        fontsize=30,
    )

    im = _show_log(axes[0, 0], intrinsic_image, "Intrinsic Galaxy")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046)

    if psf is not None:
        im = _show_log(axes[0, 1], psf, "PSF")
        fig.colorbar(im, ax=axes[0, 1], fraction=0.046)
    else:
        axes[0, 1].axis("off")
        axes[0, 1].text(
            0.5,
            0.5,
            "No PSF",
            ha="center",
            va="center",
            fontsize=30,
        )

    im = _show_log(axes[0, 2], convolved_image, "PSF-convolved")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046)

    im = _show_linear(
        axes[0, 3],
        sky_image,
        "Sky Background",
        cmap="viridis",
        symmetric=False,
    )
    fig.colorbar(im, ax=axes[0, 3], fraction=0.046)

    im = _show_linear(
        axes[0, 4],
        noise_realization,
        "Noise",
        cmap="RdBu_r",
        symmetric=True,
    )
    fig.colorbar(im, ax=axes[0, 4], fraction=0.046)

    im = _show_log(axes[0, 5], observed_image, "Observed Image")
    fig.colorbar(im, ax=axes[0, 5], fraction=0.046)

    im = _show_log(axes[1, 0], observed_image, "Observed Image")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046)

    im = _show_log(axes[1, 1], best_model_image, "Best-fit model")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046)

    im = _show_linear(
        axes[1, 2],
        residual,
        "Residual",
        cmap="RdBu_r",
        symmetric=True,
    )
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046)

    im = _show_linear(
        axes[1, 3],
        norm_residual,
        "Residual / Noise",
        cmap="RdBu_r",
        symmetric=True,
    )
    fig.colorbar(im, ax=axes[1, 3], fraction=0.046)

    im = _show_linear(
        axes[1, 4],
        noise_map,
        "Noise Map",
        cmap="bone",
        symmetric=False,
    )
    fig.colorbar(im, ax=axes[1, 4], fraction=0.046)

    _draw_statistics_panel(
        axes[1, 5],
        statistics,
    )

    if has_diagnostic_row:
        if corner_plotter is not None:
            corner_plotter(corner_ax)
            _style_axes(corner_ax)
        else:
            corner_ax.axis("off")

        if radial_profile_plotter is not None:
            radial_profile_plotter(radial_ax)
        else:
            radial_ax.axis("off")

    if has_diagnostic_row:
        fig.subplots_adjust(
            left=0.04,
            right=0.98,
            bottom=0.04,
            top=0.88,
            wspace=0.45,
            hspace=0.35,
        )
    else:
        fig.subplots_adjust(
            left=0.04,
            right=0.98,
            bottom=0.06,
            top=0.84,
            wspace=0.45,
            hspace=0.50,
        )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )

    if show:
        plt.show()

    return fig, axes