"""High-level report figures for BayMorph."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from baymorph.plotting.images import BAYMORPH_IMAGE_CMAP, _log_limits


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
    vmin=None,
    vmax=None,
):
    """Display an image with linear normalization."""
    image = np.asarray(image, dtype=float)

    if vmin is None or vmax is None:
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


def _percent_residual(
    data,
    model,
    *,
    mask=None,
):
    """Return percent residual, optionally masked outside the galaxy."""
    data = np.asarray(data, dtype=float)
    model = np.asarray(model, dtype=float)

    percent = np.full_like(data, np.nan, dtype=float)
    valid = (
        np.isfinite(data)
        & np.isfinite(model)
        & (data != 0.0)
    )
    percent[valid] = 100.0 * (data[valid] - model[valid]) / data[valid]

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        percent[~mask] = np.nan

    return percent


def _residual_over_noise(
    residual,
    noise_map,
):
    """Return residual divided by noise."""
    residual = np.asarray(residual, dtype=float)
    noise_map = np.asarray(noise_map, dtype=float)

    norm_residual = np.full_like(residual, np.nan, dtype=float)
    valid = (
        np.isfinite(residual)
        & np.isfinite(noise_map)
        & (noise_map > 0.0)
    )
    norm_residual[valid] = residual[valid] / noise_map[valid]

    return norm_residual


def _get_optional_image(
    source,
    *names,
):
    """Return the first available image attribute from an object."""
    for name in names:
        if hasattr(source, name):
            return getattr(source, name)

    return None


def _show_optional_mask(
    ax,
    image,
    title,
):
    """Display an optional mask image."""
    if image is None:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Not available",
            ha="center",
            va="center",
            fontsize=30,
        )
        ax.set_title(title, fontsize=30)
        return None

    return _show_linear(
        ax,
        image,
        title,
        cmap="gray",
        symmetric=False,
        vmin=0.0,
        vmax=1.0,
    )


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
    """Plot a BayMorph reconstruction report.

    Parameters
    ----------
    observation : object
        Observation object exposing image products used in the report.
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
    original_cutout = _get_optional_image(
        observation,
        "original_cutout",
        "intrinsic_image",
        "image",
    )
    prepared_cutout = _get_optional_image(
        observation,
        "prepared_cutout",
        "convolved_image",
        "image",
    )
    observed_image = _get_optional_image(
        observation,
        "image",
        "observed_image",
    )
    psf = _get_optional_image(
        observation,
        "psf_kernel",
        "psf",
    )
    segmentation = _get_optional_image(
        observation,
        "segmentation",
        "segmentation_map",
    )
    contaminant_mask = _get_optional_image(
        observation,
        "contaminant_mask",
        "mask",
    )
    target_mask = _get_optional_image(
        observation,
        "target_mask",
        "galaxy_mask",
    )

    if observed_image is None:
        observed_image = prepared_cutout

    best_model_image = fit.best_model_image
    residual = observed_image - best_model_image
    percent_residual = _percent_residual(
        observed_image,
        best_model_image,
        mask=target_mask,
    )
    norm_residual = _residual_over_noise(
        residual,
        noise_map,
    )

    has_diagnostic_row = (
        corner_plotter is not None
        or radial_profile_plotter is not None
    )

    if has_diagnostic_row:
        fig = plt.figure(figsize=(40, 24), constrained_layout=False)

        gs = fig.add_gridspec(
            3,
            5,
            height_ratios=[1.0, 1.0, 2.2],
        )

        axes = np.empty((2, 5), dtype=object)

        for row in range(2):
            for col in range(5):
                axes[row, col] = fig.add_subplot(gs[row, col])

        corner_ax = fig.add_subplot(gs[2, 0:2], box_aspect=1)
        radial_ax = fig.add_subplot(gs[2, 2:5], box_aspect=1)

    else:
        fig, axes = plt.subplots(
            2,
            5,
            figsize=(40, 15),
            constrained_layout=False,
        )

        corner_ax = None
        radial_ax = None

    fig.suptitle(
        "BayMorph Bayesian Reconstruction",
        fontsize=30,
        y=0.98,
    )

    im = _show_log(axes[0, 0], original_cutout, "Original cutout")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046)

    im = _show_log(axes[0, 1], prepared_cutout, "Prepared cutout")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046)

    im = _show_log(axes[0, 2], best_model_image, "Model")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046)

    im = _show_linear(
        axes[0, 3],
        residual,
        "Residual",
        cmap="RdBu_r",
        symmetric=True,
    )
    fig.colorbar(im, ax=axes[0, 3], fraction=0.046)

    im = _show_linear(
        axes[0, 4],
        percent_residual,
        "Percent residual",
        cmap="RdBu_r",
        symmetric=True,
    )
    fig.colorbar(im, ax=axes[0, 4], fraction=0.046)

    im = _show_optional_mask(
        axes[1, 0],
        segmentation,
        "Segmentation",
    )
    if im is not None:
        fig.colorbar(im, ax=axes[1, 0], fraction=0.046)

    if psf is not None:
        im = _show_log(axes[1, 1], psf, "PSF")
        fig.colorbar(im, ax=axes[1, 1], fraction=0.046)
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.5,
            0.5,
            "No PSF",
            ha="center",
            va="center",
            fontsize=30,
        )
        axes[1, 1].set_title("PSF", fontsize=30)

    im = _show_optional_mask(
        axes[1, 2],
        contaminant_mask,
        "Contaminant mask",
    )
    if im is not None:
        fig.colorbar(im, ax=axes[1, 2], fraction=0.046)

    im = _show_optional_mask(
        axes[1, 3],
        target_mask,
        "Target mask",
    )
    if im is not None:
        fig.colorbar(im, ax=axes[1, 3], fraction=0.046)

    im = _show_linear(
        axes[1, 4],
        norm_residual,
        "Residual / noise",
        cmap="RdBu_r",
        symmetric=False,
        vmin=-5.0,
        vmax=5.0,
    )
    fig.colorbar(im, ax=axes[1, 4], fraction=0.046)

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

    if statistics is not None:
        stats_ax = fig.add_axes([0.78, 0.02, 0.18, 0.16])
        _draw_statistics_panel(
            stats_ax,
            statistics,
        )

    if has_diagnostic_row:
        fig.subplots_adjust(
            left=0.04,
            right=0.98,
            bottom=0.06,
            top=0.92,
            wspace=0.45,
            hspace=0.35,
        )
    else:
        fig.subplots_adjust(
            left=0.04,
            right=0.98,
            bottom=0.06,
            top=0.90,
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
