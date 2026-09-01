"""Image plotting utilities."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

from baymorph.plotting.style import make_plot, set_plot_style, style_axes

BAYMORPH_IMAGE_CMAP = "gist_stern"


def _log_limits(
    *images,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.9,
    floor: float = 1e-3,
) -> tuple[float, float]:
    """Return robust positive limits for logarithmic image display."""
    positive_values = []

    for image in images:
        values = np.asarray(
            image,
            dtype=float,
        )

        values = values[np.isfinite(values) & (values > 0)]

        if values.size > 0:
            positive_values.append(values)

    if len(positive_values) == 0:
        return floor, 1.0

    positive_values = np.concatenate(
        positive_values,
    )

    vmin = np.percentile(
        positive_values,
        lower_percentile,
    )

    vmax = np.percentile(
        positive_values,
        upper_percentile,
    )

    vmin = max(
        float(vmin),
        floor,
    )

    vmax = max(
        float(vmax),
        vmin * 10,
    )

    return vmin, vmax


def show_log_image(
    image,
    title: str,
    cmap: str = BAYMORPH_IMAGE_CMAP,
    colorbar_label: str = "Flux",
    show: bool = False,
):
    """Display an image using robust logarithmic normalization.

    Parameters
    ----------
    image : array-like
        Input image.
    title : str
        Figure title.
    cmap : str, default=BAYMORPH_IMAGE_CMAP
        Matplotlib colormap used for the image.
    colorbar_label : str, default="Flux"
        Colorbar label.
    show : bool, default=False
        If True, call :func:`matplotlib.pyplot.show`.

    Returns
    -------
    None
        The figure is created using the active Matplotlib state.
    """
    vmin, vmax = _log_limits(
        image,
    )

    display_image = np.clip(
        np.asarray(
            image,
            dtype=float,
        ),
        vmin,
        vmax,
    )

    plt.figure(
        figsize=(6, 6),
    )

    plt.imshow(
        display_image,
        origin="lower",
        cmap=cmap,
        norm=LogNorm(
            vmin=vmin,
            vmax=vmax,
        ),
    )

    plt.colorbar(
        label=colorbar_label,
    )

    plt.title(
        f"{title} (log scale)",
    )

    if show:
        plt.show()


def plot_log_image(
    image,
    title: str = "",
    cmap: str = BAYMORPH_IMAGE_CMAP,
    colorbar_label: str = "Flux",
    figsize: tuple[float, float] = (15, 15),
    show: bool = False,
):
    """Plot a single image using BayMorph logarithmic styling.

    Parameters
    ----------
    image : array-like
        Input image.
    title : str, default=""
        Optional axis title.
    cmap : str, default=BAYMORPH_IMAGE_CMAP
        Matplotlib colormap used for the image.
    colorbar_label : str, default="Flux"
        Colorbar label.
    figsize : tuple of float, default=(15, 15)
        Figure size in inches.
    show : bool, default=False
        If True, call :func:`matplotlib.pyplot.show`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure.
    ax : matplotlib.axes.Axes
        Image axis.
    """
    set_plot_style()

    image = np.asarray(
        image,
        dtype=float,
    )

    vmin, vmax = _log_limits(
        image,
    )

    display_image = np.clip(
        image,
        vmin,
        vmax,
    )

    fig, ax = plt.subplots(
        figsize=figsize,
    )

    style_axes(
        ax,
    )

    im = ax.imshow(
        display_image,
        origin="lower",
        cmap=cmap,
        norm=LogNorm(
            vmin=vmin,
            vmax=vmax,
        ),
    )

    if title:
        ax.set_title(
            title,
        )

    ax.set_xlabel(
        "x [pix]",
    )
    ax.set_ylabel(
        "y [pix]",
    )

    fig.colorbar(
        im,
        ax=ax,
        label=colorbar_label,
    )

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_log_data_model_residual(
    data,
    model,
    noise=None,
    title: str = "",
    cmap: str = BAYMORPH_IMAGE_CMAP,
    show: bool = False,
):
    """Plot data, model, residual, and optionally residual divided by noise.

    Parameters
    ----------
    data : array-like
        Observed image.
    model : array-like
        Model image.
    noise : array-like, optional
        Noise map. If provided, a normalized residual panel is added.
    title : str, default=""
        Optional figure title.
    cmap : str, default=BAYMORPH_IMAGE_CMAP
        Matplotlib colormap used for data and model panels.
    show : bool, default=False
        If True, call :func:`matplotlib.pyplot.show`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure.
    axes : ndarray
        Array of Matplotlib axes.
    """
    set_plot_style()

    data = np.asarray(
        data,
        dtype=float,
    )

    model = np.asarray(
        model,
        dtype=float,
    )

    residual = data - model

    if noise is not None:
        noise = np.asarray(
            noise,
            dtype=float,
        )

        norm_residual = np.full_like(
            residual,
            np.nan,
            dtype=float,
        )

        valid_noise = np.isfinite(noise) & (noise > 0)

        norm_residual[valid_noise] = residual[valid_noise] / noise[valid_noise]
    else:
        norm_residual = None

    vmin, vmax = _log_limits(
        data,
        model,
    )

    data_display = np.clip(
        data,
        vmin,
        vmax,
    )

    model_display = np.clip(
        model,
        vmin,
        vmax,
    )

    residual_limit = np.nanpercentile(
        np.abs(residual),
        99,
    )

    if not np.isfinite(residual_limit) or residual_limit == 0:
        residual_limit = 1.0

    if norm_residual is None:
        ncols = 3
        figsize = (45, 15)
    else:
        ncols = 4
        figsize = (60, 15)

    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=figsize,
    )

    style_axes(
        axes,
    )

    data_image = axes[0].imshow(
        data_display,
        origin="lower",
        cmap=cmap,
        norm=LogNorm(
            vmin=vmin,
            vmax=vmax,
        ),
    )

    axes[0].set_title("Data (log)")

    fig.colorbar(
        data_image,
        ax=axes[0],
    )

    model_image = axes[1].imshow(
        model_display,
        origin="lower",
        cmap=cmap,
        norm=LogNorm(
            vmin=vmin,
            vmax=vmax,
        ),
    )

    axes[1].set_title("Model (log)")

    fig.colorbar(
        model_image,
        ax=axes[1],
    )

    residual_image = axes[2].imshow(
        residual,
        origin="lower",
        cmap="RdBu_r",
        vmin=-residual_limit,
        vmax=residual_limit,
    )

    axes[2].set_title("Residual")

    fig.colorbar(
        residual_image,
        ax=axes[2],
    )

    if norm_residual is not None:
        norm_limit = np.nanpercentile(
            np.abs(norm_residual),
            99,
        )

        if not np.isfinite(norm_limit) or norm_limit == 0:
            norm_limit = 5.0

        norm_limit = min(
            float(norm_limit),
            5.0,
        )

        norm_image = axes[3].imshow(
            norm_residual,
            origin="lower",
            cmap="RdBu_r",
            vmin=-norm_limit,
            vmax=norm_limit,
        )

        axes[3].set_title("Residual / Noise")

        fig.colorbar(
            norm_image,
            ax=axes[3],
            label=r"$\sigma$",
        )

    if title:
        fig.suptitle(
            title,
            y=1.02,
        )

    plt.tight_layout()

    if show:
        plt.show()

    return fig, axes


def plot_fit_summary(
    data,
    model,
    noise=None,
    title: str = "",
    cmap: str = BAYMORPH_IMAGE_CMAP,
    show: bool = False,
):
    """Plot a standard BayMorph fit summary.

    Parameters
    ----------
    data : array-like
        Observed image.
    model : array-like
        Model image.
    noise : array-like, optional
        Noise map used to compute normalized residuals.
    title : str, default=""
        Optional figure title.
    cmap : str, default=BAYMORPH_IMAGE_CMAP
        Matplotlib colormap used for data and model panels.
    show : bool, default=False
        If True, call :func:`matplotlib.pyplot.show`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure.
    axes : ndarray
        Array of Matplotlib axes.
    """
    return plot_log_data_model_residual(
        data=data,
        model=model,
        noise=noise,
        title=title,
        cmap=cmap,
        show=show,
    )


def _apply_baymorph_axes_style(ax):
    """Apply BayMorph image-axis styling."""
    for spine in ax.spines.values():
        spine.set_linewidth(3)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.minorticks_off()


def _robust_symmetric_limit(image, percentile=99.0, fallback=1.0):
    """Return a robust symmetric display limit."""
    values = np.asarray(image, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return float(fallback)

    limit = np.nanpercentile(np.abs(values), percentile)

    if not np.isfinite(limit) or limit <= 0:
        return float(fallback)

    return float(limit)


def _combined_mask_rgb(contaminant_mask=None, target_mask=None):
    """Return RGB masks: contaminants lightgray, target black."""
    shape = None

    if contaminant_mask is not None:
        contaminant_mask = np.asarray(contaminant_mask, dtype=bool)
        shape = contaminant_mask.shape

    if target_mask is not None:
        target_mask = np.asarray(target_mask, dtype=bool)
        shape = target_mask.shape if shape is None else shape

    if shape is None:
        return None

    image = np.ones((*shape, 3), dtype=float)

    if contaminant_mask is not None:
        image[contaminant_mask] = (0.82, 0.82, 0.82)

    if target_mask is not None:
        image[target_mask] = (1.0, 0.84, 0.0)

    return image


def _contaminant_mask_rgba(
    contaminant_mask,
    image_shape,
):
    """Return an RGBA overlay for pixels excluded from the fit."""
    if contaminant_mask is None:
        return None

    mask = np.asarray(
        contaminant_mask,
        dtype=bool,
    )

    if mask.shape != image_shape:
        raise ValueError(
            "contaminant_mask shape must match the prepared image."
        )

    overlay = np.zeros(
        (*image_shape, 4),
        dtype=float,
    )

    overlay[
        mask
    ] = (
        0.82,
        0.82,
        0.82,
        1.0,
    )

    return overlay


def _imshow_log(ax, image, *, title, cmap, vmin=None, vmax=None):
    """Display log-scaled image."""
    image = np.asarray(image, dtype=float)

    if vmin is None or vmax is None:
        vmin, vmax = _log_limits(image)

    display = np.clip(image, vmin, vmax)

    im = ax.imshow(
        display,
        origin="lower",
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )

    ax.set_title(title, fontsize=35)
    _apply_baymorph_axes_style(ax)

    return im


def _add_inset_colorbar(fig, ax, image, label=None):
    """Add a colorbar without changing the image-panel size."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)

    cax = divider.append_axes(
        "right",
        size="4%",
        pad=0.15,
    )

    cbar = fig.colorbar(
        image,
        cax=cax,
    )

    cbar.outline.set_linewidth(3)

    cbar.ax.tick_params(
        which="major",
        length=10,
        width=3,
        labelsize=25,
    )

    cbar.ax.tick_params(
        which="minor",
        length=5,
        width=3,
    )

    if label is not None:
        cbar.set_label(
            label,
            fontsize=30,
        )

    return cbar


def plot_pipeline_fit_maps(
    original_cutout,
    prepared_cutout,
    model,
    noise,
    segmentation=None,
    psf=None,
    contaminant_mask=None,
    target_mask=None,
    *,
    original_image=None,
    cutout_bounds=None,
    title: str = "",
    cmap: str = BAYMORPH_IMAGE_CMAP,
    show: bool = False,
):
    """Plot the final BayMorph 8-panel fit diagnostic report."""
    del segmentation

    set_plot_style()

    original_cutout = np.asarray(original_cutout, dtype=float)
    prepared_cutout = np.asarray(prepared_cutout, dtype=float)
    model = np.asarray(model, dtype=float)
    noise = np.asarray(noise, dtype=float)

    if original_image is None:
        original_image = original_cutout
    else:
        original_image = np.asarray(original_image, dtype=float)

    row_vmin, row_vmax = _log_limits(
        original_image,
        original_cutout,
        prepared_cutout,
        model,
    )

    residual = prepared_cutout - model

    norm_residual = np.full_like(
        residual,
        np.nan,
        dtype=float,
    )

    valid_noise = (
        np.isfinite(noise)
        & (noise > 0)
    )

    norm_residual[valid_noise] = residual[valid_noise] / noise[valid_noise]

    mask_overlay = _contaminant_mask_rgba(
        contaminant_mask,
        residual.shape,
    )

    if contaminant_mask is None:
        residual_scale_values = residual
        norm_scale_values = norm_residual
    else:
        fit_mask = np.asarray(
            contaminant_mask,
            dtype=bool,
        )

        residual_scale_values = np.where(
            fit_mask,
            np.nan,
            residual,
        )

        norm_scale_values = np.where(
            fit_mask,
            np.nan,
            norm_residual,
        )

    residual_limit = _robust_symmetric_limit(
        residual_scale_values,
        fallback=1.0,
    )

    norm_limit = _robust_symmetric_limit(
        norm_scale_values,
        fallback=5.0,
    )

    norm_limit = min(
        float(norm_limit),
        5.0,
    )

    mask_rgb = _combined_mask_rgb(
        contaminant_mask=contaminant_mask,
        target_mask=target_mask,
    )

    fig, axes = make_plot(
        2,
        4,
        flatten=False,
        constrained_layout=False,
    )

    fig.subplots_adjust(
        left=0.035,
        right=0.965,
        bottom=0.06,
        top=0.90,
        wspace=0.10,
        hspace=0.18,
    )

    if title:
        fig.suptitle(
            title,
            fontsize=40,
            y=0.985,
        )

    axes = np.asarray(axes)

    _imshow_log(
        axes[0, 0],
        original_image,
        title="Original image",
        cmap=cmap,
        vmin=row_vmin,
        vmax=row_vmax,
    )

    if cutout_bounds is not None:
        xmin = cutout_bounds.get("xmin")
        xmax = cutout_bounds.get("xmax")
        ymin = cutout_bounds.get("ymin")
        ymax = cutout_bounds.get("ymax")

        if None not in (xmin, xmax, ymin, ymax):
            axes[0, 0].add_patch(
                Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    fill=False,
                    edgecolor="black",
                    linewidth=3,
                )
            )

    _imshow_log(
        axes[0, 1],
        original_cutout,
        title="Original cutout",
        cmap=cmap,
        vmin=row_vmin,
        vmax=row_vmax,
    )

    im_prepared = _imshow_log(
        axes[0, 2],
        prepared_cutout,
        title="Prepared cutout",
        cmap=cmap,
        vmin=row_vmin,
        vmax=row_vmax,
    )

    im_model = _imshow_log(
        axes[0, 3],
        model,
        title="Model",
        cmap=cmap,
        vmin=row_vmin,
        vmax=row_vmax,
    )

    _add_inset_colorbar(fig, axes[0, 2], im_prepared)
    _add_inset_colorbar(fig, axes[0, 3], im_model)

    axes[1, 0].set_title(
        r"Target \& Masks",
        fontsize=35,
    )
    _apply_baymorph_axes_style(axes[1, 0])

    if mask_rgb is None:
        axes[1, 0].text(
            0.5,
            0.5,
            "Masks not available",
            ha="center",
            va="center",
            fontsize=25,
            transform=axes[1, 0].transAxes,
        )
    else:
        axes[1, 0].imshow(
            mask_rgb,
            origin="lower",
            interpolation="nearest",
        )

    axes[1, 1].set_title(
        "PSF",
        fontsize=35,
    )
    _apply_baymorph_axes_style(axes[1, 1])

    axes[1, 1].set_aspect(
        "equal",
        adjustable="box",
    )

    axes[1, 1].set_anchor(
        "C",
    )

    if psf is None:
        axes[1, 1].text(
            0.5,
            0.5,
            "PSF not provided",
            ha="center",
            va="center",
            fontsize=25,
            transform=axes[1, 1].transAxes,
        )
    else:
        psf = np.asarray(psf, dtype=float)
        psf_vmin, psf_vmax = _log_limits(psf)
        psf_display = np.clip(psf, psf_vmin, psf_vmax)

        axes[1, 1].imshow(
            psf_display,
            origin="lower",
            cmap=cmap,
            norm=LogNorm(
                vmin=psf_vmin,
                vmax=psf_vmax,
            ),
        )

    axes[1, 2].set_title(
        "Residual",
        fontsize=35,
    )
    _apply_baymorph_axes_style(axes[1, 2])

    residual_image = axes[1, 2].imshow(
        residual,
        origin="lower",
        cmap="BrBG",
        vmin=-residual_limit,
        vmax=residual_limit,
    )

    if mask_overlay is not None:
        axes[1, 2].imshow(
            mask_overlay,
            origin="lower",
            interpolation="nearest",
        )

    _add_inset_colorbar(
        fig,
        axes[1, 2],
        residual_image,
    )

    axes[1, 3].set_title(
        "Residual / noise",
        fontsize=35,
    )
    _apply_baymorph_axes_style(axes[1, 3])

    norm_image = axes[1, 3].imshow(
        norm_residual,
        origin="lower",
        cmap="PuOr",
        vmin=-norm_limit,
        vmax=norm_limit,
    )

    if mask_overlay is not None:
        axes[1, 3].imshow(
            mask_overlay,
            origin="lower",
            interpolation="nearest",
        )

    _add_inset_colorbar(
        fig,
        axes[1, 3],
        norm_image,
    )

    if show:
        plt.show()

    return fig, axes