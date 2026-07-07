"""Image plotting utilities."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from .style import set_plot_style, style_axes

BAYMORPH_IMAGE_CMAP = "gist_earth"


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
    figsize: tuple[float, float] = (8, 8),
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
    figsize : tuple of float, default=(8, 8)
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


def plot_pipeline_fit_maps(
    original_cutout,
    prepared_cutout,
    model,
    noise,
    segmentation=None,
    psf=None,
    contaminant_mask=None,
    target_mask=None,
    title: str = "",
    cmap: str = BAYMORPH_IMAGE_CMAP,
    show: bool = False,
):
    """Plot the final BayMorph pipeline map summary.

    The summary figure is organized in two rows. The first row shows
    the original target cutout, prepared fitting cutout, best-fit model,
    data-minus-model residual, and percent residual relative to the
    prepared cutout. The second row shows the segmentation map, PSF,
    contaminant mask, target segment, and noise-normalized residual map.

    Parameters
    ----------
    original_cutout : array-like
        Target cutout before background subtraction.
    prepared_cutout : array-like
        Background-subtracted cutout used for fitting.
    model : array-like
        Best-fit model image evaluated on the prepared cutout grid.
    noise : array-like
        Noise map associated with ``prepared_cutout``.
    segmentation : array-like, optional
        Integer segmentation map.
    psf : array-like, optional
        PSF image or kernel used for the fit.
    contaminant_mask : array-like, optional
        Boolean mask identifying contaminating sources.
    target_mask : array-like, optional
        Boolean mask identifying the target segment.
    title : str, default=""
        Optional figure title.
    cmap : str, default=BAYMORPH_IMAGE_CMAP
        Matplotlib colormap used for logarithmic image panels.
    show : bool, default=False
        If True, call :func:`matplotlib.pyplot.show`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure.
    axes : ndarray
        Two-dimensional array of Matplotlib axes.
    """
    set_plot_style()

    original_cutout = np.asarray(original_cutout, dtype=float)
    prepared_cutout = np.asarray(prepared_cutout, dtype=float)
    model = np.asarray(model, dtype=float)
    noise = np.asarray(noise, dtype=float)

    residual = prepared_cutout - model

    percent_residual = np.full_like(residual, np.nan, dtype=float)

    scale = np.nanpercentile(
        np.abs(prepared_cutout),
        95,
    )

    valid_data = (
        np.isfinite(prepared_cutout)
        & (np.abs(prepared_cutout) > 0.01 * scale)
    )

    percent_residual[valid_data] = (
        100.0 * residual[valid_data] / prepared_cutout[valid_data]
    )

    norm_residual = np.full_like(residual, np.nan, dtype=float)
    valid_noise = np.isfinite(noise) & (noise > 0)
    norm_residual[valid_noise] = residual[valid_noise] / noise[valid_noise]

    vmin, vmax = _log_limits(original_cutout, prepared_cutout, model)

    original_display = np.clip(original_cutout, vmin, vmax)
    prepared_display = np.clip(prepared_cutout, vmin, vmax)
    model_display = np.clip(model, vmin, vmax)

    residual_limit = np.nanpercentile(np.abs(residual), 99)
    if not np.isfinite(residual_limit) or residual_limit == 0:
        residual_limit = 1.0

    percent_limit = np.nanpercentile(np.abs(percent_residual), 99)
    if not np.isfinite(percent_limit) or percent_limit == 0:
        percent_limit = 100.0
    percent_limit = min(float(percent_limit), 100.0)

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(30, 12),
        constrained_layout=True,
    )

    style_axes(axes.ravel())

    panels = [
        (
            axes[0, 0],
            original_display,
            "Original cutout",
            cmap,
            LogNorm(vmin=vmin, vmax=vmax),
            None,
        ),
        (
            axes[0, 1],
            prepared_display,
            "Prepared cutout",
            cmap,
            LogNorm(vmin=vmin, vmax=vmax),
            None,
        ),
        (
            axes[0, 2],
            model_display,
            "Best-fit model",
            cmap,
            LogNorm(vmin=vmin, vmax=vmax),
            None,
        ),
        (
            axes[0, 3],
            residual,
            "Residual",
            "RdBu_r",
            None,
            (-residual_limit, residual_limit),
        ),
        (
            axes[0, 4],
            percent_residual,
            "Percent residual",
            "RdBu_r",
            None,
            (-percent_limit, percent_limit),
        ),
    ]

    for ax, image, panel_title, panel_cmap, norm, limits in panels:
        kwargs = {"origin": "lower", "cmap": panel_cmap}

        if norm is not None:
            kwargs["norm"] = norm
        elif limits is not None:
            kwargs["vmin"], kwargs["vmax"] = limits

        im = ax.imshow(image, **kwargs)
        ax.set_title(panel_title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)

    row2 = [
        (segmentation, "Segmentation", "tab20", None),
        (psf, "PSF", cmap, None),
        (contaminant_mask, "Contaminant mask", "gray", (0, 1)),
        (target_mask, "Target segment", "gray", (0, 1)),
        (norm_residual, "Residual / noise", "RdBu_r", (-5.0, 5.0)),
    ]

    for ax, (image, panel_title, panel_cmap, limits) in zip(axes[1], row2):
        ax.set_title(panel_title)
        ax.set_xticks([])
        ax.set_yticks([])

        if image is None:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "Not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        image = np.asarray(image, dtype=float)
        kwargs = {"origin": "lower", "cmap": panel_cmap}

        if panel_title == "PSF":
            psf_vmin, psf_vmax = _log_limits(image)
            image = np.clip(image, psf_vmin, psf_vmax)
            kwargs["norm"] = LogNorm(vmin=psf_vmin, vmax=psf_vmax)

        if limits is not None:
            kwargs["vmin"], kwargs["vmax"] = limits

        im = ax.imshow(image, **kwargs)

        if panel_title == "Residual / noise":
            fig.colorbar(im, ax=ax, fraction=0.046, label=r"$\sigma$")
        else:
            fig.colorbar(im, ax=ax, fraction=0.046)

    if title:
        fig.suptitle(title, y=1.02)

    if show:
        plt.show()

    return fig, axes
