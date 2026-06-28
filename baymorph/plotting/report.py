"""High-level report figures for BayMorph."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from .images import BAYMORPH_IMAGE_CMAP, _log_limits


def _show_log(ax, image, title, cmap=BAYMORPH_IMAGE_CMAP):
    vmin, vmax = _log_limits(image)
    image = np.clip(np.asarray(image, dtype=float), vmin, vmax)
    im = ax.imshow(image, origin="lower", cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _show_linear(ax, image, title, cmap="RdBu_r", symmetric=True):
    image = np.asarray(image, dtype=float)

    if symmetric:
        vmax = np.nanpercentile(np.abs(image), 99)
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        vmin = -vmax
    else:
        vmin = np.nanpercentile(image, 1)
        vmax = np.nanpercentile(image, 99)

    im = ax.imshow(image, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def plot_mock_fit_report(
    intrinsic_image,
    psf,
    convolved_image,
    sky_image,
    noise_realization,
    observed_image,
    best_model_image,
    noise_map,
    output_path=None,
    show=True,
):
    """Plot a compact mock-observation and BayMorph fit report."""
    residual = observed_image - best_model_image

    norm_residual = np.full_like(residual, np.nan, dtype=float)
    valid = np.isfinite(noise_map) & (noise_map > 0)
    norm_residual[valid] = residual[valid] / noise_map[valid]

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.suptitle("BayMorph mock-observation and fit report", fontsize=18)

    im = _show_log(axes[0, 0], intrinsic_image, "Intrinsic Sérsic")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046)

    im = _show_log(axes[0, 1], psf, "Gaussian PSF")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046)

    im = _show_log(axes[0, 2], convolved_image, "PSF-convolved galaxy")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046)

    im = _show_linear(axes[0, 3], sky_image, "Sky background", cmap="viridis", symmetric=False)
    fig.colorbar(im, ax=axes[0, 3], fraction=0.046)

    im = _show_linear(axes[0, 4], noise_realization, "Gaussian noise")
    fig.colorbar(im, ax=axes[0, 4], fraction=0.046)

    im = _show_log(axes[1, 0], observed_image, "Final mock observation")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046)

    im = _show_log(axes[1, 1], best_model_image, "Best-fit model")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046)

    im = _show_linear(axes[1, 2], residual, "Residual")
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046)

    im = _show_linear(axes[1, 3], norm_residual, "Residual / noise")
    fig.colorbar(im, ax=axes[1, 3], fraction=0.046)

    axes[1, 4].axis("off")
    axes[1, 4].text(
        0.5,
        0.5,
        "Final mock observation\n=\ninput to BayMorph\n\nBest-fit model\n=\nDynesty reconstruction",
        ha="center",
        va="center",
        fontsize=14,
    )

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes
