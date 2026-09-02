"""Empirical PSF construction utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from baymorph.core.observation import Observation
from baymorph.pipeline.config import load_resolved_config
from baymorph.pipeline.observation_metadata import observation_metadata_dict
from baymorph.pipeline.outputs import make_output_tree, pipeline_paths
from baymorph.pipeline.psf_candidates import load_psf_candidates
from baymorph.plotting.style import make_plot
from baymorph.simulation.psf import (
    EmpiricalPSF,
    GaussianPSF,
)


def save_empirical_psf(
    config: dict[str, Any],
    psf: np.ndarray,
) -> Path:
    """Save an empirical PSF image to disk.

    Parameters
    ----------
    config : dict
        Resolved pipeline configuration.
    psf : ndarray
        Empirical PSF image.

    Returns
    -------
    pathlib.Path
        Path to the saved FITS file.
    """
    make_output_tree(config)
    path = pipeline_paths(config)["psf"]["psf_fits"]

    overwrite = bool(config["outputs"].get("overwrite", False))

    fits.writeto(path, psf, overwrite=overwrite)

    return path


def load_empirical_psf(
    config: str | Path | dict[str, Any],
) -> EmpiricalPSF:
    """Load a saved empirical PSF from disk.

    Parameters
    ----------
    config : dict, str, or pathlib.Path
        Resolved configuration dictionary or path to a YAML configuration file.

    Returns
    -------
    EmpiricalPSF
        Empirical PSF object initialized from the saved FITS image.
    """
    if isinstance(config, (str, Path)):
        resolved = load_resolved_config(config)
    else:
        resolved = config

    path = pipeline_paths(resolved)["psf"]["psf_fits"]

    return EmpiricalPSF(
        fits.getdata(path)
    )


def estimate_psf_equivalent_gaussian_fwhm(
    psf: np.ndarray,
) -> float | None:
    """Estimate the moment-based Gaussian-equivalent PSF FWHM.

    The estimate is derived from the intensity-weighted second moments
    around the PSF centroid. It avoids the integer-pixel quantization of
    a direct half-maximum threshold and recovers the nominal width of a
    sampled Gaussian PSF.

    For a non-Gaussian empirical PSF, the returned value is a
    Gaussian-equivalent width rather than an exact geometric
    half-maximum diameter.

    Parameters
    ----------
    psf : numpy.ndarray
        Two-dimensional PSF image.

    Returns
    -------
    float or None
        Gaussian-equivalent FWHM in pixels, or ``None`` when the
        estimate cannot be computed.
    """
    data = np.asarray(psf, dtype=float)

    if data.ndim != 2 or not np.all(np.isfinite(data)):
        return None

    weights = np.clip(
        data,
        a_min=0.0,
        a_max=None,
    )
    total = float(weights.sum())

    if not np.isfinite(total) or total <= 0.0:
        return None

    yy, xx = np.indices(weights.shape, dtype=float)

    x_centroid = float(np.sum(weights * xx) / total)
    y_centroid = float(np.sum(weights * yy) / total)

    variance_x = float(
        np.sum(weights * (xx - x_centroid) ** 2) / total
    )
    variance_y = float(
        np.sum(weights * (yy - y_centroid) ** 2) / total
    )

    if (
        not np.isfinite(variance_x)
        or not np.isfinite(variance_y)
        or variance_x <= 0.0
        or variance_y <= 0.0
    ):
        return None

    sigma_equivalent = (variance_x * variance_y) ** 0.25
    gaussian_fwhm_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))

    return float(
        gaussian_fwhm_factor * sigma_equivalent
    )



def estimate_psf_core_fwhm(
    psf: np.ndarray,
    *,
    radial_step: float = 0.05,
    n_angles: int = 360,
) -> float | None:
    """Estimate the circularized half-maximum core FWHM of a PSF.

    The PSF is sampled along uniformly spaced radial directions using
    cubic interpolation. The azimuthal median profile is then used to
    locate the half-maximum radius at sub-pixel resolution.

    This quantity characterizes the PSF core and is intentionally
    distinct from the second-moment Gaussian-equivalent FWHM, which is
    more sensitive to extended wings.

    Parameters
    ----------
    psf : numpy.ndarray
        Two-dimensional PSF image.
    radial_step : float, default=0.05
        Radial sampling interval in pixels.
    n_angles : int, default=360
        Number of uniformly spaced radial directions.

    Returns
    -------
    float or None
        Circularized core FWHM in pixels, or ``None`` when the
        half-maximum crossing cannot be measured.

    Raises
    ------
    ValueError
        If ``radial_step`` is not positive or ``n_angles`` is too small.
    """
    from scipy.ndimage import map_coordinates

    if radial_step <= 0.0:
        raise ValueError(
            "radial_step must be positive."
        )

    if n_angles < 8:
        raise ValueError(
            "n_angles must be at least 8."
        )

    data = np.asarray(
        psf,
        dtype=float,
    )

    if (
        data.ndim != 2
        or not np.all(np.isfinite(data))
    ):
        return None

    peak = float(
        np.max(data)
    )

    if (
        not np.isfinite(peak)
        or peak <= 0.0
    ):
        return None

    y0, x0 = np.unravel_index(
        np.argmax(data),
        data.shape,
    )

    max_radius = float(
        min(
            x0,
            y0,
            data.shape[1] - 1 - x0,
            data.shape[0] - 1 - y0,
        )
    )

    if max_radius <= radial_step:
        return None

    radii = np.arange(
        0.0,
        max_radius + 0.5 * radial_step,
        radial_step,
    )

    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        n_angles,
        endpoint=False,
    )

    radial_grid = radii[
        None,
        :
    ]

    y_coordinates = (
        float(y0)
        + np.sin(angles)[:, None]
        * radial_grid
    )

    x_coordinates = (
        float(x0)
        + np.cos(angles)[:, None]
        * radial_grid
    )

    samples = map_coordinates(
        data,
        [
            y_coordinates,
            x_coordinates,
        ],
        order=3,
        mode="nearest",
        prefilter=True,
    )

    profile = np.median(
        samples,
        axis=0,
    )

    peak_profile = float(
        profile[0]
    )

    if (
        not np.isfinite(peak_profile)
        or peak_profile <= 0.0
    ):
        return None

    half_maximum = 0.5 * peak_profile

    crossings = np.flatnonzero(
        profile <= half_maximum
    )

    if (
        crossings.size == 0
        or crossings[0] == 0
    ):
        return None

    index = int(
        crossings[0]
    )

    half_maximum_radius = float(
        np.interp(
            half_maximum,
            [
                profile[index],
                profile[index - 1],
            ],
            [
                radii[index],
                radii[index - 1],
            ],
        )
    )

    return 2.0 * half_maximum_radius

def psf_radial_profile(
    psf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a median radial profile of a PSF image.

    Parameters
    ----------
    psf : ndarray
        Two-dimensional PSF image.

    Returns
    -------
    radius : ndarray
        Radial-bin centers in pixels.
    profile : ndarray
        Median PSF value in each radial bin.
    """
    data = np.asarray(psf, dtype=float)

    yy, xx = np.indices(data.shape)

    y0, x0 = np.unravel_index(
        np.nanargmax(data),
        data.shape,
    )

    radius = np.sqrt(
        (xx - x0) ** 2
        + (yy - y0) ** 2
    )

    max_radius = 0.5 * min(data.shape)

    radii = np.arange(
        0.0,
        max_radius + 1.0,
        1.0,
    )

    radius_mid = 0.5 * (
        radii[:-1]
        + radii[1:]
    )

    profile = np.full(
        radius_mid.shape,
        np.nan,
        dtype=float,
    )

    valid = np.isfinite(data)

    for i in range(radii.size - 1):
        annulus = (
            (radius >= radii[i])
            & (radius < radii[i + 1])
            & valid
        )

        if np.any(annulus):
            profile[i] = np.nanmedian(
                data[annulus]
            )

    return radius_mid, profile



def _psf_grid_shape(n_items: int) -> tuple[int, int]:
    """Return an adaptive grid shape for PSF cutout diagnostics."""
    if n_items <= 0:
        return 1, 1

    ncols = int(np.ceil(np.sqrt(n_items)))
    nrows = int(np.ceil(n_items / ncols))

    return nrows, ncols


def save_psf_diagnostics(
    config: dict[str, Any],
    psf: np.ndarray,
    cutouts: list[np.ndarray] | None = None,
    *,
    title: str = "PSF",
    nominal_fwhm: float | None = None,
) -> None:
    """Save PSF image, radial-profile, and optional cutout diagnostics.

    Parameters
    ----------
    config : dict
        Resolved pipeline configuration.
    psf : ndarray
        Final normalized PSF image.
    cutouts : list of ndarray, optional
        Individual source cutouts used to construct an empirical PSF.
    title : str, default="PSF"
        Title displayed above the PSF image.
    nominal_fwhm : float, optional
        Nominal PSF FWHM in pixels. When provided, it is included in the
        radial-profile panel.

    Returns
    -------
    None
        Diagnostic figures are written to the configured PSF directory.
    """
    paths = pipeline_paths(config)["psf"]

    diagnostics_dir = (
        paths["dir"]
        / "diagnostics"
    )

    diagnostics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    psf = np.asarray(
        psf,
        dtype=float,
    )

    core_fwhm = estimate_psf_core_fwhm(
        psf,
    )

    equivalent_gaussian_fwhm = (
        estimate_psf_equivalent_gaussian_fwhm(
            psf,
        )
    )

    fig, axes = make_plot(
        nrows=1,
        ncols=2,
        constrained_layout=True,
    )

    axes[0].imshow(
        psf,
        origin="lower",
        cmap="gist_stern",
    )

    axes[0].set_title(
        title,
    )

    axes[0].set_xlabel(
        r"$x\ [{\rm pix}]$",
    )

    axes[0].set_ylabel(
        r"$y\ [{\rm pix}]$",
    )

    radius, profile = psf_radial_profile(
        psf,
    )

    axes[1].plot(
        radius,
        profile,
        color="black",
        lw=3,
        label="Measured profile",
    )

    if core_fwhm is not None:
        axes[1].axvline(
            0.5 * core_fwhm,
            color="darkorange",
            linestyle="-.",
            lw=3,
            label=(
                r"$\mathrm{FWHM}_{\mathrm{core}}"
                rf" = {core_fwhm:.2f}\ \mathrm{{pix}}$"
            ),
        )

    if equivalent_gaussian_fwhm is not None:
        axes[1].axvline(
            0.5 * equivalent_gaussian_fwhm,
            color="mediumvioletred",
            linestyle="--",
            lw=3,
            label=(
                r"$\mathrm{FWHM}_{\mathrm{equiv.\ Gaussian}}"
                rf" = {equivalent_gaussian_fwhm:.2f}\ \mathrm{{pix}}$"
            ),
        )

    if nominal_fwhm is not None:
        axes[1].axvline(
            0.5 * nominal_fwhm,
            color="royalblue",
            linestyle=":",
            lw=3,
            label=(
                r"$\mathrm{FWHM}_{\mathrm{nominal\ Gaussian}}"
                rf" = {nominal_fwhm:.2f}\ \mathrm{{pix}}$"
            ),
        )

    axes[1].set_xlabel(
        r"$R\ [{\rm pix}]$",
    )

    axes[1].set_ylabel(
        r"$I_{\rm PSF}$",
    )

    axes[1].set_title(
        "PSF radial profile",
    )

    axes[1].legend(
        frameon=False,
    )

    fig.savefig(
        diagnostics_dir
        / "empirical_psf_diagnostic.pdf",
        bbox_inches="tight",
    )

    plt.close(
        fig,
    )

    if not cutouts:
        return

    nrows, ncols = _psf_grid_shape(
        len(cutouts)
    )

    fig, axes = make_plot(
        nrows=nrows,
        ncols=ncols,
        flatten=False,
        squeeze=False,
        constrained_layout=True,
    )

    axes_flat = np.ravel(
        axes
    )

    for ax in axes_flat:
        ax.axis("off")

    for index, cutout in enumerate(
        cutouts
    ):
        ax = axes_flat[
            index
        ]

        ax.axis("on")

        ax.imshow(
            cutout,
            origin="lower",
            cmap="gist_stern",
        )

        ax.set_title(
            rf"$\mathrm{{PSF\ cutout}}\ {index + 1}$",
        )

    fig.savefig(
        diagnostics_dir
        / "empirical_psf_cutouts.pdf",
        bbox_inches="tight",
    )

    plt.close(
        fig,
    )

def save_empirical_psf_diagnostics(
    config: dict[str, Any],
    psf: np.ndarray,
    cutouts: list[np.ndarray],
) -> None:
    """Save empirical PSF diagnostics."""
    save_psf_diagnostics(
        config,
        psf,
        cutouts,
        title="Empirical PSF",
    )


def save_simulated_psf_products(
    config: dict[str, Any],
    *,
    fwhm: float,
    size: int = 25,
) -> np.ndarray:
    """Create and save products for a simulated Gaussian PSF.

    Parameters
    ----------
    config : dict
        Resolved pipeline configuration.
    fwhm : float
        Gaussian PSF full width at half maximum in pixels.
    size : int, default=25
        Width and height of the square PSF kernel.

    Returns
    -------
    ndarray
        Unit-normalized Gaussian PSF kernel.
    """
    psf_model = GaussianPSF(
        fwhm=float(fwhm),
    )

    kernel = psf_model.kernel(
        size=int(size),
    )

    save_empirical_psf(
        config,
        kernel,
    )

    save_psf_diagnostics(
        config,
        kernel,
        cutouts=None,
        title="Simulated Gaussian PSF",
        nominal_fwhm=float(fwhm),
    )

    save_empirical_psf_metadata(
        config,
        {
            "type": "gaussian",
            "source": "simulation",
            "fwhm_pix": float(fwhm),
            "psf_core_fwhm_pix": estimate_psf_core_fwhm(
                kernel
            ),
            "psf_equivalent_gaussian_fwhm_pix": (
                estimate_psf_equivalent_gaussian_fwhm(
                    kernel
                )
            ),
            "kernel_size": int(size),
            "psf_shape": list(kernel.shape),
            "psf_sum": float(np.sum(kernel)),
            "psf_peak": float(np.max(kernel)),
        },
    )

    return kernel


def run_psf_construction_from_config(
    config: str | Path | dict[str, Any],
) -> np.ndarray:
    """Construct an empirical PSF from selected candidates.

    Candidates are validated using a local robust peak signal-to-noise
    estimate before recentering and stacking.

    Parameters
    ----------
    config : dict, str, or pathlib.Path
        Resolved configuration dictionary or path to a YAML configuration file.

    Returns
    -------
    ndarray
        Normalized empirical PSF image.

    Raises
    ------
    ValueError
        If no candidates are available, if any candidate is missing its
        cutout half-size, or if no candidate satisfies the configured
        signal-to-noise threshold.
    """
    if isinstance(config, (str, Path)):
        resolved = load_resolved_config(config)
    else:
        resolved = config

    observation = Observation.from_fits(
        resolved["data"]["image_path"]
    )
    image = observation.image

    candidates = load_psf_candidates(resolved)

    if len(candidates) == 0:
        raise ValueError("No PSF candidates available.")

    candidate_cfg = (
        resolved.get(
            "psf",
            {},
        )
        .get(
            "empirical",
            {},
        )
        .get(
            "candidate_selection",
            {},
        )
    )

    min_peak_snr = float(
        candidate_cfg.get(
            "min_peak_snr",
            30.0,
        )
    )

    if (
        not np.isfinite(min_peak_snr)
        or min_peak_snr < 0.0
    ):
        raise ValueError(
            "min_peak_snr must be finite and non-negative."
        )

    cutouts = []
    used_candidates = []
    rejected_candidates = []

    for candidate in candidates:
        if candidate.half_size is None:
            raise ValueError(
                "PSF candidate is missing half_size."
            )

        cutout = extract_centered_cutout(
            image.data,
            x=candidate.x,
            y=candidate.y,
            half_size=candidate.half_size,
        )

        peak_snr = estimate_cutout_peak_snr(
            cutout
        )

        candidate_metadata = {
            "label": candidate.label,
            "x": candidate.x,
            "y": candidate.y,
            "half_size": candidate.half_size,
            "peak_snr": peak_snr,
        }

        if (
            peak_snr is None
            or peak_snr < min_peak_snr
        ):
            rejected_candidates.append(
                {
                    **candidate_metadata,
                    "reason": (
                        "peak_snr_unavailable"
                        if peak_snr is None
                        else "peak_snr_below_threshold"
                    ),
                }
            )
            continue

        cutout = subtract_outer_background(
            cutout
        )
        cutout = recenter_cutout(
            cutout
        )

        cutouts.append(
            cutout
        )
        used_candidates.append(
            candidate_metadata
        )

    if len(cutouts) == 0:
        raise ValueError(
            "No PSF candidates satisfy "
            f"min_peak_snr={min_peak_snr:g}."
        )

    psf = stack_psf_cutouts(
        cutouts
    )

    psf_core_fwhm = estimate_psf_core_fwhm(
        psf,
    )

    psf_equivalent_gaussian_fwhm = (
        estimate_psf_equivalent_gaussian_fwhm(
            psf,
        )
    )

    save_empirical_psf_diagnostics(
        resolved,
        psf,
        cutouts,
    )

    save_empirical_psf(
        resolved,
        psf,
    )

    save_empirical_psf_metadata(
        resolved,
        {
            "observation": observation_metadata_dict(
                observation
            ),
            "n_candidates_input": len(candidates),
            "n_candidates_used": len(cutouts),
            "n_candidates_rejected_snr": len(
                rejected_candidates
            ),
            "min_peak_snr": min_peak_snr,
            "psf_shape": list(psf.shape),
            "psf_sum": float(np.sum(psf)),
            "psf_peak": float(np.nanmax(psf)),
            "psf_min": float(np.nanmin(psf)),
            "psf_max": float(np.nanmax(psf)),
            "psf_core_fwhm_pix": psf_core_fwhm,
            "psf_equivalent_gaussian_fwhm_pix": (
                psf_equivalent_gaussian_fwhm
            ),
            "used_candidates": used_candidates,
            "rejected_candidates": rejected_candidates,
            "warnings": _empirical_psf_warnings(
                psf,
                n_candidates_used=len(cutouts),
                psf_equivalent_gaussian_fwhm_pix=(
                    psf_equivalent_gaussian_fwhm
                ),
            ),
        },
    )

    return psf

def normalize_psf(psf: np.ndarray) -> np.ndarray:
    """Normalize a PSF image to unit sum.

    Parameters
    ----------
    psf : ndarray
        Input PSF image.

    Returns
    -------
    ndarray
        Unit-normalized PSF image.

    Raises
    ------
    ValueError
        If the PSF sum is not finite or is not strictly positive.
    """
    total = np.sum(psf)

    if not np.isfinite(total) or total <= 0:
        raise ValueError("PSF sum must be positive and finite.")

    return psf / total


def extract_centered_cutout(
    image: np.ndarray,
    x: float,
    y: float,
    half_size: int,
) -> np.ndarray:
    """Extract a square cutout centered on a source position.

    Parameters
    ----------
    image : ndarray
        Parent image.
    x, y : float
        Source position in image pixels.
    half_size : int
        Half-size of the requested cutout.

    Returns
    -------
    ndarray
        Extracted square cutout.

    Raises
    ------
    ValueError
        If the requested cutout extends outside the image boundaries.
    """
    x0 = int(round(x))
    y0 = int(round(y))

    y_min = y0 - half_size
    y_max = y0 + half_size + 1
    x_min = x0 - half_size
    x_max = x0 + half_size + 1

    if x_min < 0 or y_min < 0 or x_max > image.shape[1] or y_max > image.shape[0]:
        raise ValueError("Requested PSF cutout extends outside the image boundaries.")

    return image[y_min:y_max, x_min:x_max]


def _outer_border_pixels(
    cutout: np.ndarray,
    *,
    border_width: int,
) -> np.ndarray:
    """Return pixels in the outer border of a two-dimensional cutout."""
    data = np.asarray(
        cutout,
        dtype=float,
    )

    if data.ndim != 2:
        raise ValueError(
            "PSF cutout must be 2D."
        )

    if border_width <= 0:
        raise ValueError(
            "border_width must be positive."
        )

    if 2 * border_width >= min(
        data.shape
    ):
        raise ValueError(
            "border_width is too large for this cutout."
        )

    return np.concatenate(
        [
            data[:border_width, :].ravel(),
            data[-border_width:, :].ravel(),
            data[:, :border_width].ravel(),
            data[:, -border_width:].ravel(),
        ]
    )


def estimate_cutout_peak_snr(
    cutout: np.ndarray,
    *,
    border_width: int = 3,
) -> float | None:
    """Estimate robust local peak signal-to-noise for a PSF candidate.

    The local background and noise are estimated from the outer cutout
    border. Noise is represented by the Gaussian-equivalent MAD scale.

    Parameters
    ----------
    cutout : ndarray
        Candidate image cutout before background subtraction.
    border_width : int, default=3
        Width of the outer region used for background and noise estimation.

    Returns
    -------
    float or None
        Peak signal-to-noise ratio, or ``None`` when a finite positive
        local noise scale cannot be measured.
    """
    data = np.asarray(
        cutout,
        dtype=float,
    )

    border = _outer_border_pixels(
        data,
        border_width=border_width,
    )

    finite_border = border[
        np.isfinite(
            border
        )
    ]

    if finite_border.size == 0:
        return None

    background = float(
        np.median(
            finite_border
        )
    )

    mad = float(
        np.median(
            np.abs(
                finite_border
                - background
            )
        )
    )

    local_sigma = 1.4826 * mad

    if (
        not np.isfinite(local_sigma)
        or local_sigma <= 0.0
    ):
        return None

    finite_data = data[
        np.isfinite(
            data
        )
    ]

    if finite_data.size == 0:
        return None

    peak_signal = float(
        np.max(
            finite_data
        )
        - background
    )

    if (
        not np.isfinite(peak_signal)
        or peak_signal <= 0.0
    ):
        return None

    return float(
        peak_signal
        / local_sigma
    )


def subtract_outer_background(
    cutout: np.ndarray,
    border_width: int = 3,
) -> np.ndarray:
    """Subtract a border-based background estimate from a PSF cutout.

    Parameters
    ----------
    cutout : ndarray
        Input PSF-star cutout.
    border_width : int, default=3
        Width of the border region used to estimate the local background.

    Returns
    -------
    ndarray
        Background-subtracted cutout.

    Raises
    ------
    ValueError
        If ``border_width`` is not positive or is too large for the cutout.
    """
    border_pixels = _outer_border_pixels(
        cutout,
        border_width=border_width,
    )

    background = np.nanmedian(
        border_pixels
    )

    return cutout - background


def clean_empirical_psf(
    psf: np.ndarray,
    *,
    background_radius_fraction: float = 0.65,
    taper_start_fraction: float = 0.55,
) -> np.ndarray:
    """Clean and taper an empirical PSF image.

    Parameters
    ----------
    psf : ndarray
        Input empirical PSF image.
    background_radius_fraction : float, default=0.65
        Fraction of the maximum radius beyond which pixels are used to
        estimate the residual background.
    taper_start_fraction : float, default=0.55
        Fraction of the maximum radius where the cosine taper begins.

    Returns
    -------
    ndarray
        Cleaned and unit-normalized empirical PSF.

    Raises
    ------
    ValueError
        If ``psf`` is not two-dimensional.
    """
    data = np.asarray(
        psf,
        dtype=float,
    ).copy()

    if data.ndim != 2:
        raise ValueError(
            "PSF must be 2D."
        )

    if not np.all(
        np.isfinite(data)
    ):
        data = np.nan_to_num(
            data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    y0, x0 = np.unravel_index(
        np.nanargmax(data),
        data.shape,
    )

    yy, xx = np.indices(
        data.shape,
    )

    radius = np.sqrt(
        (xx - x0) ** 2
        + (yy - y0) ** 2
    )

    max_radius = 0.5 * min(
        data.shape
    )

    background_mask = (
        radius >= background_radius_fraction * max_radius
    )

    if np.any(background_mask):
        background = np.nanmedian(
            data[background_mask]
        )
    else:
        background = 0.0

    data = data - background
    data[data < 0.0] = 0.0

    taper_start = taper_start_fraction * max_radius
    taper_end = max_radius

    taper = np.ones_like(
        data,
        dtype=float,
    )

    outer = radius > taper_start

    taper[outer] = 0.5 * (
        1.0
        + np.cos(
            np.pi
            * (radius[outer] - taper_start)
            / (taper_end - taper_start)
        )
    )

    taper[radius >= taper_end] = 0.0

    data = data * taper

    return normalize_psf(
        data
    )


def stack_psf_cutouts(cutouts: list[np.ndarray]) -> np.ndarray:
    """Stack PSF-star cutouts into a cleaned empirical PSF.

    Parameters
    ----------
    cutouts : list of ndarray
        Individual background-subtracted, recentered PSF-star cutouts.

    Returns
    -------
    ndarray
        Cleaned and unit-normalized empirical PSF.

    Raises
    ------
    ValueError
        If no cutouts are provided.
    """
    if len(cutouts) == 0:
        raise ValueError("At least one PSF cutout is required.")

    half_sizes = [
        min(cutout.shape) // 2
        for cutout in cutouts
    ]

    common_half_size = int(
        np.min(half_sizes)
    )

    cropped_cutouts = [
        crop_center(
            cutout,
            common_half_size,
        )
        for cutout in cutouts
    ]

    stack = np.nanmedian(
        np.stack(
            cropped_cutouts,
            axis=0,
        ),
        axis=0,
    )

    psf = normalize_psf(
        stack
    )

    psf = clean_empirical_psf(
        psf,
    )

    return psf


def _empirical_psf_warnings(
    psf: np.ndarray,
    *,
    n_candidates_used: int,
    psf_equivalent_gaussian_fwhm_pix: float | None,
) -> list[str]:
    """Build quality-control warnings for an empirical PSF."""
    warnings = []

    data = np.asarray(
        psf,
        dtype=float,
    )

    if n_candidates_used < 5:
        warnings.append(
            "Empirical PSF is based on fewer than 5 stars and is likely unstable."
        )

    elif n_candidates_used < 10:
        warnings.append(
            "Empirical PSF is based on fewer than 10 stars; use with caution."
        )

    elif n_candidates_used < 20:
        warnings.append(
            "Empirical PSF is based on fewer than 20 stars; validation is recommended."
        )

    psf_sum = np.nansum(
        data,
    )

    if not np.isfinite(psf_sum) or not np.isclose(psf_sum, 1.0, rtol=1e-3):
        warnings.append(
            "PSF is not normalized to unit sum."
        )

    if not np.all(
        np.isfinite(data)
    ):
        warnings.append(
            "PSF contains non-finite values."
        )

    peak = np.nanmax(
        data,
    )

    if not np.isfinite(peak) or peak <= 0:
        warnings.append(
            "PSF peak is not positive."
        )

    if psf_equivalent_gaussian_fwhm_pix is None:
        warnings.append(
            "Could not estimate empirical PSF FWHM."
        )
    elif psf_equivalent_gaussian_fwhm_pix <= 0:
        warnings.append(
            "Empirical PSF FWHM is non-positive."
        )

    return warnings


def centroid_from_flux(cutout: np.ndarray) -> tuple[float, float]:
    """Compute a flux-weighted centroid for a cutout.

    Parameters
    ----------
    cutout : ndarray
        Input image cutout.

    Returns
    -------
    x_centroid, y_centroid : float
        Flux-weighted centroid coordinates.

    Raises
    ------
    ValueError
        If the shifted flux has non-positive total flux.
    """
    data = np.asarray(cutout, dtype=float)
    data = np.where(np.isfinite(data), data, 0.0)
    data = data - np.min(data)

    total = np.sum(data)
    if total <= 0:
        raise ValueError("Cannot compute centroid for non-positive flux cutout.")

    yy, xx = np.indices(data.shape)

    x_centroid = float(np.sum(xx * data) / total)
    y_centroid = float(np.sum(yy * data) / total)

    return x_centroid, y_centroid


def recenter_cutout(
    cutout: np.ndarray,
    *,
    target_x: float | None = None,
    target_y: float | None = None,
) -> np.ndarray:
    """Recenter a cutout using its flux-weighted centroid.

    Parameters
    ----------
    cutout : ndarray
        Input image cutout.
    target_x : float, optional
        Desired output x-coordinate of the centroid. If None, use the cutout
        center.
    target_y : float, optional
        Desired output y-coordinate of the centroid. If None, use the cutout
        center.

    Returns
    -------
    ndarray
        Shifted cutout.
    """
    from scipy.ndimage import shift

    y_size, x_size = cutout.shape

    if target_x is None:
        target_x = (x_size - 1) / 2

    if target_y is None:
        target_y = (y_size - 1) / 2

    x_centroid, y_centroid = centroid_from_flux(cutout)

    dx = target_x - x_centroid
    dy = target_y - y_centroid

    return shift(
        cutout,
        shift=(dy, dx),
        order=3,
        mode="nearest",
        prefilter=True,
    )


def save_empirical_psf_metadata(
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> Path:
    """Save empirical PSF metadata to disk.

    Parameters
    ----------
    config : dict
        Resolved pipeline configuration.
    metadata : dict
        JSON-serializable PSF metadata.

    Returns
    -------
    pathlib.Path
        Path to the saved metadata file.
    """
    make_output_tree(config)
    path = pipeline_paths(config)["psf"]["metadata"]

    with path.open("w") as f:
        json.dump(metadata, f, indent=2)

    return path


def estimate_psf_half_size(
    candidates,
    *,
    fallback_half_size: int = 25,
) -> int:
    """Estimate a representative PSF half-size from candidates.

    Parameters
    ----------
    candidates : iterable
        PSF candidates with optional ``half_size`` attributes.
    fallback_half_size : int, default=25
        Value returned when no candidates contain a half-size.

    Returns
    -------
    int
        Maximum candidate half-size, or the fallback value.
    """
    half_sizes = [
        candidate.half_size
        for candidate in candidates
        if candidate.half_size is not None
    ]

    if len(half_sizes) == 0:
        return fallback_half_size

    return int(np.max(half_sizes))


def crop_center(
    image: np.ndarray,
    half_size: int,
) -> np.ndarray:
    """Crop the central square region of an image.

    Parameters
    ----------
    image : ndarray
        Input image.
    half_size : int
        Half-size of the requested central crop.

    Returns
    -------
    ndarray
        Central square crop.

    Raises
    ------
    ValueError
        If ``half_size`` is not positive or the requested crop exceeds the
        image boundaries.
    """
    if half_size <= 0:
        raise ValueError("half_size must be positive.")

    y_size, x_size = image.shape
    x0 = x_size // 2
    y0 = y_size // 2

    y_min = y0 - half_size
    y_max = y0 + half_size + 1
    x_min = x0 - half_size
    x_max = x0 + half_size + 1

    if x_min < 0 or y_min < 0 or x_max > x_size or y_max > y_size:
        raise ValueError("Requested center crop extends outside the image boundaries.")

    return image[y_min:y_max, x_min:x_max]
