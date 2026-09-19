"""Background and noise estimation utilities."""

from dataclasses import dataclass

import numpy as np
from astropy.stats import mad_std
from photutils.background import Background2D, MedianBackground

from baymorph.stats.robust import truncated_gaussian_sigma


@dataclass(slots=True)
class BackgroundResult:
    """Container for background and noise estimation results.

    Parameters
    ----------
    background_level : float
        Representative scalar background level.

    sigma : float
        Estimated background RMS.

    mode : str, default="constant"
        Background model used.

    background_map : ndarray, optional
        Two-dimensional background model.

    rms_map : ndarray, optional
        Two-dimensional RMS map.

    residual : ndarray, optional
        Background-subtracted image.

    mask : ndarray of bool, optional
        Boolean mask used during the background estimation.

    parameters : dict, optional
        Parameters describing the fitted background model.
    """

    background_level: float
    sigma: float

    mode: str = "constant"

    background_map: np.ndarray | None = None
    rms_map: np.ndarray | None = None

    residual: np.ndarray | None = None
    mask: np.ndarray | None = None

    parameters: dict[str, float] | None = None

    @property
    def background(self) -> float:
        """Backward-compatible alias for ``background_level``."""
        return self.background_level

    @background.setter
    def background(self, value: float) -> None:
        self.background_level = value


def estimate_background_constant(
    data: np.ndarray,
    mask: np.ndarray | None = None,
    sigma_clip: float = 3.0,
    maxiters: int = 5,
    noise_method: str = "std",
) -> BackgroundResult:
    """Estimate a constant background and noise level using sigma clipping.

    Parameters
    ----------
    data : ndarray
        Input image.
    sigma_clip : float, default=3.0
        Sigma-clipping threshold applied during the iterative estimation.
    maxiters : int, default=5
        Maximum number of sigma-clipping iterations.
    noise_method : {"std", "mad", "corrected_std"}, default="std"
        Scale estimator applied to the surviving pixels. ``"corrected_std"``
        additionally inverts the truncation induced by the clipping, which
        removes the resulting low bias and is more efficient than ``"mad"``.

    Returns
    -------
    BackgroundResult
        Estimated scalar background level and RMS noise.

    Raises
    ------
    ValueError
        If the input image contains no finite values.

    Notes
    -----
    ``"mad"`` is highly robust but has roughly 37% efficiency, so its estimate
    carries about 1.5 times the random scatter of a standard deviation at the
    same sample size. Where that scatter propagates into a likelihood
    normalization, ``"corrected_std"`` is preferable: clipping still removes
    outliers, and the truncation correction restores both unbiasedness and
    efficiency. The correction assumes the surviving core is Gaussian.
    """
    image = np.asarray(data, dtype=float)

    if mask is not None:
        values = image[~mask]
    else:
        values = image.ravel()

    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError("Input data contains no finite values.")

    # Track the interval that actually truncated the sample. It stays infinite
    # when no pixel was ever rejected, so an unclipped sample needs no
    # truncation correction.
    threshold = np.inf
    center = float(np.median(values))

    for _ in range(maxiters):
        median = np.median(values)
        std = np.std(values)

        if std == 0:
            break

        keep = np.abs(values - median) < sigma_clip * std

        if keep.sum() == values.size:
            break

        threshold = float(sigma_clip * std)
        center = float(median)
        values = values[keep]

    if noise_method == "corrected_std":
        sigma = truncated_gaussian_sigma(
            values,
            threshold,
            center=center,
        )
    else:
        sigma = estimate_noise(
            values,
            method=noise_method,
        )

    return BackgroundResult(
        background_level=float(np.median(values)),
        sigma=sigma,
        mode="constant",
    )


def estimate_noise(
    data: np.ndarray,
    mask: np.ndarray | None = None,
    method: str = "mad",
) -> float:
    """Estimate the background noise from an image.

    Parameters
    ----------
    data : ndarray
        Input image.
    mask : ndarray of bool, optional
        Boolean mask identifying pixels to ignore.
    method : {"mad", "std"}, default="mad"
        Noise estimator.

    Returns
    -------
    float
        Estimated background RMS.
    """
    values = np.asarray(data, dtype=float)

    if mask is not None:
        values = values[~mask]

    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError("Input data contains no finite values.")

    if method == "mad":
        return float(mad_std(values))

    if method == "std":
        return float(np.std(values))

    raise ValueError(f"Unknown noise estimation method: {method!r}")


def estimate_background_photutils(
    data: np.ndarray,
    box_size: int = 20,
    filter_size: int = 3,
    mask: np.ndarray | None = None,
    exclude_percentile: float = 10.0,
) -> BackgroundResult:
    """Estimate a spatially varying background using Photutils.

    Parameters
    ----------
    data : ndarray
        Input image.
    box_size : int, default=20
        Size of the background mesh used by
        :class:`photutils.background.Background2D`.
    filter_size : int, default=3
        Size of the median filter applied to the background meshes.
    mask : ndarray of bool, optional
        Boolean mask identifying pixels to ignore during the background
        estimation.
    exclude_percentile : float, default=10.0
        Maximum fraction of masked pixels allowed within each background
        mesh.

    Returns
    -------
    BackgroundResult
        Background estimate containing both scalar summary values and the
        full background and RMS maps.
    """
    bkg = Background2D(
        data,
        box_size=box_size,
        filter_size=(
            filter_size,
            filter_size,
        ),
        mask=mask,
        exclude_percentile=exclude_percentile,
        bkg_estimator=MedianBackground(),
    )

    return BackgroundResult(
        background_level=float(np.median(bkg.background)),
        sigma=float(np.median(bkg.background_rms)),
        mode="mesh",
        background_map=bkg.background.copy(),
        rms_map=bkg.background_rms.copy(),
    )


def subtract_background(
    data: np.ndarray,
    background: float,
) -> np.ndarray:
    """Subtract a scalar background level from an image.

    Parameters
    ----------
    data : ndarray
        Input image.
    background : float
        Scalar background level to subtract.

    Returns
    -------
    ndarray
        Background-subtracted image.
    """
    return np.asarray(data, dtype=float) - background
