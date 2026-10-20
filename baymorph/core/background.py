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


def _fit_plane(
    coordinates: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares fit a plane and return its coefficients and design matrix.

    Parameters
    ----------
    coordinates : ndarray
        Array of shape ``(n_pixels, 2)`` holding centred ``x`` and ``y``.
    values : ndarray
        Pixel values.

    Returns
    -------
    coefficients : ndarray
        Constant term followed by the ``x`` and ``y`` gradients.
    design : ndarray
        Design matrix used for the fit.
    """
    design = np.column_stack(
        [
            np.ones(
                values.size
            ),
            coordinates[:, 0],
            coordinates[:, 1],
        ]
    )

    coefficients, *_ = np.linalg.lstsq(
        design,
        values,
        rcond=None,
    )

    return coefficients, design


def _gradient_diagnostic(
    coefficients: np.ndarray,
    design: np.ndarray,
    sigma: float,
) -> dict[str, float]:
    """Describe a fitted sky plane and how significant its gradient is.

    The diagnostic reports; it never selects a background model. Choosing
    between a constant and a plane remains an explicit user decision.
    """
    try:
        covariance = np.linalg.inv(
            design.T @ design
        ) * sigma**2
        errors = np.sqrt(
            np.diag(covariance)
        )
    except np.linalg.LinAlgError:
        errors = np.full(
            coefficients.size,
            np.nan,
        )

    def significance(
        index: int,
    ) -> float:
        if not np.isfinite(errors[index]) or errors[index] <= 0.0:
            return float("nan")

        return float(
            abs(
                coefficients[index]
            )
            / errors[index]
        )

    return {
        "gradient_x": float(
            coefficients[1]
        ),
        "gradient_y": float(
            coefficients[2]
        ),
        "gradient_x_error": float(
            errors[1]
        ),
        "gradient_y_error": float(
            errors[2]
        ),
        "gradient_x_significance": significance(
            1
        ),
        "gradient_y_significance": significance(
            2
        ),
    }


def _detrended_background(
    image: np.ndarray,
    mask: np.ndarray | None,
    sigma_clip: float,
    maxiters: int,
    noise_method: str,
) -> BackgroundResult:
    """Estimate a constant level and a gradient-insensitive noise scale."""
    if image.ndim != 2:
        raise ValueError(
            "detrend requires a two-dimensional image, because fitting a "
            "plane needs pixel coordinates."
        )

    rows, columns = np.indices(
        image.shape,
        dtype=float,
    )
    # Centred coordinates keep the constant term interpretable as the value at
    # the field centre and keep the design matrix well conditioned.
    columns -= (image.shape[1] - 1) / 2.0
    rows -= (image.shape[0] - 1) / 2.0

    valid = np.isfinite(image)

    if mask is not None:
        valid &= ~np.asarray(
            mask,
            dtype=bool,
        )

    values = image[valid]
    coordinates = np.column_stack(
        [
            columns[valid],
            rows[valid],
        ]
    )

    if values.size < 4:
        raise ValueError(
            "Input data contains too few finite values to fit a plane."
        )

    threshold = np.inf
    coefficients, design = _fit_plane(
        coordinates,
        values,
    )

    for _ in range(maxiters):
        residuals = values - design @ coefficients
        scatter = float(
            np.std(residuals)
        )

        if scatter == 0.0:
            break

        keep = np.abs(residuals) < sigma_clip * scatter

        if keep.sum() == values.size:
            break

        if keep.sum() < 4:
            break

        threshold = float(
            sigma_clip * scatter
        )
        values = values[keep]
        coordinates = coordinates[keep]
        coefficients, design = _fit_plane(
            coordinates,
            values,
        )

    residuals = values - design @ coefficients

    if noise_method == "corrected_std":
        # Residuals are centred on the plane by construction, so the clipping
        # interval that must be inverted is symmetric about zero.
        sigma = truncated_gaussian_sigma(
            residuals,
            threshold,
            center=0.0,
        )
    else:
        sigma = estimate_noise(
            residuals,
            method=noise_method,
        )

    return BackgroundResult(
        background_level=float(
            np.median(values)
        ),
        sigma=sigma,
        mode="constant",
        parameters=_gradient_diagnostic(
            coefficients,
            design,
            sigma,
        ),
    )


def estimate_background_constant(
    data: np.ndarray,
    mask: np.ndarray | None = None,
    sigma_clip: float = 3.0,
    maxiters: int = 5,
    noise_method: str = "std",
    detrend: bool = False,
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
    detrend : bool, default=False
        If True, measure the noise scale from residuals about a fitted plane
        instead of about a constant, and report the fitted gradient in
        ``parameters``. Requires a two-dimensional image.

    Returns
    -------
    BackgroundResult
        Estimated scalar background level and RMS noise. With ``detrend``, the
        gradient and its significance are reported in ``parameters``.

    Raises
    ------
    ValueError
        If the input image contains no finite values, or if ``detrend`` is
        requested for input that is not two-dimensional.

    Notes
    -----
    ``"mad"`` is highly robust but has roughly 37% efficiency, so its estimate
    carries about 1.5 times the random scatter of a standard deviation at the
    same sample size. Where that scatter propagates into a likelihood
    normalization, ``"corrected_std"`` is preferable: clipping still removes
    outliers, and the truncation correction restores both unbiasedness and
    efficiency. The correction assumes the surviving core is Gaussian.

    Without ``detrend``, any unmodelled sky structure is counted as noise: an
    unmodelled linear ramp of plus or minus one count on a scale of two
    inflates the estimate by about four percent. With ``detrend``, the scale
    measures noise alone, and an unmodelled gradient instead surfaces as an
    elevated reduced chi-squared during fitting. That is deliberate: the
    deficiency becomes visible rather than being absorbed into the noise term.
    The remedy is to model it, through ``fit.background.model`` set to
    ``plane``.

    The returned ``background_level`` is the median of the surviving values in
    both cases. Detrending changes how the scale is measured, not what is
    subtracted.
    """
    image = np.asarray(data, dtype=float)

    if detrend:
        return _detrended_background(
            image,
            mask,
            sigma_clip,
            maxiters,
            noise_method,
        )

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
    no_data: np.ndarray | None = None,
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
    no_data : ndarray of bool, optional
        Pixels without data (AD-053), passed to Photutils as the coverage
        mask: they take no part in the estimate, their values may be
        non-finite, and the maps hold 0 there. The scalar level and scale
        are the medians of the maps over the other pixels.

    Returns
    -------
    BackgroundResult
        Background estimate containing both scalar summary values and the
        full background and RMS maps.
    """
    covered = None

    if no_data is not None and np.any(no_data):
        covered = ~np.asarray(no_data, dtype=bool)
    else:
        no_data = None

    bkg = Background2D(
        data,
        box_size=box_size,
        filter_size=(
            filter_size,
            filter_size,
        ),
        mask=mask,
        coverage_mask=no_data,
        exclude_percentile=exclude_percentile,
        bkg_estimator=MedianBackground(),
    )
    background = bkg.background if covered is None else bkg.background[covered]
    rms = bkg.background_rms if covered is None else bkg.background_rms[covered]

    return BackgroundResult(
        background_level=float(np.median(background)),
        sigma=float(np.median(rms)),
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
