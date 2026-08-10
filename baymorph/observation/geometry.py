"""Local affine transformations of elliptical morphology geometry."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatMatrix = NDArray[np.float64]


def ellipse_quadratic_form(
    *,
    re: float,
    q: float,
    theta: float,
) -> FloatMatrix:
    """Return the quadratic form of one elliptical morphology.

    Parameters
    ----------
    re : float
        Effective semi-major-axis radius.
    q : float
        Minor-to-major axis ratio.
    theta : float
        Major-axis angle in radians, measured from the positive x-axis
        toward the positive y-axis.

    Returns
    -------
    ndarray
        Symmetric positive-definite 2 x 2 quadratic-form matrix.
    """
    re = float(re)
    q = float(q)
    theta = float(theta)

    if not np.isfinite(re) or re <= 0.0:
        raise ValueError("re must be finite and positive.")

    if not np.isfinite(q) or q <= 0.0 or q > 1.0:
        raise ValueError(
            "q must be finite and satisfy 0 < q <= 1."
        )

    if not np.isfinite(theta):
        raise ValueError("theta must be finite.")

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    rotation = np.array(
        [
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta],
        ],
        dtype=float,
    )

    principal = np.diag(
        [
            1.0 / re**2,
            1.0 / (q * re) ** 2,
        ]
    )

    return np.asarray(
        rotation
        @ principal
        @ rotation.T,
        dtype=float,
    )


def transform_quadratic_form(
    quadratic_form: ArrayLike,
    jacobian: ArrayLike,
) -> FloatMatrix:
    """Transform an ellipse through one local sky-to-detector Jacobian.

    The Jacobian convention is

    ``delta_detector = J @ delta_source``.

    Parameters
    ----------
    quadratic_form : array-like
        Source-frame symmetric positive-definite 2 x 2 matrix.
    jacobian : array-like
        Invertible 2 x 2 source-to-detector Jacobian.

    Returns
    -------
    ndarray
        Detector-frame quadratic form.
    """
    source = _validate_positive_definite_matrix(
        quadratic_form,
        name="quadratic_form",
    )
    jacobian_array = _validate_matrix(
        jacobian,
        name="jacobian",
    )

    determinant = float(
        np.linalg.det(
            jacobian_array
        )
    )

    if not np.isfinite(determinant) or np.isclose(
        determinant,
        0.0,
    ):
        raise ValueError(
            "jacobian must be invertible."
        )

    inverse = np.linalg.inv(
        jacobian_array
    )

    transformed = (
        inverse.T
        @ source
        @ inverse
    )

    return _validate_positive_definite_matrix(
        transformed,
        name="transformed quadratic form",
    )


def quadratic_form_to_ellipse(
    quadratic_form: ArrayLike,
) -> tuple[float, float, float]:
    """Recover ``re``, ``q``, and ``theta`` from a quadratic form.

    The returned angle follows the BayMorph detector convention and is
    canonicalized to the interval ``[-pi / 2, pi / 2)``.

    Parameters
    ----------
    quadratic_form : array-like
        Symmetric positive-definite 2 x 2 quadratic-form matrix.

    Returns
    -------
    tuple
        Effective semi-major-axis radius, axis ratio, and major-axis angle.
    """
    matrix = _validate_positive_definite_matrix(
        quadratic_form,
        name="quadratic_form",
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        matrix
    )

    major_index = int(
        np.argmin(
            eigenvalues
        )
    )
    minor_index = 1 - major_index

    lambda_major = float(
        eigenvalues[
            major_index
        ]
    )
    lambda_minor = float(
        eigenvalues[
            minor_index
        ]
    )

    re = 1.0 / np.sqrt(
        lambda_major
    )
    q = np.sqrt(
        lambda_major
        / lambda_minor
    )

    major_axis = eigenvectors[
        :,
        major_index,
    ]

    theta = float(
        np.arctan2(
            major_axis[1],
            major_axis[0],
        )
    )

    theta = _canonicalize_axis_angle(
        theta
    )

    return (
        float(re),
        float(q),
        theta,
    )


def _canonicalize_axis_angle(
    theta: float,
) -> float:
    """Canonicalize an unoriented-axis angle modulo pi."""
    return float(
        (
            theta
            + 0.5 * np.pi
        )
        % np.pi
        - 0.5 * np.pi
    )


def _validate_matrix(
    matrix: ArrayLike,
    *,
    name: str,
) -> FloatMatrix:
    """Return one finite 2 x 2 matrix."""
    array = np.asarray(
        matrix,
        dtype=float,
    )

    if array.shape != (
        2,
        2,
    ):
        raise ValueError(
            f"{name} must have shape (2, 2)."
        )

    if not np.all(
        np.isfinite(
            array
        )
    ):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    return array


def _validate_positive_definite_matrix(
    matrix: ArrayLike,
    *,
    name: str,
) -> FloatMatrix:
    """Return one symmetric positive-definite 2 x 2 matrix."""
    array = _validate_matrix(
        matrix,
        name=name,
    )

    if not np.allclose(
        array,
        array.T,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError(
            f"{name} must be symmetric."
        )

    eigenvalues = np.linalg.eigvalsh(
        array
    )

    if np.any(
        eigenvalues <= 0.0
    ):
        raise ValueError(
            f"{name} must be positive definite."
        )

    return np.asarray(
        array,
        dtype=float,
    )


def local_wcs_jacobian(
    wcs,
    centre,
    *,
    step_arcsec: float = 0.01,
) -> FloatMatrix:
    """Return the local sky-to-detector Jacobian at one celestial position.

    The Jacobian convention is

    ``delta_pixel = J @ delta_sky``,

    where the sky-plane coordinates are local east and north offsets in
    arcseconds. The returned matrix therefore has units of pixels per
    arcsecond.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        Two-dimensional celestial WCS.
    centre : astropy.coordinates.SkyCoord
        Celestial position at which the local transformation is evaluated.
    step_arcsec : float, default=0.01
        Symmetric finite-difference step in arcseconds.

    Returns
    -------
    ndarray
        Local 2 x 2 sky-to-detector Jacobian. Columns correspond to east
        and north tangent-plane directions; rows correspond to detector
        x and y coordinates.

    Raises
    ------
    ValueError
        If the finite-difference step is invalid or the resulting Jacobian
        is non-finite or singular.
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord, SkyOffsetFrame
    from astropy.wcs import WCS

    if not isinstance(
        wcs,
        WCS,
    ):
        raise TypeError(
            "wcs must be an astropy.wcs.WCS instance."
        )

    if not isinstance(
        centre,
        SkyCoord,
    ):
        raise TypeError(
            "centre must be an astropy.coordinates.SkyCoord instance."
        )

    step_arcsec = float(
        step_arcsec
    )

    if (
        not np.isfinite(step_arcsec)
        or step_arcsec <= 0.0
    ):
        raise ValueError(
            "step_arcsec must be finite and positive."
        )

    offset_frame = SkyOffsetFrame(
        origin=centre
    )

    step = (
        step_arcsec
        * u.arcsec
    )

    east_plus = SkyCoord(
        lon=step,
        lat=0.0 * u.arcsec,
        frame=offset_frame,
    ).transform_to(
        centre.frame
    )
    east_minus = SkyCoord(
        lon=-step,
        lat=0.0 * u.arcsec,
        frame=offset_frame,
    ).transform_to(
        centre.frame
    )

    north_plus = SkyCoord(
        lon=0.0 * u.arcsec,
        lat=step,
        frame=offset_frame,
    ).transform_to(
        centre.frame
    )
    north_minus = SkyCoord(
        lon=0.0 * u.arcsec,
        lat=-step,
        frame=offset_frame,
    ).transform_to(
        centre.frame
    )

    x_east_plus, y_east_plus = (
        wcs.world_to_pixel(
            east_plus
        )
    )
    x_east_minus, y_east_minus = (
        wcs.world_to_pixel(
            east_minus
        )
    )

    x_north_plus, y_north_plus = (
        wcs.world_to_pixel(
            north_plus
        )
    )
    x_north_minus, y_north_minus = (
        wcs.world_to_pixel(
            north_minus
        )
    )

    denominator = (
        2.0
        * step_arcsec
    )

    jacobian = np.array(
        [
            [
                (
                    x_east_plus
                    - x_east_minus
                )
                / denominator,
                (
                    x_north_plus
                    - x_north_minus
                )
                / denominator,
            ],
            [
                (
                    y_east_plus
                    - y_east_minus
                )
                / denominator,
                (
                    y_north_plus
                    - y_north_minus
                )
                / denominator,
            ],
        ],
        dtype=float,
    )

    jacobian = _validate_matrix(
        jacobian,
        name="local WCS Jacobian",
    )

    determinant = float(
        np.linalg.det(
            jacobian
        )
    )

    if np.isclose(
        determinant,
        0.0,
    ):
        raise ValueError(
            "Local WCS Jacobian must be invertible."
        )

    return jacobian
