"""Sersic morphology model."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaincinv, gammaln

FloatArray = NDArray[np.float64]


@dataclass
class SersicModel:
    r"""Single-component Sérsic surface-brightness model.

    The elliptical radius is defined as

    .. math::

        R^2 = x'^2 + \left(\frac{y'}{q}\right)^2,

    and the surface-brightness profile is

    .. math::

        I(R) = I_e
        \exp\left\{
            -b_n
            \left[
                \left(\frac{R}{R_e}\right)^{1/n} - 1
            \right]
        \right\}.

    The normalization is computed analytically so that ``flux`` is the
    continuous total flux integrated over the full elliptical plane.

    Parameters
    ----------
    flux : float
        Total integrated flux over the full continuous profile.
    re : float
        Effective (half-light) radius, in pixels.
    n : float
        Sérsic index.
    q : float
        Projected axis ratio.
    theta : float
        Position angle, in radians.
    x0 : float
        X-coordinate of the model center, in pixels.
    y0 : float
        Y-coordinate of the model center, in pixels.
    """

    flux: float
    re: float
    n: float
    q: float
    theta: float
    x0: float
    y0: float

    def __post_init__(self) -> None:
        """Validate the physical model parameters."""
        if not np.isfinite(self.flux) or self.flux <= 0.0:
            raise ValueError("Sersic flux must be finite and positive.")

        if not np.isfinite(self.re) or self.re <= 0.0:
            raise ValueError(
                "Sersic effective radius must be finite and positive."
            )

        if not np.isfinite(self.n) or self.n <= 0.0:
            raise ValueError("Sersic index must be finite and positive.")

        if not np.isfinite(self.q) or self.q <= 0.0 or self.q > 1.0:
            raise ValueError(
                "Sersic axis ratio must be finite and satisfy 0 < q <= 1."
            )

        if not np.isfinite(self.theta):
            raise ValueError("Sersic position angle must be finite.")

        if not np.isfinite(self.x0) or not np.isfinite(self.y0):
            raise ValueError("Sersic center coordinates must be finite.")

    @property
    def bn(self) -> float:
        """Return the Sérsic constant enclosing half the total flux."""
        return float(gammaincinv(2.0 * self.n, 0.5))

    @property
    def log_central_intensity(self) -> float:
        r"""Return the logarithm of the central surface brightness.

        For the adopted elliptical-radius convention, the total flux is

        .. math::

            F =
            2 \pi q I_0 R_e^2 n
            b_n^{-2n}
            \Gamma(2n),

        where ``I_0`` is the central surface brightness.
        """
        bn = self.bn

        log_normalization = (
            np.log(2.0 * np.pi)
            + np.log(self.q)
            + 2.0 * np.log(self.re)
            + np.log(self.n)
            + gammaln(2.0 * self.n)
            - 2.0 * self.n * np.log(bn)
        )

        return float(np.log(self.flux) - log_normalization)

    def surface_brightness(
        self,
        radius: ArrayLike,
    ) -> FloatArray:
        """Evaluate the continuous Sérsic surface brightness.

        Parameters
        ----------
        radius : array-like
            Elliptical radius in pixels.

        Returns
        -------
        ndarray
            Surface brightness evaluated at each radius.

        Raises
        ------
        ValueError
            If any radius is negative or non-finite.
        """
        radius_array = np.asarray(radius, dtype=float)

        if np.any(~np.isfinite(radius_array)):
            raise ValueError("Sersic radii must be finite.")

        if np.any(radius_array < 0.0):
            raise ValueError("Sersic radii must be non-negative.")

        log_intensity = self.log_central_intensity - self.bn * (
            radius_array / self.re
        ) ** (1.0 / self.n)

        return np.asarray(np.exp(log_intensity), dtype=float)

    def _elliptical_radius(
        self,
        x: FloatArray,
        y: FloatArray,
    ) -> FloatArray:
        """Return elliptical radii for Cartesian coordinates."""
        dx = x - self.x0
        dy = y - self.y0

        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)

        x_rotated = dx * cos_theta + dy * sin_theta
        y_rotated = -dx * sin_theta + dy * cos_theta

        return np.sqrt(
            x_rotated**2 + (y_rotated / self.q) ** 2
        )

    @staticmethod
    def _validate_shape(shape: tuple[int, int]) -> None:
        """Validate a requested image shape."""
        if (
            len(shape) != 2
            or any(
                not isinstance(size, (int, np.integer)) or size <= 0
                for size in shape
            )
        ):
            raise ValueError(
                "Sersic render shape must contain two positive integers."
            )

    @staticmethod
    def _validate_integration_order(integration_order: int) -> None:
        """Validate the Gauss-Legendre integration order."""
        if (
            not isinstance(integration_order, (int, np.integer))
            or integration_order <= 0
        ):
            raise ValueError(
                "Pixel integration order must be a positive integer."
            )

    def render_components(
        self,
        shape: tuple[int, int],
        *,
        integration_order: int = 5,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> dict[str, FloatArray]:
        """Render the Sérsic model as a named single-component model."""
        return {
            "sersic": self.render(
                shape=shape,
                integration_order=integration_order,
                origin=origin,
            )
        }

    def render(
        self,
        shape: tuple[int, int],
        *,
        integration_order: int = 5,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> FloatArray:
        """Integrate the continuous model over square detector pixels.

        Each output value is the integral of the continuous Sérsic surface
        brightness over one unit-area pixel. Integration uses tensor-product
        Gauss-Legendre quadrature.

        The analytic profile is never renormalized on the requested cutout.
        Therefore, pixels shared by two cutouts retain identical values and
        the sum of a finite image represents only the flux captured by that
        footprint.

        Parameters
        ----------
        shape : tuple of int
            Output image shape as ``(ny, nx)``.
        integration_order : int, optional
            One-dimensional Gauss-Legendre quadrature order used along each
            pixel axis. The number of profile evaluations per pixel is the
            square of this value.
        origin : tuple of float, optional
            Physical ``(y, x)`` coordinates of output pixel ``[0, 0]``.

        Returns
        -------
        ndarray
            Flux integrated over every output pixel.
        """
        self._validate_shape(shape)
        self._validate_integration_order(integration_order)

        ny, nx = shape
        y_origin, x_origin = origin

        pixel_y, pixel_x = np.mgrid[0:ny, 0:nx]
        pixel_y = np.asarray(pixel_y, dtype=float) + y_origin
        pixel_x = np.asarray(pixel_x, dtype=float) + x_origin

        nodes, weights = np.polynomial.legendre.leggauss(
            integration_order
        )

        offsets = 0.5 * nodes
        pixel_weights = 0.5 * weights

        image = np.zeros(shape, dtype=float)

        for y_offset, y_weight in zip(
            offsets,
            pixel_weights,
            strict=True,
        ):
            for x_offset, x_weight in zip(
                offsets,
                pixel_weights,
                strict=True,
            ):
                radius = self._elliptical_radius(
                    x=np.asarray(pixel_x + x_offset, dtype=float),
                    y=np.asarray(pixel_y + y_offset, dtype=float),
                )

                image += (
                    x_weight
                    * y_weight
                    * self.surface_brightness(radius)
                )

        return image
