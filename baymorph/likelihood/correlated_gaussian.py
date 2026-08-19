"""Correlated Gaussian image likelihood."""

from __future__ import annotations

import numpy as np
from scipy.fft import irfftn, next_fast_len, rfftn
from scipy.sparse.linalg import LinearOperator, cg

from baymorph.likelihood.base import ImageLikelihood


class CorrelatedGaussianImageLikelihood(ImageLikelihood):
    """Gaussian likelihood with fixed spatial pixel correlation.

    The covariance is factorized as

    .. math::

        C = S R S,

    where ``S`` is a fixed diagonal field of marginal standard deviations and
    ``R`` is a fixed stationary two-dimensional correlation operator. The
    likelihood is evaluated on the exact set of unmasked pixels by applying
    the principal submatrix ``R[V, V]`` matrix-free through FFT convolution
    and solving the resulting linear system with conjugate gradients.

    Parameters
    ----------
    data : array-like
        Observed two-dimensional image.
    noise : array-like or float
        Fixed marginal Gaussian standard deviation. It may be a scalar or an
        array broadcastable to the observed image shape. All evaluated values
        must be strictly positive.
    correlation_kernel : array-like
        Odd-sized two-dimensional stationary correlation kernel centred on
        unit zero-lag correlation. The kernel must satisfy covariance symmetry
        ``R(dy, dx) = R(-dy, -dx)``.
    mask : array-like of bool, optional
        Boolean mask identifying pixels excluded from likelihood evaluation.
    include_normalization : bool, default=False
        If True, include the complete Gaussian normalization. This requires a
        fixed ``correlation_logdet`` for the exact masked correlation matrix.
        If False, return the log-likelihood up to a model-independent additive
        constant.
    correlation_logdet : float, optional
        Log determinant of the exact masked correlation matrix ``R[V, V]``.
        Required only when ``include_normalization=True``.
    cg_rtol : float, default=5e-7
        Relative tolerance for conjugate-gradient solves.
    cg_maxiter : int, optional
        Maximum number of conjugate-gradient iterations. If omitted, SciPy's
        default limit is used.

    Notes
    -----
    The correlation kernel and noise field are fixed observational
    calibration products. They are not morphology parameters or priors.

    Omitting the normalization does not change posterior inference when the
    data, mask, noise field, and correlation model are fixed. It does shift
    the absolute log evidence by an unknown additive constant and therefore
    must not be used for comparisons between different statistical datasets
    or different noise models.
    """

    def __init__(
        self,
        data: np.ndarray,
        noise: np.ndarray | float,
        correlation_kernel: np.ndarray,
        mask: np.ndarray | None = None,
        include_normalization: bool = False,
        correlation_logdet: float | None = None,
        cg_rtol: float = 5.0e-7,
        cg_maxiter: int | None = None,
    ) -> None:
        super().__init__(
            data=data,
            mask=mask,
        )

        if len(self.image_shape) != 2:
            raise ValueError(
                "Correlated Gaussian likelihood requires two-dimensional data."
            )

        self.noise_values = self.prepare_fixed_array(
            noise,
            name="Noise",
            require_positive=True,
        )

        self.correlation_kernel = self._validate_correlation_kernel(
            correlation_kernel
        )

        self.include_normalization = bool(
            include_normalization
        )

        if not np.isfinite(cg_rtol) or cg_rtol <= 0.0:
            raise ValueError(
                "cg_rtol must be finite and strictly positive."
            )

        if cg_maxiter is not None and int(cg_maxiter) < 1:
            raise ValueError(
                "cg_maxiter must be at least one when provided."
            )

        self.cg_rtol = float(cg_rtol)
        self.cg_maxiter = (
            None
            if cg_maxiter is None
            else int(cg_maxiter)
        )

        if correlation_logdet is None:
            self.correlation_logdet = None
        else:
            correlation_logdet = float(
                correlation_logdet
            )

            if not np.isfinite(
                correlation_logdet
            ):
                raise ValueError(
                    "correlation_logdet must be finite."
                )

            self.correlation_logdet = (
                correlation_logdet
            )

        if (
            self.include_normalization
            and self.correlation_logdet is None
        ):
            raise ValueError(
                "include_normalization=True requires "
                "correlation_logdet."
            )

        if self.mask is None:
            valid_pixels = np.ones(
                self.image_shape,
                dtype=bool,
            )
        else:
            valid_pixels = ~self.mask

        valid_pixels.setflags(
            write=False
        )
        self._valid_pixels_2d = (
            valid_pixels
        )

        kernel_shape = (
            self.correlation_kernel.shape
        )

        self._kernel_center = (
            (kernel_shape[0] - 1) // 2,
            (kernel_shape[1] - 1) // 2,
        )

        full_shape = (
            self.image_shape[0]
            + kernel_shape[0]
            - 1,
            self.image_shape[1]
            + kernel_shape[1]
            - 1,
        )

        self._fft_shape = (
            next_fast_len(
                full_shape[0]
            ),
            next_fast_len(
                full_shape[1]
            ),
        )

        kernel_buffer = np.zeros(
            self._fft_shape,
            dtype=float,
        )

        kernel_buffer[
            : kernel_shape[0],
            : kernel_shape[1],
        ] = self.correlation_kernel

        self._kernel_fft = rfftn(
            kernel_buffer
        )

        if self.include_normalization:
            self._normalization = -0.5 * (
                self.n_valid_pixels
                * np.log(
                    2.0 * np.pi
                )
                + 2.0
                * np.sum(
                    np.log(
                        self.noise_values
                    )
                )
                + self.correlation_logdet
            )
        else:
            self._normalization = 0.0

    def _log_likelihood(
        self,
        model_values: np.ndarray,
    ) -> float:
        """Evaluate the correlated Gaussian image log-likelihood."""
        standardized_residual = (
            self.data_values
            - model_values
        ) / self.noise_values

        operator = LinearOperator(
            shape=(
                self.n_valid_pixels,
                self.n_valid_pixels,
            ),
            matvec=self._correlation_matvec,
            dtype=float,
        )

        solution, info = cg(
            operator,
            standardized_residual,
            rtol=self.cg_rtol,
            atol=0.0,
            maxiter=self.cg_maxiter,
        )

        if info != 0:
            raise RuntimeError(
                "Correlated Gaussian covariance solve "
                "did not converge "
                f"(cg info={info})."
            )

        quadratic_form = float(
            np.dot(
                standardized_residual,
                solution,
            )
        )

        if not np.isfinite(
            quadratic_form
        ):
            raise RuntimeError(
                "Correlated Gaussian quadratic "
                "form is not finite."
            )

        tolerance = (
            1.0e-12
            * max(
                1.0,
                float(
                    np.dot(
                        standardized_residual,
                        standardized_residual,
                    )
                ),
            )
        )

        if quadratic_form < -tolerance:
            raise RuntimeError(
                "Correlated Gaussian quadratic form "
                "is negative; the correlation operator "
                "may not be positive definite."
            )

        quadratic_form = max(
            0.0,
            quadratic_form,
        )

        return float(
            -0.5 * quadratic_form
            + self._normalization
        )

    def _correlation_matvec(
        self,
        vector: np.ndarray,
    ) -> np.ndarray:
        """Apply the exact masked stationary correlation matrix."""
        vector = np.asarray(
            vector,
            dtype=float,
        )

        if vector.shape != (
            self.n_valid_pixels,
        ):
            raise ValueError(
                "Correlation operator vector "
                "has incorrect shape."
            )

        source = np.zeros(
            self.image_shape,
            dtype=float,
        )

        source[
            self._valid_pixels_2d
        ] = vector

        fft_buffer = np.zeros(
            self._fft_shape,
            dtype=float,
        )

        fft_buffer[
            : self.image_shape[0],
            : self.image_shape[1],
        ] = source

        convolved = irfftn(
            rfftn(
                fft_buffer
            )
            * self._kernel_fft,
            s=self._fft_shape,
        )

        center_y, center_x = (
            self._kernel_center
        )

        same = convolved[
            center_y:
            center_y
            + self.image_shape[0],
            center_x:
            center_x
            + self.image_shape[1],
        ]

        return np.asarray(
            same[
                self._valid_pixels_2d
            ],
            dtype=float,
        )

    @staticmethod
    def _validate_correlation_kernel(
        kernel: np.ndarray,
    ) -> np.ndarray:
        """Validate one finite stationary correlation kernel."""
        kernel_array = np.array(
            kernel,
            dtype=float,
            copy=True,
        )

        if kernel_array.ndim != 2:
            raise ValueError(
                "correlation_kernel must "
                "be two-dimensional."
            )

        if any(
            size < 1
            or size % 2 == 0
            for size
            in kernel_array.shape
        ):
            raise ValueError(
                "correlation_kernel dimensions "
                "must be positive and odd."
            )

        if not np.all(
            np.isfinite(
                kernel_array
            )
        ):
            raise ValueError(
                "correlation_kernel values "
                "must be finite."
            )

        center = (
            kernel_array.shape[0]
            // 2,
            kernel_array.shape[1]
            // 2,
        )

        if not np.isclose(
            kernel_array[center],
            1.0,
            rtol=1.0e-8,
            atol=1.0e-10,
        ):
            raise ValueError(
                "correlation_kernel must have "
                "unit zero-lag correlation."
            )

        if not np.allclose(
            kernel_array,
            kernel_array[
                ::-1,
                ::-1,
            ],
            rtol=1.0e-10,
            atol=1.0e-12,
        ):
            raise ValueError(
                "correlation_kernel must satisfy "
                "covariance symmetry."
            )

        kernel_array.setflags(
            write=False
        )

        return kernel_array
