"""Tests for the correlated Gaussian image likelihood."""

import numpy as np
import pytest

from baymorph.likelihood import (
    CorrelatedGaussianImageLikelihood,
    GaussianImageLikelihood,
)


def test_identity_correlation_matches_gaussian():
    """Unit correlation must reproduce the independent Gaussian."""
    data = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    model = np.array(
        [
            [0.8, 2.1, 3.2],
            [4.3, 4.9, 5.7],
        ]
    )
    noise = np.array(
        [
            [1.0, 1.2, 0.8],
            [1.1, 0.9, 1.4],
        ]
    )
    mask = np.array(
        [
            [False, True, False],
            [False, False, False],
        ]
    )

    independent = GaussianImageLikelihood(
        data=data,
        noise=noise,
        mask=mask,
        include_normalization=True,
    )

    correlated = (
        CorrelatedGaussianImageLikelihood(
            data=data,
            noise=noise,
            correlation_kernel=np.ones(
                (1, 1)
            ),
            mask=mask,
            include_normalization=True,
            correlation_logdet=0.0,
            cg_rtol=1.0e-12,
        )
    )

    assert correlated(
        model
    ) == pytest.approx(
        independent(model),
        rel=1.0e-12,
        abs=1.0e-12,
    )


def test_correlated_gaussian_matches_dense_masked_reference():
    """FFT/CG evaluation must match a dense masked Gaussian."""
    data = np.array(
        [
            [0.3, -0.1, 0.7, 0.4],
            [1.1, 0.5, -0.2, 0.8],
            [0.2, -0.4, 0.6, 1.0],
        ]
    )

    model = np.array(
        [
            [0.1, -0.2, 0.4, 0.6],
            [0.9, 0.7, -0.1, 0.5],
            [0.3, -0.1, 0.4, 0.8],
        ]
    )

    noise = np.array(
        [
            [1.0, 1.1, 0.9, 1.2],
            [0.8, 1.0, 1.3, 0.95],
            [1.15, 0.85, 1.05, 1.25],
        ]
    )

    mask = np.array(
        [
            [False, True, False, False],
            [False, False, False, True],
            [True, False, False, False],
        ]
    )

    kernel = np.array(
        [
            [0.02, 0.05, 0.02],
            [0.08, 1.00, 0.08],
            [0.02, 0.05, 0.02],
        ]
    )

    valid = ~mask
    coordinates = np.argwhere(
        valid
    )

    dy = (
        coordinates[:, None, 0]
        - coordinates[None, :, 0]
    )
    dx = (
        coordinates[:, None, 1]
        - coordinates[None, :, 1]
    )

    correlation = np.zeros(
        dy.shape,
        dtype=float,
    )

    inside = (
        (np.abs(dy) <= 1)
        & (np.abs(dx) <= 1)
    )

    correlation[
        inside
    ] = kernel[
        dy[inside] + 1,
        dx[inside] + 1,
    ]

    sign, correlation_logdet = (
        np.linalg.slogdet(
            correlation
        )
    )

    assert sign > 0.0

    noise_values = noise[
        valid
    ]

    covariance = (
        noise_values[:, None]
        * correlation
        * noise_values[None, :]
    )

    residual = (
        data[valid]
        - model[valid]
    )

    _, covariance_logdet = (
        np.linalg.slogdet(
            covariance
        )
    )

    quadratic = float(
        residual
        @ np.linalg.solve(
            covariance,
            residual,
        )
    )

    expected = -0.5 * (
        quadratic
        + covariance_logdet
        + residual.size
        * np.log(
            2.0 * np.pi
        )
    )

    likelihood = (
        CorrelatedGaussianImageLikelihood(
            data=data,
            noise=noise,
            correlation_kernel=kernel,
            mask=mask,
            include_normalization=True,
            correlation_logdet=(
                correlation_logdet
            ),
            cg_rtol=1.0e-12,
            cg_maxiter=200,
        )
    )

    assert likelihood(
        model
    ) == pytest.approx(
        expected,
        rel=1.0e-10,
        abs=1.0e-10,
    )


def test_normalization_requires_correlation_logdet():
    """Absolute normalization must not silently omit log|R|."""
    with pytest.raises(
        ValueError,
        match="requires correlation_logdet",
    ):
        CorrelatedGaussianImageLikelihood(
            data=np.ones(
                (2, 2)
            ),
            noise=1.0,
            correlation_kernel=np.ones(
                (1, 1)
            ),
            include_normalization=True,
        )


def test_asymmetric_kernel_is_rejected():
    """Correlation kernels must obey covariance symmetry."""
    kernel = np.array(
        [
            [0.0, 0.1, 0.0],
            [0.2, 1.0, 0.1],
            [0.0, 0.1, 0.0],
        ]
    )

    with pytest.raises(
        ValueError,
        match="covariance symmetry",
    ):
        CorrelatedGaussianImageLikelihood(
            data=np.ones(
                (3, 3)
            ),
            noise=1.0,
            correlation_kernel=kernel,
        )


def test_factory_builds_correlated_gaussian_likelihood():
    """Factory must construct the correlated Gaussian likelihood."""
    from baymorph.likelihood import make_likelihood

    data = np.zeros(
        (4, 5),
        dtype=float,
    )

    noise = np.ones_like(
        data
    )

    kernel = np.ones(
        (1, 1),
        dtype=float,
    )

    likelihood = make_likelihood(
        {
            "type": "correlated_gaussian",
            "include_normalization": False,
            "cg_rtol": 1.0e-10,
        },
        data=data,
        noise=noise,
        correlation_kernel=kernel,
    )

    assert isinstance(
        likelihood,
        CorrelatedGaussianImageLikelihood,
    )


def test_factory_correlated_gaussian_requires_kernel():
    """Factory must not silently fall back to independent pixels."""
    from baymorph.likelihood import make_likelihood

    with pytest.raises(
        ValueError,
        match="requires a correlation kernel",
    ):
        make_likelihood(
            {
                "type": "correlated_gaussian",
            },
            data=np.zeros(
                (3, 3),
                dtype=float,
            ),
            noise=1.0,
        )


@pytest.mark.parametrize(
    "likelihood_type",
    [
        "correlated_gaussian",
        "correlated-gaussian",
        "gaussian_correlated",
        "gaussian-correlated",
    ],
)
def test_factory_accepts_correlated_gaussian_aliases(
    likelihood_type,
):
    """Supported correlated-Gaussian aliases must resolve consistently."""
    from baymorph.likelihood import make_likelihood

    likelihood = make_likelihood(
        likelihood_type,
        data=np.zeros(
            (3, 3),
            dtype=float,
        ),
        noise=1.0,
        correlation_kernel=np.ones(
            (1, 1),
            dtype=float,
        ),
    )

    assert isinstance(
        likelihood,
        CorrelatedGaussianImageLikelihood,
    )


def test_factory_does_not_pass_kernel_to_independent_gaussian():
    """Correlation information must not modify the Gaussian likelihood."""
    from baymorph.likelihood import make_likelihood

    likelihood = make_likelihood(
        {
            "type": "gaussian",
        },
        data=np.zeros(
            (2, 2),
            dtype=float,
        ),
        noise=1.0,
        correlation_kernel=np.ones(
            (1, 1),
            dtype=float,
        ),
    )

    assert isinstance(
        likelihood,
        GaussianImageLikelihood,
    )
