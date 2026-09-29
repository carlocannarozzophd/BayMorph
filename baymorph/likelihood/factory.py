"""Factory functions for constructing BayMorph likelihoods."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from baymorph.likelihood.base import ImageLikelihood
from baymorph.likelihood.cash import CashImageLikelihood
from baymorph.likelihood.correlated_gaussian import (
    CorrelatedGaussianImageLikelihood,
)
from baymorph.likelihood.gaussian import GaussianImageLikelihood
from baymorph.likelihood.poisson_gaussian import (
    PoissonGaussianImageLikelihood,
)
from baymorph.likelihood.vecchia_gaussian import (
    VecchiaCorrelatedGaussianImageLikelihood,
    VecchiaFactorization,
)


def make_likelihood(
    config: Mapping[str, Any] | str | None,
    *,
    data: np.ndarray,
    noise: np.ndarray | float | None = None,
    mask: np.ndarray | None = None,
    correlation_kernel: np.ndarray | None = None,
    correlation_logdet: float | None = None,
    vecchia_factorization: VecchiaFactorization | None = None,
) -> ImageLikelihood:
    """Construct an image likelihood from configuration.

    Parameters
    ----------
    config : mapping, str, or None
        Likelihood configuration. A string is interpreted as the likelihood
        type. If ``None``, the Gaussian likelihood is selected.
    data : array-like
        Observed image.
    noise : array-like or float, optional
        Noise information supplied by the image-preparation pipeline.

        For Gaussian likelihoods, this is interpreted as the marginal
        standard deviation.

        For the Poisson-Gaussian likelihood, this is used as read noise only
        when ``read_noise`` is not explicitly supplied in ``config``.

        It is ignored by the Cash likelihood.
    mask : array-like of bool, optional
        Boolean mask identifying pixels excluded from likelihood evaluation.
    correlation_kernel : array-like, optional
        Fixed two-dimensional stationary correlation kernel. Required by the
        correlated Gaussian likelihood and ignored by all other likelihoods.
    correlation_logdet : float, optional
        Generated log determinant of the exact masked correlation matrix.
        Pipeline callers supply this separately from user configuration.
    vecchia_factorization : VecchiaFactorization, optional
        Precomputed, source-validated nearest-neighbour conditional Gaussian
        factorization. Required when ``factorization.type`` is
        ``"nearest_neighbor"``.

    Returns
    -------
    ImageLikelihood
        Configured likelihood object.

    Raises
    ------
    TypeError
        If ``config`` is not a mapping, string, or ``None``.
    ValueError
        If the likelihood type is unknown, required values are missing, or
        unsupported configuration options are supplied.
    """
    likelihood_config = _normalize_config(config)

    likelihood_type = str(
        likelihood_config.pop(
            "type",
            "gaussian",
        )
    ).strip().lower()

    likelihood_type = normalize_likelihood_type(
        likelihood_type
    )

    if likelihood_type == "gaussian":
        return _make_gaussian_likelihood(
            config=likelihood_config,
            data=data,
            noise=noise,
            mask=mask,
        )

    if likelihood_type == "correlated_gaussian":
        return _make_correlated_gaussian_likelihood(
            config=likelihood_config,
            data=data,
            noise=noise,
            mask=mask,
            correlation_kernel=correlation_kernel,
            correlation_logdet=correlation_logdet,
            vecchia_factorization=vecchia_factorization,
        )

    if likelihood_type == "cash":
        return _make_cash_likelihood(
            config=likelihood_config,
            data=data,
            mask=mask,
        )

    if likelihood_type == "poisson_gaussian":
        return _make_poisson_gaussian_likelihood(
            config=likelihood_config,
            data=data,
            noise=noise,
            mask=mask,
        )

    available = (
        "gaussian",
        "correlated_gaussian",
        "cash",
        "poisson_gaussian",
    )

    raise ValueError(
        f"Unknown likelihood type {likelihood_type!r}. "
        f"Available types are: {', '.join(available)}."
    )


def _make_gaussian_likelihood(
    *,
    config: dict[str, Any],
    data: np.ndarray,
    noise: np.ndarray | float | None,
    mask: np.ndarray | None,
) -> GaussianImageLikelihood:
    """Construct a Gaussian image likelihood."""
    gaussian_noise = config.pop(
        "noise",
        noise,
    )

    if gaussian_noise is None:
        raise ValueError(
            "Gaussian likelihood requires a noise map or scalar noise."
        )

    include_normalization = config.pop(
        "include_normalization",
        True,
    )

    _reject_unknown_options(
        config,
        likelihood_type="gaussian",
    )

    return GaussianImageLikelihood(
        data=data,
        noise=gaussian_noise,
        mask=mask,
        include_normalization=include_normalization,
    )


def _make_correlated_gaussian_likelihood(
    *,
    config: dict[str, Any],
    data: np.ndarray,
    noise: np.ndarray | float | None,
    mask: np.ndarray | None,
    correlation_kernel: np.ndarray | None,
    correlation_logdet: float | None,
    vecchia_factorization: VecchiaFactorization | None,
) -> ImageLikelihood:
    """Construct a correlated Gaussian image likelihood."""
    calibration = config.pop("calibration", None)
    if calibration is not None and not isinstance(calibration, Mapping):
        raise TypeError("correlated_gaussian calibration must be a mapping.")
    config.pop("correlation_logdet_fingerprint", None)

    gaussian_noise = config.pop(
        "noise",
        noise,
    )

    if gaussian_noise is None:
        raise ValueError(
            "Correlated Gaussian likelihood requires "
            "a noise map or scalar noise."
        )

    if correlation_kernel is None:
        raise ValueError(
            "Correlated Gaussian likelihood requires "
            "a correlation kernel."
        )

    include_normalization = config.pop(
        "include_normalization",
        False,
    )

    factorization_config = config.pop(
        "factorization",
        None,
    )

    if factorization_config is not None:
        return _make_vecchia_correlated_gaussian_likelihood(
            config=config,
            factorization_config=factorization_config,
            data=data,
            noise=gaussian_noise,
            mask=mask,
            correlation_kernel=correlation_kernel,
            correlation_logdet=correlation_logdet,
            vecchia_factorization=vecchia_factorization,
            include_normalization=include_normalization,
        )

    if vecchia_factorization is not None:
        raise ValueError(
            "A nearest-neighbour factorization was supplied without setting "
            "factorization.type='nearest_neighbor'."
        )

    configured_logdet = config.pop("correlation_logdet", None)
    if correlation_logdet is not None and configured_logdet is not None:
        raise ValueError(
            "correlation_logdet must come from either the generated pipeline "
            "artifact or low-level configuration, not both."
        )
    if correlation_logdet is None:
        correlation_logdet = configured_logdet

    cg_rtol = config.pop(
        "cg_rtol",
        5.0e-7,
    )

    cg_maxiter = config.pop(
        "cg_maxiter",
        None,
    )

    _reject_unknown_options(
        config,
        likelihood_type="correlated_gaussian",
    )

    return CorrelatedGaussianImageLikelihood(
        data=data,
        noise=gaussian_noise,
        correlation_kernel=correlation_kernel,
        mask=mask,
        include_normalization=include_normalization,
        correlation_logdet=correlation_logdet,
        cg_rtol=cg_rtol,
        cg_maxiter=cg_maxiter,
    )


def _make_vecchia_correlated_gaussian_likelihood(
    *,
    config: dict[str, Any],
    factorization_config: Mapping[str, Any],
    data: np.ndarray,
    noise: np.ndarray | float,
    mask: np.ndarray | None,
    correlation_kernel: np.ndarray,
    correlation_logdet: float | None,
    vecchia_factorization: VecchiaFactorization | None,
    include_normalization: bool,
) -> VecchiaCorrelatedGaussianImageLikelihood:
    """Construct an explicitly configured nearest-neighbour likelihood."""
    if not isinstance(factorization_config, Mapping):
        raise TypeError(
            "correlated_gaussian factorization must be a mapping."
        )

    options = dict(factorization_config)
    factorization_type = str(
        options.pop(
            "type",
            "",
        )
    ).strip().lower()

    if factorization_type != "nearest_neighbor":
        raise ValueError(
            "correlated_gaussian factorization.type must be exactly "
            "'nearest_neighbor'."
        )

    if (
        correlation_logdet is not None
        or "correlation_logdet" in config
    ):
        raise ValueError(
            "Nearest-neighbour inference cannot use correlation_logdet; "
            "normalization and quadratic form must come from the same "
            "factorization artifact."
        )

    if vecchia_factorization is None:
        raise ValueError(
            "factorization.type='nearest_neighbor' requires a precomputed "
            "and source-validated factorization artifact."
        )

    max_neighbors = options.pop(
        "max_neighbors",
        64,
    )
    min_conditional_variance = options.pop(
        "min_conditional_variance",
        1.0e-12,
    )

    _reject_unknown_options(
        options,
        likelihood_type=(
            "correlated_gaussian nearest-neighbour factorization"
        ),
    )
    _reject_unknown_options(
        config,
        likelihood_type="correlated_gaussian",
    )

    return VecchiaCorrelatedGaussianImageLikelihood(
        data=data,
        noise=noise,
        correlation_kernel=correlation_kernel,
        mask=mask,
        factorization=vecchia_factorization,
        max_neighbors=max_neighbors,
        min_conditional_variance=min_conditional_variance,
        include_normalization=include_normalization,
    )


def _make_cash_likelihood(
    *,
    config: dict[str, Any],
    data: np.ndarray,
    mask: np.ndarray | None,
) -> CashImageLikelihood:
    """Construct a Cash image likelihood."""
    include_data_term = config.pop(
        "include_data_term",
        False,
    )

    require_integer_data = config.pop(
        "require_integer_data",
        True,
    )

    integer_tolerance = config.pop(
        "integer_tolerance",
        1.0e-10,
    )

    _reject_unknown_options(
        config,
        likelihood_type="cash",
    )

    return CashImageLikelihood(
        data=data,
        mask=mask,
        include_data_term=include_data_term,
        require_integer_data=require_integer_data,
        integer_tolerance=integer_tolerance,
    )


def _make_poisson_gaussian_likelihood(
    *,
    config: dict[str, Any],
    data: np.ndarray,
    noise: np.ndarray | float | None,
    mask: np.ndarray | None,
) -> PoissonGaussianImageLikelihood:
    """Construct a Poisson-Gaussian image likelihood."""
    read_noise = config.pop(
        "read_noise",
        noise,
    )

    if read_noise is None:
        raise ValueError(
            "Poisson-Gaussian likelihood requires read_noise."
        )

    tail_probability = config.pop(
        "tail_probability",
        1.0e-12,
    )

    minimum_terms = config.pop(
        "minimum_terms",
        20,
    )

    maximum_terms = config.pop(
        "maximum_terms",
        100000,
    )

    _reject_unknown_options(
        config,
        likelihood_type="poisson_gaussian",
    )

    return PoissonGaussianImageLikelihood(
        data=data,
        read_noise=read_noise,
        mask=mask,
        tail_probability=tail_probability,
        minimum_terms=minimum_terms,
        maximum_terms=maximum_terms,
    )


def _normalize_config(
    config: Mapping[str, Any] | str | None,
) -> dict[str, Any]:
    """Return a mutable likelihood configuration dictionary."""
    if config is None:
        return {
            "type": "gaussian",
        }

    if isinstance(config, str):
        return {
            "type": config,
        }

    if not isinstance(config, Mapping):
        raise TypeError(
            "Likelihood configuration must be a mapping, string, or None."
        )

    return dict(config)


def normalize_likelihood_type(
    likelihood_type: str,
) -> str:
    """Return the canonical name of a supported likelihood alias.

    This is the single alias table shared by the likelihood factory, the
    pipeline likelihood state, and the runner.
    """
    likelihood_type = str(
        likelihood_type
    ).strip().lower()

    aliases = {
        "normal": "gaussian",
        "correlated-gaussian": "correlated_gaussian",
        "gaussian_correlated": "correlated_gaussian",
        "gaussian-correlated": "correlated_gaussian",
        "poisson": "cash",
        "cash_statistic": "cash",
        "poisson-gaussian": "poisson_gaussian",
        "poisson+gaussian": "poisson_gaussian",
    }

    return aliases.get(
        likelihood_type,
        likelihood_type,
    )


def _reject_unknown_options(
    config: Mapping[str, Any],
    *,
    likelihood_type: str,
) -> None:
    """Reject unsupported likelihood configuration options."""
    if not config:
        return

    unknown_options = ", ".join(
        sorted(config)
    )

    raise ValueError(
        f"Unknown options for {likelihood_type!r} likelihood: "
        f"{unknown_options}."
    )
