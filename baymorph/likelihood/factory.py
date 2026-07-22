"""Factory functions for constructing BayMorph likelihoods."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from baymorph.likelihood.base import ImageLikelihood
from baymorph.likelihood.cash import CashImageLikelihood
from baymorph.likelihood.gaussian import GaussianImageLikelihood
from baymorph.likelihood.poisson_gaussian import (
    PoissonGaussianImageLikelihood,
)


def make_likelihood(
    config: Mapping[str, Any] | str | None,
    *,
    data: np.ndarray,
    noise: np.ndarray | float | None = None,
    mask: np.ndarray | None = None,
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

        For the Gaussian likelihood, this is interpreted as the Gaussian
        standard deviation.

        For the Poisson-Gaussian likelihood, this is used as read noise only
        when ``read_noise`` is not explicitly supplied in ``config``.

        It is ignored by the Cash likelihood.
    mask : array-like of bool, optional
        Boolean mask identifying pixels excluded from likelihood evaluation.

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

    likelihood_type = _normalize_likelihood_type(
        likelihood_type
    )

    if likelihood_type == "gaussian":
        return _make_gaussian_likelihood(
            config=likelihood_config,
            data=data,
            noise=noise,
            mask=mask,
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


def _normalize_likelihood_type(
    likelihood_type: str,
) -> str:
    """Normalize supported likelihood aliases."""
    aliases = {
        "normal": "gaussian",
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