"""Observation-level covariates for structured probabilistic coupling."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from astropy import units as u


def wavelength_covariate(
    observations: Mapping[str, Mapping[str, object]],
) -> dict[str, float]:
    """Return centered logarithmic wavelength coordinates.

    Parameters
    ----------
    observations : mapping
        Observation configuration entries containing an explicit
        ``covariates.wavelength`` mapping with ``value`` and ``unit``.

    Returns
    -------
    dict
        Dimensionless centered log-wavelength coordinate for each
        observation.

    Notes
    -----
    BayMorph does not infer wavelength from observation names or filter
    identifiers. Wavelength metadata must be supplied explicitly.
    """
    if not isinstance(observations, Mapping) or not observations:
        raise ValueError(
            "Observations must be a non-empty mapping."
        )

    names: list[str] = []
    wavelengths: list[float] = []

    for name, config in observations.items():
        if not isinstance(name, str) or not name:
            raise ValueError(
                "Observation names must be non-empty strings."
            )

        if not isinstance(config, Mapping):
            raise TypeError(
                f"Observation '{name}' configuration must be a mapping."
            )

        covariates = config.get(
            "covariates"
        )

        if not isinstance(covariates, Mapping):
            raise ValueError(
                f"Observation '{name}' requires a 'covariates' mapping."
            )

        wavelength = covariates.get(
            "wavelength"
        )

        if not isinstance(wavelength, Mapping):
            raise ValueError(
                f"Observation '{name}' requires an explicit "
                "'covariates.wavelength' mapping."
            )

        if set(wavelength) != {
            "value",
            "unit",
        }:
            raise ValueError(
                f"Observation '{name}' wavelength requires exactly "
                "'value' and 'unit'."
            )

        try:
            quantity = float(
                wavelength["value"]
            ) * u.Unit(
                str(
                    wavelength["unit"]
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Observation '{name}' has an invalid wavelength."
            ) from exc

        if not quantity.unit.is_equivalent(
            u.m
        ):
            raise ValueError(
                f"Observation '{name}' wavelength unit must have "
                "physical type length."
            )

        value = float(
            quantity.to_value(
                u.m
            )
        )

        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"Observation '{name}' wavelength must be finite "
                "and positive."
            )

        names.append(
            name
        )
        wavelengths.append(
            value
        )

    log_wavelength = np.log(
        np.asarray(
            wavelengths,
            dtype=float,
        )
    )

    centered = (
        log_wavelength
        - np.mean(
            log_wavelength
        )
    )

    return {
        name: float(value)
        for name, value in zip(
            names,
            centered,
            strict=True,
        )
    }


def wavelength_reference_correlation_matrix(
    observations: Mapping[str, Mapping[str, object]],
    *,
    reference_correlation: float,
    reference_ratio: float = 2.0,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Build a wavelength-dependent Gaussian-copula correlation matrix.

    Parameters
    ----------
    observations : mapping
        Observation configuration entries containing explicit wavelength
        covariates.
    reference_correlation : float
        Latent Gaussian correlation assigned to two observations whose
        wavelengths differ by ``reference_ratio``.
    reference_ratio : float, default=2
        Positive wavelength ratio greater than one used to define the
        correlation scale.

    Returns
    -------
    names : tuple of str
        Observation names in matrix order.
    matrix : numpy.ndarray
        Positive-semidefinite wavelength correlation matrix.

    Notes
    -----
    The off-diagonal correlation is

    ``reference_correlation ** d**2``

    where ``d = abs(log(lambda_i / lambda_j)) / log(reference_ratio)``.

    Equal wavelengths therefore have unit correlation. The construction is
    invariant to the wavelength unit and is equivalent to a
    squared-exponential kernel in log wavelength.
    """
    correlation = float(
        reference_correlation
    )

    if (
        not np.isfinite(correlation)
        or correlation < 0.0
        or correlation > 1.0
    ):
        raise ValueError(
            "Reference correlation must be finite and lie in [0, 1]."
        )

    ratio = float(
        reference_ratio
    )

    if (
        not np.isfinite(ratio)
        or ratio <= 1.0
    ):
        raise ValueError(
            "Reference wavelength ratio must be finite and greater than one."
        )

    coordinates = wavelength_covariate(
        observations
    )

    names = tuple(
        coordinates
    )

    values = np.asarray(
        [
            coordinates[name]
            for name in names
        ],
        dtype=float,
    )

    separation = np.abs(
        values[:, None]
        - values[None, :]
    ) / np.log(
        ratio
    )

    if correlation == 0.0:
        matrix = np.where(
            separation == 0.0,
            1.0,
            0.0,
        )
    else:
        matrix = np.power(
            correlation,
            separation**2,
        )

    np.fill_diagonal(
        matrix,
        1.0,
    )

    return names, np.asarray(
        matrix,
        dtype=float,
    )
