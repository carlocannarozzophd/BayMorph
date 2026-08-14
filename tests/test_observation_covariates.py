"""Tests for observation-level probabilistic covariates."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting.observation_covariates import (
    wavelength_covariate,
)


def _observations(
    *,
    unit: str = "Angstrom",
    scale: float = 1.0,
):
    return {
        "first": {
            "covariates": {
                "wavelength": {
                    "value": 4000.0 * scale,
                    "unit": unit,
                }
            }
        },
        "second": {
            "covariates": {
                "wavelength": {
                    "value": 6000.0 * scale,
                    "unit": unit,
                }
            }
        },
        "third": {
            "covariates": {
                "wavelength": {
                    "value": 9000.0 * scale,
                    "unit": unit,
                }
            }
        },
    }


def test_log_wavelength_covariate_is_centered():
    values = wavelength_covariate(
        _observations()
    )

    assert sum(
        values.values()
    ) == pytest.approx(
        0.0,
        abs=1.0e-14,
    )


def test_log_wavelength_covariate_preserves_log_ratios():
    values = wavelength_covariate(
        _observations()
    )

    assert (
        values["third"]
        - values["first"]
    ) == pytest.approx(
        np.log(
            9000.0 / 4000.0
        )
    )


def test_log_wavelength_covariate_is_unit_invariant():
    angstrom = wavelength_covariate(
        _observations()
    )

    nanometre = wavelength_covariate(
        {
            "first": {
                "covariates": {
                    "wavelength": {
                        "value": 400.0,
                        "unit": "nm",
                    }
                }
            },
            "second": {
                "covariates": {
                    "wavelength": {
                        "value": 600.0,
                        "unit": "nm",
                    }
                }
            },
            "third": {
                "covariates": {
                    "wavelength": {
                        "value": 900.0,
                        "unit": "nm",
                    }
                }
            },
        }
    )

    assert nanometre == pytest.approx(
        angstrom,
        abs=1.0e-14,
    )


def test_wavelength_covariate_requires_explicit_metadata():
    observations = _observations()
    del observations[
        "second"
    ][
        "covariates"
    ]

    with pytest.raises(
        ValueError,
        match="covariates",
    ):
        wavelength_covariate(
            observations
        )


def test_wavelength_covariate_rejects_non_length_unit():
    observations = _observations()
    observations[
        "first"
    ][
        "covariates"
    ][
        "wavelength"
    ][
        "unit"
    ] = "s"

    with pytest.raises(
        ValueError,
        match="length",
    ):
        wavelength_covariate(
            observations
        )


def test_wavelength_covariate_does_not_use_observation_names():
    observations = {
        "F200W": {
            "covariates": {
                "wavelength": {
                    "value": 5000.0,
                    "unit": "Angstrom",
                }
            }
        },
        "g": {
            "covariates": {
                "wavelength": {
                    "value": 7000.0,
                    "unit": "Angstrom",
                }
            }
        },
    }

    values = wavelength_covariate(
        observations
    )

    assert values["F200W"] < values["g"]


def test_wavelength_correlation_matrix_is_unit_invariant():
    from baymorph.fitting.observation_covariates import (
        wavelength_correlation_matrix,
    )

    angstrom_names, angstrom_matrix = wavelength_correlation_matrix(
        _observations(),
        correlation=0.8,
        length_scale=0.5,
    )

    nanometre_names, nanometre_matrix = wavelength_correlation_matrix(
        {
            "first": {
                "covariates": {
                    "wavelength": {
                        "value": 400.0,
                        "unit": "nm",
                    }
                }
            },
            "second": {
                "covariates": {
                    "wavelength": {
                        "value": 600.0,
                        "unit": "nm",
                    }
                }
            },
            "third": {
                "covariates": {
                    "wavelength": {
                        "value": 900.0,
                        "unit": "nm",
                    }
                }
            },
        },
        correlation=0.8,
        length_scale=0.5,
    )

    assert nanometre_names == angstrom_names
    assert nanometre_matrix == pytest.approx(
        angstrom_matrix,
        abs=1.0e-14,
    )


def test_wavelength_correlation_decreases_with_wavelength_ratio():
    from baymorph.fitting.observation_covariates import (
        wavelength_correlation_matrix,
    )

    names, matrix = wavelength_correlation_matrix(
        _observations(),
        correlation=0.9,
        length_scale=0.5,
    )

    index = {
        name: position
        for position, name in enumerate(names)
    }

    first = index["first"]
    second = index["second"]
    third = index["third"]

    assert (
        matrix[first, second]
        > matrix[first, third]
    )


def test_wavelength_correlation_preserves_unit_diagonal():
    from baymorph.fitting.observation_covariates import (
        wavelength_correlation_matrix,
    )

    _, matrix = wavelength_correlation_matrix(
        _observations(),
        correlation=0.75,
        length_scale=0.4,
    )

    assert np.diag(
        matrix
    ) == pytest.approx(
        np.ones(3)
    )


def test_reference_correlation_has_literal_factor_two_meaning():
    from baymorph.fitting.observation_covariates import (
        wavelength_reference_correlation_matrix,
    )

    observations = {
        "first": {
            "covariates": {
                "wavelength": {
                    "value": 4000.0,
                    "unit": "Angstrom",
                }
            }
        },
        "second": {
            "covariates": {
                "wavelength": {
                    "value": 8000.0,
                    "unit": "Angstrom",
                }
            }
        },
    }

    _, matrix = wavelength_reference_correlation_matrix(
        observations,
        reference_correlation=0.6,
    )

    assert matrix[0, 1] == pytest.approx(
        0.6
    )


def test_reference_correlation_is_unit_invariant():
    from baymorph.fitting.observation_covariates import (
        wavelength_reference_correlation_matrix,
    )

    _, angstrom = wavelength_reference_correlation_matrix(
        _observations(),
        reference_correlation=0.5,
    )

    nanometre_observations = {
        "first": {
            "covariates": {
                "wavelength": {
                    "value": 400.0,
                    "unit": "nm",
                }
            }
        },
        "second": {
            "covariates": {
                "wavelength": {
                    "value": 600.0,
                    "unit": "nm",
                }
            }
        },
        "third": {
            "covariates": {
                "wavelength": {
                    "value": 900.0,
                    "unit": "nm",
                }
            }
        },
    }

    _, nanometre = wavelength_reference_correlation_matrix(
        nanometre_observations,
        reference_correlation=0.5,
    )

    assert nanometre == pytest.approx(
        angstrom,
        abs=1.0e-14,
    )


def test_equal_wavelengths_have_unit_correlation():
    from baymorph.fitting.observation_covariates import (
        wavelength_reference_correlation_matrix,
    )

    observations = {
        "first": {
            "covariates": {
                "wavelength": {
                    "value": 5000.0,
                    "unit": "Angstrom",
                }
            }
        },
        "second": {
            "covariates": {
                "wavelength": {
                    "value": 500.0,
                    "unit": "nm",
                }
            }
        },
    }

    _, matrix = wavelength_reference_correlation_matrix(
        observations,
        reference_correlation=0.3,
    )

    assert matrix == pytest.approx(
        np.ones(
            (2, 2)
        )
    )


def test_zero_reference_correlation_gives_independence_for_distinct_wavelengths():
    from baymorph.fitting.observation_covariates import (
        wavelength_reference_correlation_matrix,
    )

    _, matrix = wavelength_reference_correlation_matrix(
        _observations(),
        reference_correlation=0.0,
    )

    assert matrix == pytest.approx(
        np.eye(3)
    )


def test_unit_reference_correlation_gives_exact_sharing():
    from baymorph.fitting.observation_covariates import (
        wavelength_reference_correlation_matrix,
    )

    _, matrix = wavelength_reference_correlation_matrix(
        _observations(),
        reference_correlation=1.0,
    )

    assert matrix == pytest.approx(
        np.ones(
            (3, 3)
        )
    )
