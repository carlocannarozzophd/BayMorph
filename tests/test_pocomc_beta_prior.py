"""Tests for pocoMC translation of BayMorph beta priors."""

from __future__ import annotations

import pytest

from baymorph.fitting.parameters import (
    Parameter,
    ParameterSet,
)
from baymorph.fitting.pocomc import (
    _build_pocomc_prior,
)
from baymorph.fitting.priors import (
    BetaPrior,
)


class _FakePocoMC:
    """Minimal pocoMC namespace used to inspect translated priors."""

    @staticmethod
    def Prior(distributions):
        """Return translated distributions unchanged."""
        return distributions


def test_pocomc_translates_scaled_beta_prior():
    parameters = ParameterSet(
        [
            Parameter(
                "rho",
                prior=BetaPrior(
                    alpha=2.0,
                    beta=2.0,
                    lower=0.0,
                    upper=1.0,
                ),
            )
        ]
    )

    distributions = _build_pocomc_prior(
        _FakePocoMC,
        parameters,
    )

    assert len(distributions) == 1

    distribution = distributions[0]

    assert distribution.mean() == pytest.approx(
        0.5
    )
    assert distribution.ppf(
        0.5
    ) == pytest.approx(
        0.5
    )
