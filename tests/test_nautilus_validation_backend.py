"""Tests for the Nautilus scientific-validation adapter."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import validation.backends.nautilus as nautilus_validation
from validation.backends import NautilusValidationBackend
from validation.problems.mathematical import (
    make_gaussian_1d_problem,
)


def test_nautilus_validation_backend_name():
    """The validation backend exposes a stable identifier."""
    assert (
        NautilusValidationBackend().name
        == "nautilus"
    )


def test_nautilus_validation_uses_production_fitter(
    monkeypatch,
):
    """Validation must route through the production Nautilus fitter."""
    captured = {}

    def fake_run(
        self,
        *,
        init_kwargs=None,
        run_kwargs=None,
        execution=None,
        progress_label=None,
    ):
        captured["init"] = dict(
            init_kwargs
        )
        captured["run"] = dict(
            run_kwargs
        )
        captured["execution"] = execution
        captured["progress_label"] = progress_label

        self.loglikelihood(
            np.asarray(
                [0.0],
                dtype=float,
            )
        )

        return SimpleNamespace(
            samples=np.asarray(
                [
                    [-0.2],
                    [0.3],
                ],
                dtype=float,
            ),
            weights=np.asarray(
                [
                    0.4,
                    0.6,
                ],
                dtype=float,
            ),
            log_likelihood=np.asarray(
                [
                    -1.0,
                    -0.5,
                ],
                dtype=float,
            ),
            logz=-2.1,
            logz_error=0.07,
            diagnostics={
                "calls": 1,
                "ess": 120.0,
            },
            evidence=SimpleNamespace(
                uncertainty_kind=(
                    "statistical_ess_approximation"
                ),
            ),
        )

    monkeypatch.setattr(
        nautilus_validation.NautilusFitter,
        "run",
        fake_run,
    )

    result = NautilusValidationBackend().run(
        make_gaussian_1d_problem(),
        seed=37,
        config={
            "init": {
                "n_live": 200,
            },
            "run": {
                "n_eff": 1000,
                "f_live": 0.01,
            },
        },
    )

    assert captured["init"] == {
        "n_live": 200,
        "seed": 37,
    }

    assert captured["run"] == {
        "n_eff": 1000,
        "f_live": 0.01,
    }

    assert captured["execution"] is None
    assert captured["progress_label"] is None

    assert result.backend_name == "nautilus"
    assert result.ncall == 1
    assert result.logz == pytest.approx(
        -2.1
    )
    assert result.logzerr == pytest.approx(
        0.07
    )

    assert result.metadata[
        "reported_likelihood_calls"
    ] == 1

    assert result.metadata[
        "counted_likelihood_calls"
    ] == 1


def test_nautilus_validation_rejects_configured_seed():
    """Experiment-level seeds must remain authoritative."""
    with pytest.raises(
        ValueError,
        match="controlled by the experiment",
    ):
        NautilusValidationBackend().run(
            make_gaussian_1d_problem(),
            seed=4,
            config={
                "init": {
                    "seed": 10,
                }
            },
        )
