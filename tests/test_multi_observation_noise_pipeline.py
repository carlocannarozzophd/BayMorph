"""Tests for multi-observation pre-fit noise diagnostics."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from baymorph.pipeline import multi_observation_fit
from baymorph.pipeline.multi_observation_noise import (
    estimate_cross_observation_noise_diagnostics,
)


def _products(
    data: np.ndarray,
    *,
    sky_region: bool = True,
):
    return SimpleNamespace(
        prepared_data=np.asarray(
            data,
            dtype=float,
        ),
        noise=np.ones_like(
            data,
            dtype=float,
        ),
        sky_region=(
            SimpleNamespace(
                mask=np.ones_like(
                    data,
                    dtype=bool,
                )
            )
            if sky_region
            else None
        ),
        cutout_bounds={
            "xmin": 0,
            "xmax": data.shape[1],
            "ymin": 0,
            "ymax": data.shape[0],
        },
    )


def _configs():
    return {
        "g": {
            "data": {
                "image_path": "registered.fits",
            }
        },
        "r": {
            "data": {
                "image_path": "registered.fits",
            }
        },
    }


def test_registered_blank_sky_correlation_is_estimated():
    rng = np.random.default_rng(
        1234
    )

    base = rng.normal(
        size=(
            20,
            20,
        )
    )

    diagnostics = (
        estimate_cross_observation_noise_diagnostics(
            image_products={
                "g": _products(
                    base
                ),
                "r": _products(
                    2.0 * base
                ),
            },
            observation_configs=_configs(),
            max_lag=1,
            minimum_overlap=50,
        )
    )

    pair = diagnostics[
        "pairs"
    ][0]

    assert pair[
        "status"
    ] == "estimated"

    assert pair[
        "alignment"
    ] == "registered"

    assert pair[
        "zero_lag_correlation"
    ] == pytest.approx(
        1.0
    )

    assert pair[
        "peak_abs_correlation"
    ] == pytest.approx(
        1.0
    )

    assert pair[
        "peak_lag_y"
    ] == 0

    assert pair[
        "peak_lag_x"
    ] == 0


def test_shape_mismatch_is_not_resampled():
    diagnostics = (
        estimate_cross_observation_noise_diagnostics(
            image_products={
                "g": _products(
                    np.zeros(
                        (
                            20,
                            20,
                        )
                    )
                ),
                "r": _products(
                    np.zeros(
                        (
                            21,
                            20,
                        )
                    )
                ),
            },
            observation_configs=_configs(),
        )
    )

    pair = diagnostics[
        "pairs"
    ][0]

    assert pair[
        "status"
    ] == "not_estimable_shape_mismatch"


def test_missing_sky_region_is_reported():
    diagnostics = (
        estimate_cross_observation_noise_diagnostics(
            image_products={
                "g": _products(
                    np.zeros(
                        (
                            20,
                            20,
                        )
                    )
                ),
                "r": _products(
                    np.zeros(
                        (
                            20,
                            20,
                        )
                    ),
                    sky_region=False,
                ),
            },
            observation_configs=_configs(),
        )
    )

    pair = diagnostics[
        "pairs"
    ][0]

    assert pair[
        "status"
    ] == "not_estimable_missing_sky_region"


def test_noise_diagnostic_precedes_sampler(
    monkeypatch,
):
    events = []

    raw_config = {
        "fit": {
            "models": [
                "single_sersic",
            ],
            "sampler": {
                "name": "nautilus",
            },
        }
    }

    context = SimpleNamespace(
        problem=object(),
        image_products={},
        observation_configs={},
        likelihood_states={},
    )

    monkeypatch.setattr(
        multi_observation_fit,
        "_load_multi_observation_config",
        lambda config: (
            raw_config,
            Path("."),
        ),
    )

    monkeypatch.setattr(
        multi_observation_fit,
        "_build_multi_observation_context",
        lambda config, base_dir: context,
    )

    def fake_diagnostic(
        *,
        context,
        observation_configs,
    ):
        events.append(
            "noise_diagnostic"
        )

        return {
            "status": "completed",
            "pairs": [],
        }

    monkeypatch.setattr(
        multi_observation_fit,
        "_estimate_pre_fit_noise_diagnostics",
        fake_diagnostic,
    )

    result = SimpleNamespace(
        diagnostics={},
    )

    def fake_sampler(
        *,
        fit_config,
        model_name,
        context,
        checkpoint_directory=None,
    ):
        events.append(
            "sampler"
        )
        return result

    monkeypatch.setattr(
        multi_observation_fit,
        "_run_multi_observation_inference",
        fake_sampler,
    )

    returned = (
        multi_observation_fit
        .run_multi_observation_inference_from_config(
            {}
        )
    )

    assert returned is result

    assert events == [
        "noise_diagnostic",
        "sampler",
    ]

    assert (
        returned.diagnostics[
            "pre_fit_cross_observation_noise"
        ][
            "status"
        ]
        == "completed"
    )
