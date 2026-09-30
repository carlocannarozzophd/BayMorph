"""Tests for the shared checkpoint-signature contracts."""

from __future__ import annotations

import pytest

from baymorph.pipeline import multi_observation_fit, real_fit
from baymorph.pipeline.checkpoint_signing import (
    canonical_json_sha256,
    prepare_signed_sampler_checkpoint,
)


def _unused_signature() -> str:
    """Fail if a problem signature is computed when none is needed."""
    raise AssertionError(
        "The problem signature must not be computed."
    )


def test_canonical_json_serialization_is_pinned():
    """Every signature and completion marker depends on these exact bytes."""
    payload = {
        "schema": "test",
        "values": [1, 2.5, None, True],
        "nested": {
            "z": -0.5,
            "label": "Sérsic",
        },
    }

    assert canonical_json_sha256(payload) == (
        "0dbdb7e64eb30bcd61ab0b72259179ddb32ec9a9101f7a702ae2d2aa9bec367d"
    )


def test_canonical_json_rejects_non_finite_numbers():
    with pytest.raises(
        ValueError,
    ):
        canonical_json_sha256(
            {
                "value": float("nan"),
            }
        )


def test_checkpoint_schemas_are_owned_by_each_workflow():
    """Signature schemas are part of every recorded checkpoint target."""
    assert real_fit._CHECKPOINT_SCHEMAS == {
        "nautilus": "baymorph_nautilus_checkpoint_target_v1",
        "pocomc": "baymorph_pocomc_checkpoint_target_v1",
    }
    assert multi_observation_fit._CHECKPOINT_SCHEMAS == {
        "nautilus": (
            "baymorph_multi_observation_nautilus_checkpoint_target_v1"
        ),
        "pocomc": (
            "baymorph_multi_observation_pocomc_checkpoint_target_v1"
        ),
    }


def test_backend_without_checkpoint_is_returned_unchanged():
    sampler_config = {
        "name": "dynesty",
        "mode": "static",
        "init": {},
        "run": {},
    }

    resolved = prepare_signed_sampler_checkpoint(
        sampler_config,
        schemas=real_fit._CHECKPOINT_SCHEMAS,
        checkpoint_directory=None,
        problem_signature=_unused_signature,
    )

    assert resolved is sampler_config


@pytest.mark.parametrize(
    "sampler_config",
    [
        {
            "name": "nautilus",
            "init": {
                "seed": 1,
            },
        },
        {
            "name": "pocomc",
            "run": {
                "n_total": 128,
            },
        },
    ],
    ids=(
        "nautilus",
        "pocomc",
    ),
)
def test_undeclared_checkpoint_does_not_sign_the_problem(
    sampler_config,
):
    """Without a checkpoint path the configuration is copied, not signed."""
    resolved = prepare_signed_sampler_checkpoint(
        sampler_config,
        schemas=real_fit._CHECKPOINT_SCHEMAS,
        checkpoint_directory=None,
        problem_signature=_unused_signature,
    )

    assert resolved == sampler_config
    assert resolved is not sampler_config
