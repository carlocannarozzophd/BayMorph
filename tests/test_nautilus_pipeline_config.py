"""Tests for Nautilus production configuration support."""

from __future__ import annotations

import pytest

from baymorph.pipeline.config import (
    SAMPLER_DEFAULTS,
    resolve_config,
    validate_sampler_config,
)


def _minimal_config() -> dict:
    """Return a minimal valid single-observation configuration."""
    return {
        "object": {
            "name": "test",
        },
        "data": {
            "image_path": "test.fits",
        },
        "outputs": {
            "root": "runs",
        },
    }


def test_nautilus_is_registered_as_pipeline_sampler() -> None:
    """The production configuration layer must expose Nautilus."""
    assert SAMPLER_DEFAULTS["nautilus"] == {
        "name": "nautilus",
        "init": {},
        "run": {},
    }


def test_resolve_config_accepts_nautilus_name() -> None:
    """A simple Nautilus sampler selection must resolve cleanly."""
    config = _minimal_config()

    config["fit"] = {
        "sampler": {
            "name": "nautilus",
        }
    }

    resolved = resolve_config(
        config
    )

    assert resolved["fit"]["sampler"] == {
        "name": "nautilus",
        "init": {},
        "run": {},
    }


def test_resolve_config_preserves_native_nautilus_options() -> None:
    """Native Nautilus constructor and run options must remain nested."""
    config = _minimal_config()

    config["fit"] = {
        "sampler": {
            "name": "nautilus",
            "init": {
                "n_live": 500,
                "seed": 42,
            },
            "run": {
                "n_eff": 1000,
            },
        }
    }

    resolved = resolve_config(
        config
    )

    assert resolved["fit"]["sampler"] == {
        "name": "nautilus",
        "init": {
            "n_live": 500,
            "seed": 42,
        },
        "run": {
            "n_eff": 1000,
        },
    }


def test_nautilus_rejects_baymorph_mode() -> None:
    """Nautilus must not inherit the Dynesty-specific mode contract."""
    with pytest.raises(
        ValueError,
        match="does not define a BayMorph mode",
    ):
        validate_sampler_config(
            {
                "name": "nautilus",
                "mode": "static",
                "init": {},
                "run": {},
            }
        )
