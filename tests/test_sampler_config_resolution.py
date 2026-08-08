"""Tests for backend-owned sampler configuration resolution."""

from __future__ import annotations

from baymorph.pipeline.config import resolve_config


def _base_config():
    """Return the minimum valid pipeline configuration."""
    return {
        "object": {
            "name": "test",
        },
        "data": {
            "image_path": "image.fits",
        },
        "outputs": {
            "root": "outputs",
        },
    }


def test_default_sampler_is_pocomc():
    resolved = resolve_config(
        _base_config()
    )

    sampler = resolved[
        "fit"
    ][
        "sampler"
    ]

    assert sampler["name"] == "pocomc"
    assert sampler["n_effective"] == 512
    assert sampler["n_active"] == 256

    assert "nlive" not in sampler
    assert "bound" not in sampler
    assert "queue_size" not in sampler


def test_explicit_dynesty_uses_only_dynesty_defaults():
    config = _base_config()

    config["fit"] = {
        "sampler": {
            "name": "dynesty",
        }
    }

    resolved = resolve_config(
        config
    )

    sampler = resolved[
        "fit"
    ][
        "sampler"
    ]

    assert sampler["name"] == "dynesty"
    assert sampler["nlive"] == 500
    assert sampler["sample"] == "auto"
    assert sampler["bound"] == "multi"

    assert "n_effective" not in sampler
    assert "n_active" not in sampler
    assert "flow" not in sampler


def test_sampler_overrides_backend_defaults():
    config = _base_config()

    config["fit"] = {
        "sampler": {
            "name": "pocomc",
            "n_effective": 1024,
            "n_active": 512,
            "flow": "custom-flow",
        }
    }

    resolved = resolve_config(
        config
    )

    sampler = resolved[
        "fit"
    ][
        "sampler"
    ]

    assert sampler["n_effective"] == 1024
    assert sampler["n_active"] == 512
    assert sampler["flow"] == "custom-flow"

    assert "nlive" not in sampler


def test_explicit_dynesty_overrides_do_not_leak_pocomc_defaults():
    config = _base_config()

    config["fit"] = {
        "sampler": {
            "name": "dynesty",
            "nlive": 1000,
            "sample": "rwalk",
        }
    }

    resolved = resolve_config(
        config
    )

    sampler = resolved[
        "fit"
    ][
        "sampler"
    ]

    assert sampler["nlive"] == 1000
    assert sampler["sample"] == "rwalk"

    assert "n_effective" not in sampler
    assert "n_evidence" not in sampler
    assert "mcmc_sampler" not in sampler
