"""Tests for backend-owned sampler configuration resolution."""

from __future__ import annotations

import pytest

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


def test_default_sampler_is_nautilus():
    resolved = resolve_config(
        _base_config()
    )

    assert resolved["fit"]["sampler"] == {
        "name": "nautilus",
        "execution": {
            "processes": "auto",
            "progress": True,
        },
        "init": {},
        "run": {},
    }


def test_archived_emcee_sampler_is_rejected():
    config = _base_config()
    config["fit"] = {
        "sampler": {
            "name": "emcee",
        }
    }

    with pytest.raises(
        ValueError,
        match="Unknown sampler 'emcee'",
    ):
        resolve_config(config)


def test_pocomc_native_kwargs_are_preserved_verbatim():
    config = _base_config()

    config["fit"] = {
        "sampler": {
            "name": "pocomc",
            "init": {
                "n_effective": 1024,
                "n_active": 512,
                "sample": "tpcn",
                "custom_future_option": 17,
            },
            "run": {
                "n_total": 8192,
                "n_evidence": 4096,
                "future_run_option": "value",
            },
        }
    }

    resolved = resolve_config(
        config
    )

    expected = dict(
        config["fit"]["sampler"]
    )
    expected["execution"] = {
        "processes": "auto",
        "progress": True,
    }

    assert resolved["fit"]["sampler"] == expected


def test_dynesty_static_native_kwargs_are_preserved_verbatim():
    config = _base_config()

    config["fit"] = {
        "sampler": {
            "name": "dynesty",
            "mode": "static",
            "init": {
                "nlive": 1000,
                "sample": "rwalk",
                "bound": "multi",
                "future_init_option": 17,
            },
            "run": {
                "dlogz": 0.01,
                "future_run_option": "value",
            },
        }
    }

    resolved = resolve_config(
        config
    )

    expected = dict(
        config["fit"]["sampler"]
    )
    expected["execution"] = {
        "processes": "auto",
        "progress": True,
    }

    assert resolved["fit"]["sampler"] == expected


def test_dynesty_dynamic_native_kwargs_are_preserved_verbatim():
    config = _base_config()

    config["fit"] = {
        "sampler": {
            "name": "dynesty",
            "mode": "dynamic",
            "init": {
                "sample": "rwalk",
                "bound": "multi",
            },
            "run": {
                "nlive_init": 500,
                "nlive_batch": 250,
                "dlogz_init": 0.05,
            },
        }
    }

    resolved = resolve_config(
        config
    )

    expected = dict(
        config["fit"]["sampler"]
    )
    expected["execution"] = {
        "processes": "auto",
        "progress": True,
    }

    assert resolved["fit"]["sampler"] == expected


def test_dynesty_defaults_to_static_mode():
    config = _base_config()

    config["fit"] = {
        "sampler": {
            "name": "dynesty",
        }
    }

    resolved = resolve_config(
        config
    )

    assert resolved["fit"]["sampler"] == {
        "name": "dynesty",
        "mode": "static",
        "execution": {
            "processes": "auto",
            "progress": True,
        },
        "init": {},
        "run": {},
    }


def test_dynesty_rejects_unknown_mode():
    config = _base_config()

    config["fit"] = {
        "sampler": {
            "name": "dynesty",
            "mode": "hyperdynamic",
        }
    }

    with pytest.raises(
        ValueError,
        match="static.*dynamic",
    ):
        resolve_config(config)


def test_pocomc_rejects_dynesty_mode():
    config = _base_config()

    config["fit"] = {
        "sampler": {
            "name": "pocomc",
            "mode": "dynamic",
        }
    }

    with pytest.raises(
        ValueError,
        match="does not define",
    ):
        resolve_config(config)


@pytest.mark.parametrize(
    "section",
    [
        "init",
        "run",
    ],
)
def test_sampler_native_sections_must_be_mappings(
    section,
):
    config = _base_config()

    config["fit"] = {
        "sampler": {
            "name": "pocomc",
            section: 42,
        }
    }

    with pytest.raises(
        TypeError,
        match=rf"sampler\.{section}.*mapping",
    ):
        resolve_config(config)


def test_sampler_execution_override_is_preserved():
    """Explicit execution settings override the automatic defaults."""
    config = _base_config()

    config["fit"] = {
        "sampler": {
            "name": "nautilus",
            "execution": {
                "processes": 3,
                "progress": False,
            },
        }
    }

    resolved = resolve_config(
        config
    )

    assert resolved["fit"]["sampler"]["execution"] == {
        "processes": 3,
        "progress": False,
    }

def test_enabled_comparison_requires_exact_scientific_model_pair():
    """Single-band comparison must not silently accept another model set."""
    config = _base_config()
    config["fit"] = {
        "models": [
            "single_sersic",
        ],
        "comparison": {
            "enabled": True,
            "metric": "bayes_factor",
        },
    }

    with pytest.raises(
        ValueError,
        match="requires exactly",
    ):
        resolve_config(config)


def test_enabled_comparison_accepts_single_and_double_sersic():
    """The frozen comparison pair must resolve successfully."""
    config = _base_config()
    config["fit"] = {
        "models": [
            "single_sersic",
            "double_sersic",
        ],
        "comparison": {
            "enabled": True,
            "metric": "bayes_factor",
        },
    }

    resolved = resolve_config(
        config
    )

    assert resolved["fit"]["comparison"]["enabled"] is True
