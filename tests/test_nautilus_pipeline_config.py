"""Tests for Nautilus production configuration support."""

from __future__ import annotations

import pytest

from baymorph.pipeline.config import (
    SAMPLER_DEFAULTS,
    resolve_config,
    validate_sampler_config,
)
from baymorph.pipeline.real_fit import (
    _prepare_sampler_checkpoint,
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
        "execution": {
            "processes": "auto",
            "progress": True,
        },
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
        "execution": {
            "processes": "auto",
            "progress": True,
        },
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
        "execution": {
            "processes": "auto",
            "progress": True,
        },
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


def _prepare_checkpoint(
    tmp_path,
    monkeypatch,
    *,
    model_name="single_sersic",
):
    """Resolve one signed test checkpoint without building a real problem."""
    monkeypatch.setattr(
        "baymorph.pipeline.real_fit._real_fit_problem_signature",
        lambda **kwargs: f"signature:{kwargs['model_name']}",
    )

    return _prepare_sampler_checkpoint(
        {
            "name": "nautilus",
            "execution": {
                "processes": 1,
                "progress": False,
            },
            "init": {
                "filepath": "auto",
                "resume": True,
                "seed": 42,
            },
            "run": {
                "n_eff": 1000,
            },
        },
        model_name=model_name,
        problem=object(),
        likelihood_cfg={
            "type": "gaussian",
            "include_normalization": True,
        },
        data_fit=None,
        noise_fit=None,
        mask_fit=None,
        correlation_kernel_fit=None,
        checkpoint_directory=(
            tmp_path
            / model_name
        ),
    )


def test_nautilus_auto_checkpoint_is_model_specific(
    tmp_path,
    monkeypatch,
) -> None:
    """Automatic checkpoints and signatures must live per fitted model."""
    single = _prepare_checkpoint(
        tmp_path,
        monkeypatch,
        model_name="single_sersic",
    )
    double = _prepare_checkpoint(
        tmp_path,
        monkeypatch,
        model_name="double_sersic",
    )

    single_path = single["init"]["filepath"]
    double_path = double["init"]["filepath"]

    assert single_path != double_path
    assert single_path.endswith(
        "single_sersic/nautilus_checkpoint.hdf5"
    )
    assert double_path.endswith(
        "double_sersic/nautilus_checkpoint.hdf5"
    )
    assert (
        tmp_path
        / "single_sersic"
        / "nautilus_checkpoint.hdf5.target.json"
    ).is_file()


def test_nautilus_resume_rejects_target_mismatch(
    tmp_path,
    monkeypatch,
) -> None:
    """An existing checkpoint must not resume against another target."""
    resolved = _prepare_checkpoint(
        tmp_path,
        monkeypatch,
    )
    checkpoint = tmp_path / "single_sersic" / "nautilus_checkpoint.hdf5"
    checkpoint.touch()
    monkeypatch.setattr(
        "baymorph.pipeline.real_fit._real_fit_problem_signature",
        lambda **kwargs: "changed-target",
    )

    with pytest.raises(
        RuntimeError,
        match="does not match",
    ):
        _prepare_sampler_checkpoint(
            resolved,
            model_name="single_sersic",
            problem=object(),
            likelihood_cfg={
                "type": "gaussian",
                "include_normalization": True,
            },
            data_fit=None,
            noise_fit=None,
            mask_fit=None,
            correlation_kernel_fit=None,
            checkpoint_directory=checkpoint.parent,
        )
