"""Tests for inference-region configuration."""

from __future__ import annotations

from copy import deepcopy

import pytest

from baymorph.pipeline.config import (
    DEFAULT_CONFIG,
    resolve_config,
)


def _base_config() -> dict:
    """Return one minimal valid pipeline configuration."""
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


def test_default_fit_region_is_full() -> None:
    resolved = resolve_config(
        _base_config()
    )

    assert resolved["fit"]["region"] == {
        "mode": "full",
        "radius": None,
    }


def test_circular_fit_region_is_resolved() -> None:
    config = _base_config()
    config["fit"] = {
        "region": {
            "mode": "circular",
            "radius": 300,
        },
    }

    resolved = resolve_config(
        config
    )

    assert resolved["fit"]["region"] == {
        "mode": "circular",
        "radius": 300.0,
    }


def test_fit_region_mode_is_normalized() -> None:
    config = _base_config()
    config["fit"] = {
        "region": {
            "mode": " Circular ",
            "radius": 250,
        },
    }

    resolved = resolve_config(
        config
    )

    assert (
        resolved["fit"]["region"]["mode"]
        == "circular"
    )


@pytest.mark.parametrize(
    "region",
    [
        "circular",
        [],
        300,
    ],
)
def test_fit_region_must_be_mapping(
    region,
) -> None:
    config = _base_config()
    config["fit"] = {
        "region": region,
    }

    with pytest.raises(
        TypeError,
        match="fit.region must be a mapping",
    ):
        resolve_config(
            config
        )


def test_fit_region_rejects_unknown_mode() -> None:
    config = _base_config()
    config["fit"] = {
        "region": {
            "mode": "ellipse",
        },
    }

    with pytest.raises(
        ValueError,
        match="fit.region.mode",
    ):
        resolve_config(
            config
        )


def test_full_fit_region_rejects_radius() -> None:
    config = _base_config()
    config["fit"] = {
        "region": {
            "mode": "full",
            "radius": 300,
        },
    }

    with pytest.raises(
        ValueError,
        match="must be null",
    ):
        resolve_config(
            config
        )


def test_circular_fit_region_requires_radius() -> None:
    config = _base_config()
    config["fit"] = {
        "region": {
            "mode": "circular",
        },
    }

    with pytest.raises(
        ValueError,
        match="radius is required",
    ):
        resolve_config(
            config
        )


@pytest.mark.parametrize(
    "radius",
    [
        0,
        -1,
        float("inf"),
        float("nan"),
    ],
)
def test_circular_fit_region_requires_positive_finite_radius(
    radius,
) -> None:
    config = _base_config()
    config["fit"] = {
        "region": {
            "mode": "circular",
            "radius": radius,
        },
    }

    with pytest.raises(
        ValueError,
        match="finite positive number",
    ):
        resolve_config(
            config
        )


@pytest.mark.parametrize(
    "radius",
    [
        True,
        "300",
    ],
)
def test_circular_fit_region_rejects_non_numeric_radius(
    radius,
) -> None:
    config = _base_config()
    config["fit"] = {
        "region": {
            "mode": "circular",
            "radius": radius,
        },
    }

    with pytest.raises(
        TypeError,
        match="finite positive number",
    ):
        resolve_config(
            config
        )


def test_fit_region_rejects_unknown_entries() -> None:
    config = _base_config()
    config["fit"] = {
        "region": {
            "mode": "circular",
            "radius": 300,
            "axis_ratio": 0.8,
        },
    }

    with pytest.raises(
        ValueError,
        match="Unsupported fit.region entries",
    ):
        resolve_config(
            config
        )


def test_resolving_config_does_not_modify_default_region() -> None:
    expected = deepcopy(
        DEFAULT_CONFIG["fit"]["region"]
    )

    config = _base_config()
    config["fit"] = {
        "region": {
            "mode": "circular",
            "radius": 300,
        },
    }

    resolve_config(
        config
    )

    assert DEFAULT_CONFIG["fit"]["region"] == expected
