from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def object_output_dir(config: dict[str, Any]) -> Path:
    root = Path(config["outputs"]["root"])
    object_name = config["object"]["name"]
    return root / object_name


def make_output_tree(config: dict[str, Any]) -> Path:
    base = object_output_dir(config)

    subdirs = [
        base,
        base / "image_preparation",
        base / "psf",
        base / "fits",
        base / "fits" / "single_sersic",
        base / "fits" / "double_sersic",
        base / "model_comparison",
    ]

    for path in subdirs:
        path.mkdir(parents=True, exist_ok=True)

    return base


def save_resolved_config(config: dict[str, Any]) -> Path:
    base = make_output_tree(config)
    path = base / "config_resolved.yaml"

    with path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return path