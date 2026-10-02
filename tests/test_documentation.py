"""Keep the user guide synchronized with the configuration and the CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from baymorph.cli import _build_parser
from baymorph.models import get_model, registered_models
from baymorph.pipeline.config import DEFAULT_CONFIG


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GUIDE_DIRECTORY = PROJECT_ROOT / "docs" / "guide"

# The model priors are checked by model and parameter below; the background
# priors are documented by parameter.
MODEL_PRIORS = "fit.priors"
BACKGROUND_PRIORS = "fit.background.priors"


def _leaf_paths(node, prefix: str = ""):
    """Yield the dotted path of every documented configuration leaf."""
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else str(key)

        if path == MODEL_PRIORS:
            continue

        if prefix == BACKGROUND_PRIORS or not isinstance(value, dict) or not value:
            yield path
            continue

        yield from _leaf_paths(value, path)


def test_configuration_reference_lists_every_default_key() -> None:
    text = (GUIDE_DIRECTORY / "CONFIGURATION.md").read_text(encoding="utf-8")

    missing = [
        path
        for path in _leaf_paths(DEFAULT_CONFIG)
        if f"`{path}`" not in text
    ]

    assert missing == []


def test_configuration_reference_lists_every_model_parameter() -> None:
    text = (GUIDE_DIRECTORY / "CONFIGURATION.md").read_text(encoding="utf-8")

    missing = [
        f"{model}.{parameter}"
        for model in registered_models()
        for parameter in (model, *get_model(model).parameter_names)
        if f"`{parameter}`" not in text
    ]

    assert missing == []


def test_cli_page_lists_every_command_and_option() -> None:
    text = (GUIDE_DIRECTORY / "CLI.md").read_text(encoding="utf-8")
    parser = _build_parser()

    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )

    missing = []

    for command, subparser in subparsers.choices.items():
        if f"`baymorph {command}`" not in text:
            missing.append(command)

        for action in subparser._actions:
            names = action.option_strings or [action.dest]

            for name in names:
                if name in {"-h", "--help"}:
                    continue

                if f"`{name}" not in text:
                    missing.append(f"{command} {name}")

    assert missing == []
