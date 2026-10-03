"""Keep the documentation synchronized with the configuration and the CLI."""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path
from urllib.parse import unquote

from baymorph.cli import _build_parser
from baymorph.models import get_model, registered_models
from baymorph.pipeline.config import DEFAULT_CONFIG


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_DIRECTORY = PROJECT_ROOT / "docs" / "reference"

MODELS_DIRECTORY = PROJECT_ROOT / "docs" / "models"

THEORY_DIRECTORY = PROJECT_ROOT / "docs" / "theory"

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
    text = (REFERENCE_DIRECTORY / "CONFIGURATION.md").read_text(encoding="utf-8")

    missing = [
        path
        for path in _leaf_paths(DEFAULT_CONFIG)
        if f"`{path}`" not in text
    ]

    assert missing == []


def test_configuration_reference_lists_every_model_parameter() -> None:
    text = (REFERENCE_DIRECTORY / "CONFIGURATION.md").read_text(encoding="utf-8")

    missing = [
        f"{model}.{parameter}"
        for model in registered_models()
        for parameter in (model, *get_model(model).parameter_names)
        if f"`{parameter}`" not in text
    ]

    assert missing == []


def test_cli_page_lists_every_command_and_option() -> None:
    text = (REFERENCE_DIRECTORY / "CLI.md").read_text(encoding="utf-8")
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


def test_every_registered_model_has_a_page_with_its_parameters() -> None:
    index = (MODELS_DIRECTORY / "index.md").read_text(encoding="utf-8")

    missing = []

    for model in registered_models():
        page = MODELS_DIRECTORY / f"{model.upper()}.md"

        if not page.is_file() or f"`{model}`" not in index:
            missing.append(model)
            continue

        text = page.read_text(encoding="utf-8")
        missing.extend(
            f"{model}.{parameter}"
            for parameter in get_model(model).parameter_names
            if f"`{parameter}`" not in text
        )

    assert missing == []


FIGURE_REFERENCE = re.compile(
    r"\(figures/([a-z_]+)-(light|dark)\.png#gh-(light|dark)-mode-only\)"
)


def _figure_script():
    """Load the generator of the model figures without running it."""
    path = PROJECT_ROOT / "scripts" / "make_model_figures.py"
    spec = importlib.util.spec_from_file_location("make_model_figures", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_model_figures_are_generated_referenced_and_themed() -> None:
    referenced = set()

    for page in sorted(MODELS_DIRECTORY.glob("*.md")):
        text = page.read_text(encoding="utf-8")

        for name, variant, mode in FIGURE_REFERENCE.findall(text):
            assert variant == mode, f"{page.name}: {name}-{variant} in {mode} mode"
            referenced.add(f"{name}-{variant}.png")

    script = _figure_script()
    generated = {
        f"{name}-{theme.name}.png"
        for name in script.FIGURES
        for theme in script.THEMES
    }
    present = {path.name for path in (MODELS_DIRECTORY / "figures").glob("*.png")}

    assert present == generated
    assert referenced == generated


def test_theory_figures_are_generated_referenced_and_themed() -> None:
    referenced = set()

    for page in sorted(THEORY_DIRECTORY.glob("*.md")):
        text = page.read_text(encoding="utf-8")

        for name, variant, mode in FIGURE_REFERENCE.findall(text):
            assert variant == mode, f"{page.name}: {name}-{variant} in {mode} mode"
            referenced.add(f"{name}-{variant}.png")

    script = _figure_script()
    generated = {
        f"{name}-{theme.name}.png"
        for name in script.THEORY_FIGURES
        for theme in script.THEMES
    }
    present = {path.name for path in (THEORY_DIRECTORY / "figures").glob("*.png")}

    assert present == generated
    assert referenced == generated


REFERENCE_USE = re.compile(r"\]\[([a-z0-9-]+)\]")

REFERENCE_DEFINITION = re.compile(r"^\[([a-z0-9-]+)\]: (\S+)$", re.MULTILINE)

ADS_ABSTRACT = re.compile(r"https://ui\.adsabs\.harvard\.edu/abs/([^/]+)/abstract")

BIBCODE_LENGTH = 19


def test_model_and_theory_pages_cite_their_sources_with_ads_links() -> None:
    problems = []
    sources: dict[str, str] = {}

    pages = (
        *sorted(MODELS_DIRECTORY.glob("*.md")),
        *sorted(THEORY_DIRECTORY.glob("*.md")),
    )

    for page in pages:
        if page.name == "index.md":
            continue

        text = page.read_text(encoding="utf-8")
        body, heading, references = text.partition("\n## References\n")

        if not heading:
            problems.append(f"{page.name}: no References section")
            continue

        definitions = dict(REFERENCE_DEFINITION.findall(references))
        listed = set(REFERENCE_USE.findall(references))
        cited = set(REFERENCE_USE.findall(body))

        if not definitions:
            problems.append(f"{page.name}: no references")

        problems.extend(
            f"{page.name}: [{label}] is not listed"
            for label in sorted(cited - listed)
        )
        problems.extend(
            f"{page.name}: [{label}] has no link"
            for label in sorted(listed - set(definitions))
        )
        problems.extend(
            f"{page.name}: [{label}] is never listed"
            for label in sorted(set(definitions) - listed)
        )

        for label, url in definitions.items():
            match = ADS_ABSTRACT.fullmatch(url)

            if match is None or len(unquote(match.group(1))) != BIBCODE_LENGTH:
                problems.append(f"{page.name}: [{label}] is not an ADS record: {url}")

            if sources.setdefault(label, url) != url:
                problems.append(
                    f"{page.name}: [{label}] links {url}, elsewhere {sources[label]}"
                )

    assert problems == []
