"""Helpers shared by the corner, residual-map, and radial diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from baymorph.pipeline.config import (
    load_config,
    load_resolved_config,
    resolve_config,
)


def resolve_diagnostic_config(
    config: str | Path | Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    bool,
    Path,
]:
    """Resolve a diagnostic configuration and identify its workflow mode.

    Parameters
    ----------
    config : str, pathlib.Path, or mapping
        Single-observation configuration path or mapping, or a joint
        multi-observation configuration containing ``observations``.

    Returns
    -------
    tuple
        ``(resolved_config, is_multi_observation, base_dir)`` where the
        base directory resolves relative child paths of a joint
        configuration. Single-observation configurations are resolved
        against BayMorph defaults; joint configurations are returned raw.
    """
    if isinstance(
        config,
        (str, Path),
    ):
        path = Path(
            config
        )

        raw_config = load_config(
            path
        )

        if "observations" in raw_config:
            return (
                raw_config,
                True,
                path.parent,
            )

        return (
            load_resolved_config(
                path
            ),
            False,
            path.parent,
        )

    if isinstance(
        config,
        Mapping,
    ):
        raw_config = dict(
            config
        )

        if "observations" in raw_config:
            return (
                raw_config,
                True,
                Path.cwd(),
            )

        return (
            resolve_config(
                raw_config
            ),
            False,
            Path.cwd(),
        )

    raise TypeError(
        "config must be a path or a mapping."
    )


def posterior_samples(posterior):
    """Return equal-weight posterior samples from a result-like object.

    Accepts an ``InferenceResult`` (through its ``posterior`` attribute), a
    ``PosteriorSamples`` container, or a plain array-like of samples. The
    equal-weight samples are a derived visualization product; the weighted
    samples remain the statistical source of truth.
    """
    nested_posterior = getattr(
        posterior,
        "posterior",
        None,
    )

    if (
        nested_posterior is not None
        and hasattr(
            nested_posterior,
            "equal_weight_samples",
        )
    ):
        return nested_posterior.equal_weight_samples

    if hasattr(
        posterior,
        "equal_weight_samples",
    ):
        return posterior.equal_weight_samples

    if hasattr(
        posterior,
        "posterior_samples",
    ):
        return posterior.posterior_samples

    if hasattr(
        posterior,
        "samples",
    ):
        return posterior.samples

    return posterior


def object_label(
    config: Mapping[str, Any],
) -> str | None:
    """Return a human-readable object label for diagnostic figures."""
    object_cfg = config.get(
        "object",
        {},
    )

    if not isinstance(
        object_cfg,
        Mapping,
    ):
        return None

    name = object_cfg.get(
        "name"
    )

    if name is None:
        return None

    label = str(
        name
    ).strip()

    if not label:
        return None

    return label.replace(
        "_",
        " ",
    )


def diagnostic_title(
    config: Mapping[str, Any],
    model_name: str,
) -> str:
    """Return the model part of a diagnostic title.

    A title configured under ``diagnostics.titles.<model>`` is returned
    verbatim; otherwise the registered model name is rendered as words.
    """
    diagnostics_cfg = config.get(
        "diagnostics",
        {},
    )

    titles = (
        diagnostics_cfg.get(
            "titles",
            {},
        )
        if isinstance(
            diagnostics_cfg,
            Mapping,
        )
        else {}
    )

    if isinstance(
        titles,
        Mapping,
    ) and model_name in titles:
        return str(
            titles[
                model_name
            ]
        )

    return (
        model_name
        .replace(
            "_",
            " ",
        )
        .title()
        .replace(
            "Sersic",
            "Sérsic",
        )
    )


def fit_diagnostic_title(
    config: Mapping[str, Any],
    model_name: str,
) -> str:
    """Return an object- and model-aware diagnostic title.

    A configured ``diagnostics.titles.<model>`` entry is a complete
    override; otherwise the object label and the model label are joined.
    """
    diagnostics_cfg = config.get(
        "diagnostics",
        {},
    )

    if isinstance(
        diagnostics_cfg,
        Mapping,
    ):
        configured_titles = diagnostics_cfg.get(
            "titles",
            {},
        )

        if (
            isinstance(
                configured_titles,
                Mapping,
            )
            and model_name in configured_titles
        ):
            return str(
                configured_titles[
                    model_name
                ]
            )

    model_label = diagnostic_title(
        config,
        model_name,
    )

    label = object_label(
        config
    )

    if label is None:
        return model_label

    return (
        f"{label} | {model_label}"
    )
