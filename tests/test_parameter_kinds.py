"""Tests for the kinds of the parameters of the built-in models (AD-049)."""

from __future__ import annotations

import pytest

from baymorph.models.labels import (
    MODEL_PARAMETER_LABELS,
    PARAMETER_KIND_NAMES,
    PARAMETER_KINDS,
    parameter_kind,
    parameter_names,
)


def test_every_parameter_of_every_model_has_a_kind() -> None:
    """The kinds must cover the parameters of the ten models, in order."""
    assert set(PARAMETER_KINDS) == set(MODEL_PARAMETER_LABELS)
    assert len(PARAMETER_KINDS) == 10

    for model_name, kinds in PARAMETER_KINDS.items():
        assert list(kinds) == parameter_names(model_name), model_name
        assert set(kinds.values()) <= set(PARAMETER_KIND_NAMES), model_name


def test_kinds_follow_the_meaning_of_the_parameters() -> None:
    """Fluxes, centres and position angles must carry their kinds."""
    for model_name, kinds in PARAMETER_KINDS.items():
        assert kinds["x0"] == "position"
        assert kinds["y0"] == "position"

        for name, kind in kinds.items():
            if name.startswith("flux"):
                assert kind == "flux", (model_name, name)
            if name.startswith("theta"):
                assert kind == "angle", (model_name, name)
            if name.startswith(("re", "h_")) or name == "rb":
                assert kind == "length", (model_name, name)
            if name in ("n", "q") or name.startswith(("n_", "q_")):
                assert kind == "dimensionless", (model_name, name)

        positions = [name for name, kind in kinds.items() if kind == "position"]
        assert positions == ["x0", "y0"], model_name


def test_parameter_kind_reads_one_entry_and_rejects_unknown_names() -> None:
    """Unknown models and parameters must be rejected, not defaulted."""
    assert parameter_kind("single_sersic", "re") == "length"
    assert parameter_kind("bulge_disk_spiral", "phase_spiral") == "angle"
    assert parameter_kind("core_sersic", "rb") == "length"

    with pytest.raises(ValueError, match="Unknown model"):
        parameter_kind("triple_sersic", "re")

    with pytest.raises(ValueError, match="has no parameter"):
        parameter_kind("single_sersic", "c0")
