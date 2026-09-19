"""Tests for simulation-truth markers in corner-plot diagnostics."""

from __future__ import annotations

import json
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import baymorph.pipeline.corner as corner_pipeline  # noqa: E402
from baymorph.plotting.corner import corner_plot  # noqa: E402

TRUTH_COLOR = "darkorange"


def _truth_line_count(
    figure,
) -> int:
    """Return the number of truth-marker lines drawn on one figure."""
    return sum(
        1
        for axis in figure.axes
        for line in axis.lines
        if line.get_color() == TRUTH_COLOR
    )


def _samples(
    n_params: int = 3,
) -> np.ndarray:
    """Return reproducible posterior samples."""
    rng = np.random.default_rng(
        7
    )

    return rng.normal(
        size=(
            n_params,
            400,
        )
    )


def test_corner_without_truths_draws_no_markers() -> None:
    """Omitting truths must leave the established plot unchanged."""
    figure, _, _ = corner_plot(
        _samples(),
        parameter_labels=[
            "flux",
            "re",
            "n",
        ],
    )

    assert _truth_line_count(
        figure
    ) == 0


def test_corner_truths_mark_diagonal_and_two_dimensional_panels() -> None:
    """Each known truth must be marked on its diagonal and 2D panels."""
    figure, _, _ = corner_plot(
        _samples(),
        parameter_labels=[
            "flux",
            "re",
            "n",
        ],
        truths=[
            0.1,
            -0.2,
            0.3,
        ],
    )

    # Three diagonal lines, plus two lines in each of the three 2D panels.
    assert _truth_line_count(
        figure
    ) == 9


def test_corner_truths_skip_unknown_values() -> None:
    """None and NaN entries must leave their parameter unmarked."""
    expected = 6

    for missing in (
        None,
        np.nan,
    ):
        figure, _, _ = corner_plot(
            _samples(),
            parameter_labels=[
                "flux",
                "re",
                "background_level",
            ],
            truths=[
                0.1,
                -0.2,
                missing,
            ],
        )

        assert _truth_line_count(
            figure
        ) == expected


def test_corner_truths_reject_wrong_length() -> None:
    """A truth vector must match the number of parameters."""
    with pytest.raises(
        ValueError,
        match="truths",
    ):
        corner_plot(
            _samples(),
            parameter_labels=[
                "flux",
                "re",
                "n",
            ],
            truths=[
                0.1,
            ],
        )


def test_corner_truths_do_not_change_the_posterior_rendering() -> None:
    """Truth markers must not alter medians or plotted densities."""
    labels = [
        "flux",
        "re",
        "n",
    ]
    samples = _samples()

    without_truths, _, _ = corner_plot(
        samples,
        parameter_labels=labels,
    )
    with_truths, _, _ = corner_plot(
        samples,
        parameter_labels=labels,
        truths=[
            0.1,
            -0.2,
            0.3,
        ],
    )

    def non_truth_lines(
        figure,
    ):
        return [
            line.get_xydata().tolist()
            for axis in figure.axes
            for line in axis.lines
            if line.get_color() != TRUTH_COLOR
        ]

    assert non_truth_lines(
        without_truths
    ) == non_truth_lines(
        with_truths
    )


def _write_truth(
    tmp_path,
    truth: dict,
) -> dict:
    """Write a persisted simulation truth and return its configuration."""
    config = {
        "object": {
            "name": "case",
        },
        "outputs": {
            "root": str(
                tmp_path
                / "runs"
            ),
        },
    }

    truth_path = (
        tmp_path
        / "runs"
        / "case"
        / "truth"
        / "truth.json"
    )
    truth_path.parent.mkdir(
        parents=True,
    )
    truth_path.write_text(
        json.dumps(
            {
                "truth": truth,
            }
        ),
        encoding="utf-8",
    )

    return config


def _patch_cutout_bounds(
    monkeypatch,
    *,
    xmin: int,
    ymin: int,
) -> None:
    """Make prepared products report one deliberate cutout origin."""
    monkeypatch.setattr(
        corner_pipeline,
        "load_image_preparation_products",
        lambda config: SimpleNamespace(
            cutout_bounds={
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmin + 50,
                "ymax": ymin + 50,
            },
        ),
    )


def test_truth_positions_are_transformed_into_cutout_coordinates(
    tmp_path,
    monkeypatch,
) -> None:
    """Truth x0/y0 must be shifted by the prepared cutout origin."""
    config = _write_truth(
        tmp_path,
        {
            "flux": 20000.0,
            "re": 6.0,
            "x0": 60.0,
            "y0": 45.0,
        },
    )

    _patch_cutout_bounds(
        monkeypatch,
        xmin=17,
        ymin=4,
    )

    values = corner_pipeline._simulation_truth_vector(
        config,
        [
            "flux",
            "re",
            "x0",
            "y0",
        ],
    )

    # Only positions are offset; fluxes and sizes are frame-independent.
    assert values == [
        20000.0,
        6.0,
        43.0,
        41.0,
    ]


def test_simulated_sky_is_never_marked_as_background_level(
    tmp_path,
    monkeypatch,
) -> None:
    """The removed sky plateau is not the residual background parameter."""
    config = _write_truth(
        tmp_path,
        {
            "flux": 20000.0,
            "sky": 20.0,
        },
    )

    _patch_cutout_bounds(
        monkeypatch,
        xmin=0,
        ymin=0,
    )

    values = corner_pipeline._simulation_truth_vector(
        config,
        [
            "flux",
            "background_level",
        ],
    )

    assert values == [
        20000.0,
        None,
    ]


def test_missing_truth_file_yields_no_markers(
    tmp_path,
    monkeypatch,
) -> None:
    """A real-data fit has no persisted truth and stays unmarked."""
    config = {
        "object": {
            "name": "observed",
        },
        "outputs": {
            "root": str(
                tmp_path
                / "runs"
            ),
        },
    }

    _patch_cutout_bounds(
        monkeypatch,
        xmin=0,
        ymin=0,
    )

    assert corner_pipeline._simulation_truth_vector(
        config,
        [
            "flux",
        ],
    ) is None


def test_truth_without_any_sampled_match_yields_no_markers(
    tmp_path,
    monkeypatch,
) -> None:
    """A truth sharing no name with the sampled parameters is ignored."""
    config = _write_truth(
        tmp_path,
        {
            "sky": 20.0,
        },
    )

    _patch_cutout_bounds(
        monkeypatch,
        xmin=0,
        ymin=0,
    )

    assert corner_pipeline._simulation_truth_vector(
        config,
        [
            "flux",
            "background_level",
        ],
    ) is None


def test_missing_cutout_bounds_fail_closed(
    tmp_path,
    monkeypatch,
) -> None:
    """Untransformable truth must raise rather than be silently misplaced."""
    config = _write_truth(
        tmp_path,
        {
            "x0": 60.0,
        },
    )

    monkeypatch.setattr(
        corner_pipeline,
        "load_image_preparation_products",
        lambda config: SimpleNamespace(
            cutout_bounds=None,
        ),
    )

    with pytest.raises(
        ValueError,
        match="cutout_bounds",
    ):
        corner_pipeline._simulation_truth_vector(
            config,
            [
                "x0",
            ],
        )
