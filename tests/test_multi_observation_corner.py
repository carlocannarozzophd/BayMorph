"""Tests for multi-observation corner diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import baymorph.pipeline.corner as corner_pipeline


class _Figure:
    """Minimal figure test double."""

    def __init__(self) -> None:
        self.saved_path = None
        self.saved_kwargs = None

    def savefig(
        self,
        path,
        **kwargs,
    ) -> None:
        self.saved_path = path
        self.saved_kwargs = kwargs


def _config(
    tmp_path,
) -> dict:
    return {
        "observations": {
            "observation_a": {
                "config_path": "a.yaml",
            },
            "visit_02": {
                "config_path": "b.yaml",
            },
        },
        "outputs": {
            "root": str(
                tmp_path
                / "runs"
            ),
            "analysis_name": "joint_case",
        },
        "fit": {
            "models": [
                "single_sersic",
            ],
        },
    }


def test_multi_observation_corner_uses_joint_posterior_and_joint_path(
    tmp_path,
    monkeypatch,
):
    parameter_names = [
        "flux__observation_a",
        "re_arcsec__visit_02",
        "ra",
    ]

    posterior = SimpleNamespace(
        posterior_samples=np.asarray(
            [
                [
                    10.0,
                    0.8,
                    150.0,
                ],
                [
                    11.0,
                    0.9,
                    150.0001,
                ],
                [
                    12.0,
                    1.0,
                    149.9999,
                ],
            ],
            dtype=float,
        ),
        parameter_names=parameter_names,
    )

    fit = SimpleNamespace(
        model_name="single_sersic",
        posterior=posterior,
    )

    captured = {}
    figure = _Figure()

    def fake_corner_plot(
        *,
        parameters,
        parameter_labels,
        sigma,
    ):
        captured[
            "parameters"
        ] = parameters
        captured[
            "parameter_labels"
        ] = parameter_labels
        captured[
            "sigma"
        ] = sigma

        return (
            figure,
            None,
            None,
        )

    monkeypatch.setattr(
        corner_pipeline,
        "corner_plot",
        fake_corner_plot,
    )

    monkeypatch.setattr(
        corner_pipeline.plt,
        "close",
        lambda fig: captured.setdefault(
            "closed",
            fig,
        ),
    )

    corner_pipeline.run_corner_diagnostics(
        _config(
            tmp_path
        ),
        fit,
    )

    assert captured[
        "parameter_labels"
    ] == parameter_names

    np.testing.assert_allclose(
        captured[
            "parameters"
        ],
        posterior.posterior_samples.T,
    )

    assert captured[
        "sigma"
    ] == [
        1,
        2,
        3,
    ]

    assert figure.saved_path == (
        tmp_path
        / "runs"
        / "joint_case"
        / "fits"
        / "single_sersic"
        / "corner.pdf"
    )

    assert figure.saved_kwargs == {
        "bbox_inches": "tight",
    }

    assert captured[
        "closed"
    ] is figure

    assert figure.saved_path.parent.is_dir()


def test_multi_observation_corner_accepts_mapping_of_joint_results(
    tmp_path,
    monkeypatch,
):
    posterior = SimpleNamespace(
        posterior_samples=np.asarray(
            [
                [
                    1.0,
                ],
                [
                    2.0,
                ],
                [
                    3.0,
                ],
            ],
            dtype=float,
        ),
        parameter_names=[
            "shared_parameter",
        ],
    )

    fit = SimpleNamespace(
        model_name="single_sersic",
        posterior=posterior,
    )

    calls = []

    monkeypatch.setattr(
        corner_pipeline,
        "corner_plot",
        lambda **kwargs: (
            calls.append(
                kwargs
            )
            or (
                _Figure(),
                None,
                None,
            )
        ),
    )

    monkeypatch.setattr(
        corner_pipeline.plt,
        "close",
        lambda fig: None,
    )

    corner_pipeline.run_corner_diagnostics(
        _config(
            tmp_path
        ),
        {
            "single_sersic": fit,
        },
    )

    assert len(
        calls
    ) == 1
