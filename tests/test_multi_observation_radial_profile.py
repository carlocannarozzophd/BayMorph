"""Tests for multi-observation elliptical radial diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import baymorph.pipeline.radial_profile as radial_pipeline


class _Figure:
    """Minimal figure test double."""

    def __init__(self) -> None:
        self.saved_path = None

    def savefig(
        self,
        path,
        **kwargs,
    ) -> None:
        self.saved_path = path
        self.saved_kwargs = kwargs


class _Representation:
    """Simple galaxy representation for radial-profile tests."""

    def __init__(
        self,
        amplitude,
    ) -> None:
        self.amplitude = float(
            amplitude
        )

    def render_components(
        self,
        shape,
    ):
        return {
            "galaxy": np.full(
                shape,
                self.amplitude,
                dtype=float,
            ),
        }


class _RepresentationFactory:
    """Observation-aware representation test double."""

    def __init__(
        self,
    ) -> None:
        self.calls = []

    def build(
        self,
        observation_name,
        parameters,
    ):
        self.calls.append(
            (
                observation_name,
                dict(
                    parameters
                ),
            )
        )

        return _Representation(
            parameters[
                "amplitude"
            ]
        )


class _Problem:
    """Minimal joint inference problem used by the diagnostic test."""

    parameter_names = [
        "amplitude",
    ]

    def __init__(
        self,
    ) -> None:
        self.representation_factory = (
            _RepresentationFactory()
        )

    def routed_parameters(
        self,
        sample,
    ):
        amplitude = float(
            np.asarray(
                sample
            )[
                0
            ]
        )

        return {
            "image_A": {
                "amplitude": amplitude,
            },
            "repeat_02": {
                "amplitude": 2.0 * amplitude,
            },
        }


def _image_products(
    shape,
):
    return SimpleNamespace(
        prepared_data=np.full(
            shape,
            10.0,
            dtype=float,
        ),
        fitting_mask=np.zeros(
            shape,
            dtype=bool,
        ),
    )


def _config(
    tmp_path,
):
    return {
        "observations": {
            "image_A": {
                "config_path": "image_a.yaml",
            },
            "repeat_02": {
                "config_path": "repeat_02.yaml",
            },
        },
        "fit": {
            "models": [
                "single_sersic",
            ],
        },
        "outputs": {
            "root": str(
                tmp_path
                / "runs"
            ),
            "analysis_name": "joint_case",
        },
        "diagnostics": {
            "radial_profile": {
                "binning": "fixed",
                "dr": 2.0,
            },
            "posterior_profile_bands": True,
        },
    }


def test_multi_observation_radial_profiles_use_detector_geometries(
    tmp_path,
    monkeypatch,
):
    problem = _Problem()

    context = SimpleNamespace(
        problem=problem,
        image_products={
            "image_A": _image_products(
                (
                    25,
                    27,
                )
            ),
            "repeat_02": _image_products(
                (
                    31,
                    35,
                )
            ),
        },
    )

    monkeypatch.setattr(
        radial_pipeline,
        "_build_multiband_context",
        lambda config, base_dir: context,
    )

    observation_fits = {
        "image_A": SimpleNamespace(
            parameters={
                "x0": 12.0,
                "y0": 11.0,
                "q": 0.45,
                "theta": 0.30,
                "sky": 1.0,
            },
        ),
        "repeat_02": SimpleNamespace(
            parameters={
                "x0": 17.0,
                "y0": 14.0,
                "q": 0.72,
                "theta": -0.55,
                "sky": 2.0,
            },
        ),
    }

    fit = SimpleNamespace(
        model_name="single_sersic",
        best_parameters={
            "amplitude": 3.0,
        },
        observations=observation_fits,
        posterior=SimpleNamespace(
            posterior_samples=np.asarray(
                [
                    [
                        2.5,
                    ],
                    [
                        3.0,
                    ],
                    [
                        3.5,
                    ],
                ],
                dtype=float,
            ),
        ),
    )

    posterior_calls = []

    def fake_compute_posterior_profile_bands(
        *,
        posterior,
        model_factory,
        image_shape,
        radii,
        center_x,
        center_y,
        axis_ratio,
        position_angle,
        mask,
    ):
        del mask

        model = model_factory(
            posterior[
                0
            ]
        )

        components = model.render_components(
            image_shape
        )

        assert "galaxy" in components

        posterior_calls.append(
            {
                "shape": image_shape,
                "center_x": center_x,
                "center_y": center_y,
                "axis_ratio": axis_ratio,
                "position_angle": position_angle,
                "radii": np.asarray(
                    radii
                ),
            }
        )

        return None

    monkeypatch.setattr(
        radial_pipeline,
        "compute_posterior_profile_bands",
        fake_compute_posterior_profile_bands,
    )

    plot_calls = []
    figures = []

    def fake_plot_radial_profile_diagnostics(
        **kwargs,
    ):
        figure = _Figure()

        plot_calls.append(
            kwargs
        )
        figures.append(
            figure
        )

        return (
            figure,
            object(),
        )

    monkeypatch.setattr(
        radial_pipeline,
        "plot_radial_profile_diagnostics",
        fake_plot_radial_profile_diagnostics,
    )

    monkeypatch.setattr(
        radial_pipeline.plt,
        "close",
        lambda figure: None,
    )

    result = radial_pipeline.run_radial_profile_diagnostics(
        _config(
            tmp_path
        ),
        fit,
    )

    assert set(
        result
    ) == {
        "image_A",
        "repeat_02",
    }

    assert len(
        posterior_calls
    ) == 2

    assert posterior_calls[
        0
    ][
        "axis_ratio"
    ] == pytest.approx(
        0.45
    )

    assert posterior_calls[
        0
    ][
        "position_angle"
    ] == pytest.approx(
        0.30
    )

    assert posterior_calls[
        1
    ][
        "axis_ratio"
    ] == pytest.approx(
        0.72
    )

    assert posterior_calls[
        1
    ][
        "position_angle"
    ] == pytest.approx(
        -0.55
    )

    assert plot_calls[
        0
    ][
        "radius"
    ].shape != (
        0,
    )

    assert plot_calls[
        0
    ][
        "profile_data_mean"
    ].shape == plot_calls[
        0
    ][
        "profile_data"
    ].shape

    assert plot_calls[
        0
    ][
        "profile_data_p16"
    ].shape == plot_calls[
        0
    ][
        "profile_data"
    ].shape

    assert plot_calls[
        0
    ][
        "profile_data_p84"
    ].shape == plot_calls[
        0
    ][
        "profile_data"
    ].shape

    assert figures[
        0
    ].saved_path == (
        tmp_path
        / "runs"
        / "joint_case"
        / "fits"
        / "single_sersic"
        / "observations"
        / "image_A"
        / "radial_profile.pdf"
    )

    assert figures[
        1
    ].saved_path == (
        tmp_path
        / "runs"
        / "joint_case"
        / "fits"
        / "single_sersic"
        / "observations"
        / "repeat_02"
        / "radial_profile.pdf"
    )

    built_observations = [
        call[
            0
        ]
        for call in problem.representation_factory.calls
    ]

    assert "image_A" in built_observations
    assert "repeat_02" in built_observations


def test_multi_observation_radial_profiles_validate_products_before_build(
    tmp_path,
    monkeypatch,
):
    fit = SimpleNamespace(
        model_name="single_sersic",
        observations={
            "image_A": object(),
        },
    )

    monkeypatch.setattr(
        radial_pipeline,
        "_build_multiband_context",
        lambda config, base_dir: pytest.fail(
            "Context must not be built before validating observation products."
        ),
    )

    with pytest.raises(
        ValueError,
        match="repeat_02",
    ):
        radial_pipeline.run_radial_profile_diagnostics(
            _config(
                tmp_path
            ),
            fit,
        )
