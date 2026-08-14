"""Tests for multi-observation residual-map diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import baymorph.pipeline.residual_maps as residual_maps


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
            "image_A": {
                "config_path": "image_a.yaml",
            },
            "repeat_02": {
                "config_path": "repeat_02.yaml",
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


def _image_products(
    shape,
):
    return SimpleNamespace(
        cutout=np.ones(
            shape,
            dtype=float,
        ),
        prepared_data=np.full(
            shape,
            2.0,
            dtype=float,
        ),
        noise=np.full(
            shape,
            0.5,
            dtype=float,
        ),
        segmentation=np.zeros(
            shape,
            dtype=int,
        ),
        fitting_mask=np.zeros(
            shape,
            dtype=bool,
        ),
        target_segment=np.ones(
            shape,
            dtype=bool,
        ),
        original_image=np.full(
            shape,
            3.0,
            dtype=float,
        ),
        cutout_bounds={
            "xmin": 0,
            "xmax": shape[1],
            "ymin": 0,
            "ymax": shape[0],
        },
    )


def test_multi_observation_residual_maps_use_native_products_and_paths(
    tmp_path,
    monkeypatch,
):
    observation_configs = {
        "image_A": {
            "object": {
                "name": "target_a",
            },
        },
        "repeat_02": {
            "object": {
                "name": "target_b",
            },
        },
    }

    monkeypatch.setattr(
        residual_maps,
        "resolve_multi_observation_config",
        lambda config, base_dir: (
            observation_configs,
            object(),
        ),
    )

    image_products = {
        id(
            observation_configs[
                "image_A"
            ]
        ): _image_products(
            (7, 9)
        ),
        id(
            observation_configs[
                "repeat_02"
            ]
        ): _image_products(
            (11, 13)
        ),
    }

    monkeypatch.setattr(
        residual_maps,
        "load_image_preparation_products",
        lambda config: image_products[
            id(
                config
            )
        ],
    )

    psfs = {
        id(
            observation_configs[
                "image_A"
            ]
        ): np.ones(
            (3, 3),
            dtype=float,
        ),
        id(
            observation_configs[
                "repeat_02"
            ]
        ): np.ones(
            (5, 5),
            dtype=float,
        ),
    }

    monkeypatch.setattr(
        residual_maps,
        "load_empirical_psf",
        lambda config: psfs[
            id(
                config
            )
        ],
    )

    monkeypatch.setattr(
        residual_maps,
        "_diagnostic_title",
        lambda config, model_name: (
            f"{config['object']['name']} | {model_name}"
        ),
    )

    observation_fits = {
        "image_A": SimpleNamespace(
            best_model_image=np.full(
                (7, 9),
                1.5,
                dtype=float,
            ),
            noise=np.full(
                (7, 9),
                0.2,
                dtype=float,
            ),
        ),
        "repeat_02": SimpleNamespace(
            best_model_image=np.full(
                (11, 13),
                1.7,
                dtype=float,
            ),
            noise=np.full(
                (11, 13),
                0.3,
                dtype=float,
            ),
        ),
    }

    fit = SimpleNamespace(
        model_name="single_sersic",
        observations=observation_fits,
    )

    calls = []
    figures = []

    def fake_plot_pipeline_fit_maps(
        **kwargs,
    ):
        figure = _Figure()

        calls.append(
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
        residual_maps,
        "plot_pipeline_fit_maps",
        fake_plot_pipeline_fit_maps,
    )

    closed = []

    monkeypatch.setattr(
        residual_maps.plt,
        "close",
        lambda figure: closed.append(
            figure
        ),
    )

    residual_maps.run_residual_map_diagnostics(
        _config(
            tmp_path
        ),
        fit,
    )

    assert len(
        calls
    ) == 2

    np.testing.assert_allclose(
        calls[0][
            "model"
        ],
        observation_fits[
            "image_A"
        ].best_model_image,
    )

    np.testing.assert_allclose(
        calls[1][
            "model"
        ],
        observation_fits[
            "repeat_02"
        ].best_model_image,
    )

    np.testing.assert_allclose(
        calls[0][
            "noise"
        ],
        observation_fits[
            "image_A"
        ].noise,
    )

    np.testing.assert_allclose(
        calls[1][
            "noise"
        ],
        observation_fits[
            "repeat_02"
        ].noise,
    )

    assert calls[0][
        "prepared_cutout"
    ].shape == (
        7,
        9,
    )

    assert calls[1][
        "prepared_cutout"
    ].shape == (
        11,
        13,
    )

    assert calls[0][
        "psf"
    ].shape == (
        3,
        3,
    )

    assert calls[1][
        "psf"
    ].shape == (
        5,
        5,
    )

    assert calls[0][
        "title"
    ].endswith(
        "observation: image_A"
    )

    assert calls[1][
        "title"
    ].endswith(
        "observation: repeat_02"
    )

    assert figures[0].saved_path == (
        tmp_path
        / "runs"
        / "joint_case"
        / "fits"
        / "single_sersic"
        / "observations"
        / "image_A"
        / "data_model_residual.pdf"
    )

    assert figures[1].saved_path == (
        tmp_path
        / "runs"
        / "joint_case"
        / "fits"
        / "single_sersic"
        / "observations"
        / "repeat_02"
        / "data_model_residual.pdf"
    )

    assert figures[0].saved_kwargs == {
        "bbox_inches": "tight",
    }

    assert figures[1].saved_kwargs == {
        "bbox_inches": "tight",
    }

    assert closed == figures


def test_multi_observation_residual_maps_require_every_observation_product(
    tmp_path,
    monkeypatch,
):
    observation_configs = {
        "image_A": {
            "object": {
                "name": "target_a",
            },
        },
        "repeat_02": {
            "object": {
                "name": "target_b",
            },
        },
    }

    monkeypatch.setattr(
        residual_maps,
        "resolve_multi_observation_config",
        lambda config, base_dir: (
            observation_configs,
            object(),
        ),
    )

    fit = SimpleNamespace(
        model_name="single_sersic",
        observations={
            "image_A": object(),
        },
    )

    try:
        residual_maps.run_residual_map_diagnostics(
            _config(
                tmp_path
            ),
            fit,
        )
    except ValueError as exc:
        assert (
            "repeat_02"
            in str(
                exc
            )
        )
    else:
        raise AssertionError(
            "Expected missing observation products to raise ValueError."
        )
