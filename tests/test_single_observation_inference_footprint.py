"""Tests for persisted single-observation inference footprints."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import baymorph.pipeline.real_fit as real_fit
import baymorph.pipeline.residual_maps as residual_maps
from baymorph.pipeline.config import resolve_config
from baymorph.results import RealFitProducts


class _Figure:
    """Minimal figure test double."""

    def __init__(self) -> None:
        self.saved_path = None

    def savefig(
        self,
        path,
        **kwargs,
    ) -> None:
        del kwargs
        self.saved_path = path


def _config(
    tmp_path,
) -> dict:
    """Return one resolved single-observation configuration."""
    return resolve_config(
        {
            "object": {
                "name": "target",
            },
            "data": {
                "image_path": "image.fits",
            },
            "outputs": {
                "root": str(
                    tmp_path
                    / "runs"
                ),
            },
            "fit": {
                "models": [
                    "single_sersic",
                ],
                "region": {
                    "mode": "circular",
                    "radius": 1.5,
                },
            },
        }
    )


def test_real_fit_round_trip_preserves_inference_footprint(
    tmp_path,
    monkeypatch,
) -> None:
    """Persisted fits must preserve bounds, mask, and local noise."""
    config = _config(
        tmp_path
    )

    full_noise = np.arange(
        42,
        dtype=float,
    ).reshape(
        6,
        7,
    ) + 1.0

    monkeypatch.setattr(
        real_fit,
        "load_image_preparation_products",
        lambda config: SimpleNamespace(
            noise=full_noise,
        ),
    )

    bounds = {
        "xmin": 2,
        "xmax": 5,
        "ymin": 1,
        "ymax": 4,
    }

    inference_mask = np.array(
        [
            [True, False, True],
            [False, False, False],
            [True, False, True],
        ],
        dtype=bool,
    )

    products = RealFitProducts(
        posterior=None,
        best_parameters={
            "flux": 10.0,
        },
        best_model_image=np.zeros(
            (3, 3),
            dtype=float,
        ),
        residual=np.zeros(
            (3, 3),
            dtype=float,
        ),
        norm_residual=np.zeros(
            (3, 3),
            dtype=float,
        ),
        chi2=1.0,
        ndof=3,
        reduced_chi2=1.0 / 3.0,
        model_name="single_sersic",
        noise=full_noise[
            1:4,
            2:5,
        ],
        inference_bounds=bounds,
        inference_mask=inference_mask,
    )

    real_fit.save_real_fit_products(
        config,
        "single_sersic",
        products,
    )

    loaded = real_fit.load_real_fit_products(
        config,
        "single_sersic",
    )

    assert loaded.inference_bounds == bounds

    np.testing.assert_array_equal(
        loaded.inference_mask,
        inference_mask,
    )

    np.testing.assert_array_equal(
        loaded.noise,
        full_noise[
            1:4,
            2:5,
        ],
    )


def test_single_observation_residual_map_uses_inference_footprint(
    tmp_path,
    monkeypatch,
) -> None:
    """Residual diagnostics must use exactly the fitted image footprint."""
    shape = (
        6,
        7,
    )

    image_products = SimpleNamespace(
        cutout=np.arange(
            np.prod(
                shape
            ),
            dtype=float,
        ).reshape(
            shape
        ),
        prepared_data=np.arange(
            np.prod(
                shape
            ),
            dtype=float,
        ).reshape(
            shape
        ) + 100.0,
        noise=np.ones(
            shape,
            dtype=float,
        ),
        segmentation=np.arange(
            np.prod(
                shape
            ),
            dtype=int,
        ).reshape(
            shape
        ),
        fitting_mask=np.zeros(
            shape,
            dtype=bool,
        ),
        target_segment=np.ones(
            shape,
            dtype=bool,
        ),
        original_image=np.zeros(
            (20, 30),
            dtype=float,
        ),
        cutout_bounds={
            "xmin": 100,
            "xmax": 107,
            "ymin": 200,
            "ymax": 206,
        },
    )

    bounds = {
        "xmin": 2,
        "xmax": 5,
        "ymin": 1,
        "ymax": 4,
    }

    inference_mask = np.array(
        [
            [True, False, True],
            [False, False, False],
            [True, False, True],
        ],
        dtype=bool,
    )

    fit = RealFitProducts(
        posterior=None,
        best_parameters={},
        best_model_image=np.zeros(
            (3, 3),
            dtype=float,
        ),
        residual=np.zeros(
            (3, 3),
            dtype=float,
        ),
        norm_residual=np.zeros(
            (3, 3),
            dtype=float,
        ),
        chi2=1.0,
        ndof=1,
        reduced_chi2=1.0,
        model_name="single_sersic",
        noise=np.full(
            (3, 3),
            2.0,
            dtype=float,
        ),
        inference_bounds=bounds,
        inference_mask=inference_mask,
    )

    monkeypatch.setattr(
        residual_maps,
        "load_image_preparation_products",
        lambda config: image_products,
    )

    monkeypatch.setattr(
        residual_maps,
        "load_empirical_psf",
        lambda config: None,
    )

    output_path = (
        tmp_path
        / "residual.pdf"
    )

    monkeypatch.setattr(
        residual_maps,
        "pipeline_paths",
        lambda config: {
            "fits": {
                "single_sersic": {
                    "residual": output_path,
                },
            },
        },
    )

    calls = []

    def fake_plot_pipeline_fit_maps(
        **kwargs,
    ):
        calls.append(
            kwargs
        )
        return (
            _Figure(),
            object(),
        )

    monkeypatch.setattr(
        residual_maps,
        "plot_pipeline_fit_maps",
        fake_plot_pipeline_fit_maps,
    )

    monkeypatch.setattr(
        residual_maps.plt,
        "close",
        lambda figure: None,
    )

    residual_maps._run_single_observation_residual_maps(
        {
            "fit": {
                "models": [
                    "single_sersic",
                ],
            },
        },
        fit,
    )

    assert len(
        calls
    ) == 1

    call = calls[0]

    np.testing.assert_array_equal(
        call[
            "original_cutout"
        ],
        image_products.cutout[
            1:4,
            2:5,
        ],
    )

    np.testing.assert_array_equal(
        call[
            "prepared_cutout"
        ],
        image_products.prepared_data[
            1:4,
            2:5,
        ],
    )

    np.testing.assert_array_equal(
        call[
            "segmentation"
        ],
        image_products.segmentation[
            1:4,
            2:5,
        ],
    )

    np.testing.assert_array_equal(
        call[
            "contaminant_mask"
        ],
        inference_mask,
    )

    np.testing.assert_array_equal(
        call[
            "noise"
        ],
        fit.noise,
    )

    assert call[
        "cutout_bounds"
    ] == {
        "xmin": 102,
        "xmax": 105,
        "ymin": 201,
        "ymax": 204,
    }

    assert call[
        "original_image"
    ] is image_products.original_image
