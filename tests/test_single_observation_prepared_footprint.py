"""Tests for the canonical single-observation prepared footprint."""

from types import SimpleNamespace

import numpy as np

import baymorph.pipeline.real_fit as real_fit
from baymorph.pipeline.config import resolve_config
from baymorph.pipeline.residual_maps import (
    _single_observation_plot_products,
)
from baymorph.results import RealFitProducts


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
            },
        }
    )


def test_real_fit_round_trip_preserves_prepared_noise(
    tmp_path,
    monkeypatch,
) -> None:
    """Persisted fits must retain noise on the complete prepared footprint."""
    config = _config(
        tmp_path
    )

    shape = (
        6,
        7,
    )

    noise = np.arange(
        np.prod(
            shape
        ),
        dtype=float,
    ).reshape(
        shape
    ) + 1.0

    monkeypatch.setattr(
        real_fit,
        "load_image_preparation_products",
        lambda config: SimpleNamespace(
            noise=noise,
        ),
    )

    products = RealFitProducts(
        posterior=None,
        best_parameters={
            "flux": 10.0,
        },
        best_model_image=np.zeros(
            shape,
            dtype=float,
        ),
        residual=np.zeros(
            shape,
            dtype=float,
        ),
        norm_residual=np.zeros(
            shape,
            dtype=float,
        ),
        chi2=1.0,
        ndof=10,
        reduced_chi2=0.1,
        model_name="single_sersic",
        noise=noise,
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

    np.testing.assert_array_equal(
        loaded.noise,
        noise,
    )

    assert (
        loaded.best_model_image.shape
        == shape
    )

    assert (
        "inference_bounds"
        not in loaded.summary()
    )

    assert not list(
        tmp_path.rglob(
            "inference_mask.npy"
        )
    )


def test_residual_products_use_complete_prepared_footprint() -> None:
    """Residual diagnostics must remain aligned with prepared products."""
    shape = (
        6,
        7,
    )

    fitting_mask = np.zeros(
        shape,
        dtype=bool,
    )
    fitting_mask[
        0,
        0,
    ] = True

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
        segmentation=np.zeros(
            shape,
            dtype=int,
        ),
        fitting_mask=fitting_mask,
        target_segment=np.ones(
            shape,
            dtype=bool,
        ),
        cutout_bounds={
            "xmin": 100,
            "xmax": 107,
            "ymin": 200,
            "ymax": 206,
        },
    )

    fit = RealFitProducts(
        posterior=None,
        best_parameters={},
        best_model_image=np.zeros(
            shape,
            dtype=float,
        ),
        residual=np.zeros(
            shape,
            dtype=float,
        ),
        norm_residual=np.zeros(
            shape,
            dtype=float,
        ),
        chi2=1.0,
        ndof=1,
        reduced_chi2=1.0,
        model_name="single_sersic",
        noise=np.full(
            shape,
            2.0,
            dtype=float,
        ),
    )

    plot_products = (
        _single_observation_plot_products(
            image_products,
            fit,
        )
    )

    np.testing.assert_array_equal(
        plot_products[
            "original_cutout"
        ],
        image_products.cutout,
    )

    np.testing.assert_array_equal(
        plot_products[
            "prepared_cutout"
        ],
        image_products.prepared_data,
    )

    np.testing.assert_array_equal(
        plot_products[
            "contaminant_mask"
        ],
        fitting_mask,
    )

    np.testing.assert_array_equal(
        plot_products[
            "noise"
        ],
        fit.noise,
    )

    assert (
        plot_products[
            "cutout_bounds"
        ]
        == image_products.cutout_bounds
    )
