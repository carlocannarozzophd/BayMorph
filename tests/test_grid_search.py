"""Tests for grid-search fitting utilities."""

import numpy as np

from baymorph import (
    GaussianImageLikelihood,
    SersicModel,
    grid_search_1d,
    make_constant_noise_map,
)


def test_grid_search_recovers_flux():
    """The grid search should recover the true flux."""
    true_flux = 1000.0

    true_model = SersicModel(
        flux=true_flux,
        re=8.0,
        n=2.0,
        q=0.8,
        theta=0.0,
        x0=30.0,
        y0=30.0,
    )

    data = true_model.render(
        shape=(60, 60),
    )

    noise = make_constant_noise_map(
        shape=data.shape,
        sigma=1.0,
    )

    likelihood = GaussianImageLikelihood(
        data=data,
        noise=noise,
    )

    flux_grid = np.array(
        [
            500.0,
            750.0,
            1000.0,
            1250.0,
            1500.0,
        ]
    )

    def model_factory(
        flux,
    ):
        return SersicModel(
            flux=flux,
            re=8.0,
            n=2.0,
            q=0.8,
            theta=0.0,
            x0=30.0,
            y0=30.0,
        )

    result = grid_search_1d(
        values=flux_grid,
        model_factory=model_factory,
        likelihood=likelihood,
    )

    assert result.best_value == true_flux
    assert result.best_index == 2
    assert np.isfinite(
        result.best_loglike
    )