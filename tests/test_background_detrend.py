"""Tests for the gradient-insensitive background scale and its diagnostic."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.core.background import estimate_background_constant

TRUE_SIGMA = 2.0
TRUE_LEVEL = 20.0
SIDE = 108


def _field(
    seed: int,
    *,
    gradient: float = 0.0,
) -> np.ndarray:
    """Return a noisy sky field with an optional linear gradient in x."""
    rng = np.random.default_rng(
        seed
    )

    image = rng.normal(
        TRUE_LEVEL,
        TRUE_SIGMA,
        (
            SIDE,
            SIDE,
        ),
    )

    if gradient:
        ramp = np.linspace(
            -1.0,
            1.0,
            SIDE,
        ) * gradient
        image = image + ramp[None, :]

    return image


def _estimate(
    image: np.ndarray,
    *,
    detrend: bool,
):
    """Return the production-style estimate of one field."""
    return estimate_background_constant(
        image,
        noise_method="corrected_std",
        detrend=detrend,
    )


def test_detrending_removes_the_gradient_from_the_scale() -> None:
    """A gradient must not be counted as noise once it is detrended.

    The non-detrended behaviour is asserted too, so the test measures the fix
    rather than assuming it.
    """
    image = _field(
        11,
        gradient=3.0,
    )

    inflated = _estimate(
        image,
        detrend=False,
    ).sigma
    corrected = _estimate(
        image,
        detrend=True,
    ).sigma

    assert inflated / TRUE_SIGMA > 1.20
    assert abs(
        corrected / TRUE_SIGMA - 1.0
    ) < 0.02


def test_detrending_leaves_a_constant_sky_unchanged() -> None:
    """Fitting a plane to a flat sky must not change the estimate."""
    image = _field(
        12
    )

    plain = _estimate(
        image,
        detrend=False,
    ).sigma
    detrended = _estimate(
        image,
        detrend=True,
    ).sigma

    assert detrended == pytest.approx(
        plain,
        rel=0.01,
    )


def test_background_level_is_unaffected_by_detrending() -> None:
    """Detrending changes how the scale is measured, not what is subtracted."""
    image = _field(
        13
    )

    plain = _estimate(
        image,
        detrend=False,
    ).background_level
    detrended = _estimate(
        image,
        detrend=True,
    ).background_level

    assert detrended == pytest.approx(
        plain,
        abs=0.05,
    )
    assert detrended == pytest.approx(
        TRUE_LEVEL,
        abs=0.05,
    )


def test_gradient_diagnostic_recovers_an_injected_gradient() -> None:
    """The reported gradient must match the injected one."""
    gradient = 1.0
    image = _field(
        14,
        gradient=gradient,
    )

    parameters = _estimate(
        image,
        detrend=True,
    ).parameters

    expected = 2.0 * gradient / (SIDE - 1)

    assert parameters["gradient_x"] == pytest.approx(
        expected,
        rel=0.10,
    )
    assert abs(
        parameters["gradient_y"]
    ) < 3.0 * parameters["gradient_y_error"]


def test_gradient_diagnostic_discriminates() -> None:
    """The diagnostic must separate a flat sky from a gradient-affected one.

    A number that merely exists is useless; what matters is that its
    significance is small when there is no gradient and large when there is.
    """
    flat = _estimate(
        _field(
            15
        ),
        detrend=True,
    ).parameters
    sloped = _estimate(
        _field(
            15,
            gradient=1.0,
        ),
        detrend=True,
    ).parameters

    assert flat["gradient_x_significance"] < 3.0
    assert sloped["gradient_x_significance"] > 10.0


def test_gradient_diagnostic_is_absent_without_detrending() -> None:
    """The plain estimator reports no background-model parameters."""
    assert _estimate(
        _field(
            16
        ),
        detrend=False,
    ).parameters is None


def test_detrending_rejects_one_dimensional_input() -> None:
    """Fitting a plane needs pixel coordinates, so 1D input must fail."""
    rng = np.random.default_rng(
        17
    )

    with pytest.raises(
        ValueError,
        match="two-dimensional",
    ):
        estimate_background_constant(
            rng.normal(
                TRUE_LEVEL,
                TRUE_SIGMA,
                1000,
            ),
            detrend=True,
        )


def test_production_preparation_detrends_and_reports_the_gradient(
    tmp_path,
) -> None:
    """The canonical preparation must be wired to the detrended scale.

    Testing the estimator alone would not catch the production call losing its
    ``detrend`` argument, so this exercises the real preparation end to end on
    a synthetic galaxy with an injected sky gradient.
    """
    from copy import deepcopy

    from astropy.io import fits

    from baymorph.models.sersic import SersicModel
    from baymorph.pipeline import run_image_preparation
    from baymorph.pipeline.config import DEFAULT_CONFIG

    side = 121
    gradient = 2.0
    rng = np.random.default_rng(
        20
    )

    galaxy = SersicModel(
        flux=20000.0,
        re=6.0,
        n=2.0,
        q=0.75,
        theta=0.4363,
        x0=60.0,
        y0=60.0,
    )
    image = (
        galaxy.render(
            (
                side,
                side,
            )
        )
        + TRUE_LEVEL
        + rng.normal(
            0.0,
            TRUE_SIGMA,
            (
                side,
                side,
            ),
        )
        + np.linspace(
            -gradient,
            gradient,
            side,
        )[None, :]
    )

    image_path = tmp_path / "image.fits"
    fits.writeto(
        image_path,
        image.astype(float),
    )

    config = deepcopy(
        DEFAULT_CONFIG
    )
    config["object"] = {
        "name": "gradient_case",
    }
    config["data"]["image_path"] = str(
        image_path
    )
    config["data"]["target_position"] = {
        "x": 60,
        "y": 60,
    }
    config["data"]["cutout"] = {
        "mode": "fixed",
        "size": side,
    }
    config["outputs"] = {
        "root": str(
            tmp_path / "runs"
        ),
        "overwrite": True,
    }
    config["psf"] = {
        "type": "gaussian",
        "gaussian": {
            "fwhm": 2.0,
        },
    }

    products = run_image_preparation(
        config
    )

    # The gradient must not have been counted as noise.
    assert abs(
        float(
            products.noise[0, 0]
        )
        / TRUE_SIGMA
        - 1.0
    ) < 0.03

    assert products.sky_gradient is not None
    assert products.sky_gradient[
        "gradient_x"
    ] == pytest.approx(
        2.0 * gradient / (side - 1),
        rel=0.05,
    )
    assert products.sky_gradient[
        "gradient_x_significance"
    ] > 10.0
    assert products.sky_gradient[
        "gradient_y_significance"
    ] < 5.0


def test_detrending_keeps_robustness_to_bright_pixels() -> None:
    """Clipping must still reject unmasked sources when detrending."""
    rng = np.random.default_rng(
        18
    )

    image = _field(
        19
    )
    contaminated = image.copy()
    rows = rng.integers(
        0,
        SIDE,
        400,
    )
    columns = rng.integers(
        0,
        SIDE,
        400,
    )
    contaminated[rows, columns] += rng.uniform(
        5.0,
        200.0,
        400,
    )

    estimate = _estimate(
        contaminated,
        detrend=True,
    ).sigma

    assert abs(
        estimate / TRUE_SIGMA - 1.0
    ) < 0.03
