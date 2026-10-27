"""The target position is the centre of the galaxy: the checks of the centre."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import baymorph.pipeline.real_fit as real_fit_module
from baymorph.pipeline.preparation_steps import TargetGeometry
from baymorph.pipeline.real_fit import position_on_segment, require_centre_within_priors

CENTRE_PRIORS = {
    "x0": {"type": "uniform", "lower": -8.0, "upper": 8.0, "relative_to": "target_x"},
    "y0": {"type": "uniform", "lower": -8.0, "upper": 8.0, "relative_to": "target_y"},
}
SEED = {"target_x": 50.0, "target_y": 40.0}


def test_a_centre_inside_the_window_passes() -> None:
    require_centre_within_priors(
        centre=(57.5, 33.0), priors=CENTRE_PRIORS, references=SEED
    )


def test_a_centre_outside_the_window_stops_with_the_numbers() -> None:
    with pytest.raises(ValueError) as error:
        require_centre_within_priors(
            centre=(62.0, 45.0),
            priors=CENTRE_PRIORS,
            references=SEED,
            origin=(100.0, 200.0),
            pixel_scale=0.05,
            model_name="single_sersic",
        )

    message = str(error.value)

    assert "lies outside the priors of the centre of single_sersic" in message
    # The positions are given in pixels of the whole image.
    assert "x = 162.0, y = 245.0" in message
    assert "x0 from 142.0 to 158.0 and y0 from 232.0 to 248.0" in message
    assert "about the target position x = 150.0, y = 240.0" in message
    assert "13.0 pixels (0.65 arcsec) from that centre" in message
    assert "catalogue position in RA and Dec" in message
    assert "declare wider priors for x0 and y0" in message
    assert not message.startswith("No source")


def test_a_position_on_no_source_is_named() -> None:
    with pytest.raises(ValueError, match="^No source was detected at the target"):
        require_centre_within_priors(
            centre=(80.0, 40.0),
            priors=CENTRE_PRIORS,
            references=SEED,
            seed_on_target=False,
        )


def test_absolute_and_bounded_priors_are_checked() -> None:
    priors = {
        "x0": {"type": "uniform", "lower": 45.0, "upper": 55.0},
        "y0": {"type": "beta", "alpha": 2.0, "beta": 2.0, "lower": 30.0, "upper": 50.0},
    }

    require_centre_within_priors(centre=(54.0, 31.0), priors=priors, references=SEED)

    with pytest.raises(ValueError, match="y0 from 30.0 to 50.0"):
        require_centre_within_priors(
            centre=(54.0, 51.0), priors=priors, references=SEED
        )


def test_unbounded_priors_and_models_without_a_centre_are_not_checked() -> None:
    normal = {
        "x0": {"type": "normal", "mean": 0.0, "sigma": 2.0},
        "y0": {"type": "normal", "mean": 0.0, "sigma": 2.0},
    }

    require_centre_within_priors(centre=(500.0, 500.0), priors=normal, references=SEED)
    require_centre_within_priors(
        centre=(500.0, 500.0),
        priors={"flux": {"type": "uniform", "lower": 1.0, "upper": 2.0}},
        references=SEED,
    )


def test_position_on_segment() -> None:
    segment = np.zeros((10, 20), dtype=bool)
    segment[4:6, 8:12] = True

    assert position_on_segment(segment, 9.4, 4.6)
    assert not position_on_segment(segment, 2.0, 4.0)
    assert not position_on_segment(segment, 25.0, 4.0)
    assert not position_on_segment(segment, 9.0, -1.0)


def _prepared(centre_x: float, centre_y: float, *, geometry: bool = True):
    """Prepared products of a cutout of 100 x 80 pixels starting at (300, 400)."""
    segment = np.zeros((80, 100), dtype=bool)
    segment[30:50, 40:60] = True

    return SimpleNamespace(
        geometry=(
            TargetGeometry(x0=centre_x, y0=centre_y, q=1.0, theta=0.0)
            if geometry
            else None
        ),
        target_x=int(round(centre_x)),
        target_y=int(round(centre_y)),
        target_segment=segment,
        cutout_bounds={"xmin": 300, "xmax": 400, "ymin": 400, "ymax": 480},
    )


def test_the_fit_checks_the_priors_of_every_model() -> None:
    resolved = {
        "fit": {
            "models": ["single_sersic", "double_sersic"],
            "priors": {
                "single_sersic": dict(CENTRE_PRIORS),
                "double_sersic": {
                    "x0": {
                        "type": "uniform",
                        "lower": -2.0,
                        "upper": 2.0,
                        "relative_to": "target_x",
                    },
                    "y0": dict(CENTRE_PRIORS["y0"]),
                },
            },
        }
    }
    references = {"target_x": 50.0, "target_y": 40.0}

    with pytest.raises(ValueError, match="the centre of double_sersic"):
        real_fit_module._require_prepared_centre_within_priors(
            resolved,
            image_products=_prepared(55.0, 40.0),
            references=references,
            observation=None,
        )

    real_fit_module._require_prepared_centre_within_priors(
        resolved,
        image_products=_prepared(51.0, 40.0),
        references=references,
        observation=None,
    )


def test_products_without_a_measured_geometry_use_the_target_pixel() -> None:
    resolved = {
        "fit": {"models": ["single_sersic"], "priors": {"single_sersic": CENTRE_PRIORS}}
    }

    with pytest.raises(ValueError, match="x = 361.0, y = 440.0"):
        real_fit_module._require_prepared_centre_within_priors(
            resolved,
            image_products=_prepared(61.0, 40.0, geometry=False),
            references={"target_x": 50.0, "target_y": 40.0},
            observation=None,
        )


def test_the_fit_stops_before_it_writes_anything(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    written: list = []
    config = {
        "object": {"name": "galaxy"},
        "data": {
            "image_path": str(tmp_path / "missing.fits"),
            "target_position": {"x": 350.0, "y": 440.0},
        },
        "outputs": {"root": str(tmp_path / "runs")},
        "fit": {
            "models": ["single_sersic"],
            "priors": {
                "single_sersic": {
                    "flux": {"type": "uniform", "lower": 1.0, "upper": 2.0}
                }
            },
            "likelihood": {
                "type": "correlated_gaussian",
                "effective_gain": {"source": "absent"},
            },
        },
    }

    monkeypatch.setattr(
        real_fit_module,
        "load_image_preparation_products",
        lambda resolved: _prepared(70.0, 40.0),
    )
    monkeypatch.setattr(real_fit_module, "_load_psf_from_config", lambda resolved: None)
    monkeypatch.setattr(
        real_fit_module,
        "prepare_real_fit_likelihood_state",
        lambda resolved, products: None,
    )
    monkeypatch.setattr(
        real_fit_module, "require_complete_likelihood_normalization", lambda state: None
    )
    monkeypatch.setattr(
        real_fit_module, "make_output_tree", lambda resolved: written.append(resolved)
    )

    with pytest.raises(ValueError, match="lies outside the priors of the centre"):
        real_fit_module.run_real_fit_from_config(config)

    assert written == []
    assert not (tmp_path / "runs").exists()
