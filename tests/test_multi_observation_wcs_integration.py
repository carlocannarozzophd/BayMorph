"""Scientific integration tests for heterogeneous WCS-aware inference."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.wcs import WCS

from baymorph.fitting.model_factories import ObservedModelFactory
from baymorph.fitting.multi_observation import (
    MultiObservationParameterLayout,
    MultiObservationParameterSpec,
)
from baymorph.fitting.multi_observation_problem import (
    MultiObservationInferenceProblem,
)
from baymorph.fitting.parameters import Parameter
from baymorph.fitting.representations import (
    RepresentationFactory,
)
from baymorph.likelihood import GaussianImageLikelihood
from baymorph.likelihood.multi_observation import MultiObservationLikelihood
from baymorph.models.geometry import get_model_geometry
from baymorph.observation.model_parameter_transforms import (
    SkyModelParameterTransform,
)
from baymorph.simulation.psf import GaussianPSF


def _make_wcs(
    *,
    pixel_scale_arcsec: float,
    rotation_deg: float,
    crpix: tuple[float, float],
) -> WCS:
    angle = np.deg2rad(
        rotation_deg
    )
    scale = (
        pixel_scale_arcsec
        / 3600.0
    )

    rotation = np.array(
        [
            [
                np.cos(angle),
                -np.sin(angle),
            ],
            [
                np.sin(angle),
                np.cos(angle),
            ],
        ],
        dtype=float,
    )

    reflection = np.array(
        [
            [-1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = list(
        crpix
    )
    wcs.wcs.crval = [
        150.0,
        30.0,
    ]
    wcs.wcs.ctype = [
        "RA---TAN",
        "DEC--TAN",
    ]
    wcs.wcs.cd = (
        scale
        * rotation
        @ reflection
    )

    return wcs


def _layout() -> MultiObservationParameterLayout:
    return MultiObservationParameterLayout(
        observations=[
            "g",
            "r",
        ],
        specifications=[
            MultiObservationParameterSpec.independent(
                "flux",
                {
                    "g": Parameter(
                        "flux",
                        lower=500.0,
                        upper=1500.0,
                    ),
                    "r": Parameter(
                        "flux",
                        lower=500.0,
                        upper=1500.0,
                    ),
                },
            ),
            MultiObservationParameterSpec.shared_parameter(
                Parameter(
                    "re_arcsec",
                    lower=0.2,
                    upper=3.0,
                )
            ),
            MultiObservationParameterSpec.shared_parameter(
                Parameter(
                    "n",
                    lower=0.5,
                    upper=5.0,
                )
            ),
            MultiObservationParameterSpec.shared_parameter(
                Parameter(
                    "q",
                    lower=0.3,
                    upper=1.0,
                )
            ),
            MultiObservationParameterSpec.shared_parameter(
                Parameter(
                    "position_angle",
                    lower=-1.5,
                    upper=1.5,
                )
            ),
            MultiObservationParameterSpec.shared_parameter(
                Parameter(
                    "ra",
                    lower=149.999,
                    upper=150.001,
                )
            ),
            MultiObservationParameterSpec.shared_parameter(
                Parameter(
                    "dec",
                    lower=29.999,
                    upper=30.001,
                )
            ),
            MultiObservationParameterSpec.independent(
                "background_level",
                {
                    "g": Parameter(
                        "background_level",
                        lower=-5.0,
                        upper=5.0,
                    ),
                    "r": Parameter(
                        "background_level",
                        lower=-5.0,
                        upper=5.0,
                    ),
                },
            ),
        ],
    )


def _factory(
    *,
    g_wcs: WCS,
    r_wcs: WCS,
) -> RepresentationFactory:
    geometry = get_model_geometry(
        "single_sersic"
    )

    psf = GaussianPSF(
        fwhm=2.355,
    )

    return RepresentationFactory(
        factory=ObservedModelFactory(
            "single_sersic",
            background_model="constant",
        ),
        fixed_kwargs={
            "g": {
                "psf": psf,
            },
            "r": {
                "psf": psf,
            },
        },
        parameter_transforms={
            "g": SkyModelParameterTransform(
                wcs=g_wcs,
                geometry=geometry,
            ),
            "r": SkyModelParameterTransform(
                wcs=r_wcs,
                geometry=geometry,
            ),
        },
    )


def _truth_vector(
    layout: MultiObservationParameterLayout,
) -> np.ndarray:
    values = {
        "flux__g": 900.0,
        "flux__r": 1200.0,
        "re_arcsec": 1.2,
        "n": 2.0,
        "q": 0.65,
        "position_angle": 0.35,
        "ra": 150.0,
        "dec": 30.0,
        "background_level__g": 0.0,
        "background_level__r": 0.0,
    }

    return np.asarray(
        [
            values[name]
            for name in layout.parameter_set.names
        ],
        dtype=float,
    )


def test_heterogeneous_wcs_joint_problem_preserves_one_sky_geometry():
    g_wcs = _make_wcs(
        pixel_scale_arcsec=0.2,
        rotation_deg=0.0,
        crpix=(31.0, 31.0),
    )
    r_wcs = _make_wcs(
        pixel_scale_arcsec=0.35,
        rotation_deg=32.0,
        crpix=(38.0, 24.0),
    )

    layout = _layout()
    factory = _factory(
        g_wcs=g_wcs,
        r_wcs=r_wcs,
    )

    routed = layout.route(
        {
            name: value
            for name, value in zip(
                layout.parameter_set.names,
                _truth_vector(layout),
                strict=True,
            )
        }
    )

    g_parameters = (
        factory.parameter_transforms["g"](
            routed["g"]
        )
    )
    r_parameters = (
        factory.parameter_transforms["r"](
            routed["r"]
        )
    )

    assert g_parameters["x0"] != pytest.approx(
        r_parameters["x0"]
    )
    assert g_parameters["y0"] != pytest.approx(
        r_parameters["y0"]
    )

    assert g_parameters["re"] == pytest.approx(
        1.2 / 0.2,
        rel=1.0e-6,
    )
    assert r_parameters["re"] == pytest.approx(
        1.2 / 0.35,
        rel=1.0e-6,
    )

    assert g_parameters["theta"] != pytest.approx(
        r_parameters["theta"]
    )


def test_heterogeneous_wcs_joint_likelihood_prefers_truth():
    g_wcs = _make_wcs(
        pixel_scale_arcsec=0.2,
        rotation_deg=0.0,
        crpix=(31.0, 31.0),
    )
    r_wcs = _make_wcs(
        pixel_scale_arcsec=0.35,
        rotation_deg=32.0,
        crpix=(38.0, 24.0),
    )

    layout = _layout()
    factory = _factory(
        g_wcs=g_wcs,
        r_wcs=r_wcs,
    )
    truth = _truth_vector(
        layout
    )

    temporary_likelihood = MultiObservationLikelihood(
        {
            "g": GaussianImageLikelihood(
                data=np.zeros(
                    (61, 61)
                ),
                noise=np.ones(
                    (61, 61)
                ),
            ),
            "r": GaussianImageLikelihood(
                data=np.zeros(
                    (47, 73)
                ),
                noise=np.ones(
                    (47, 73)
                ),
            ),
        }
    )

    generator = MultiObservationInferenceProblem(
        representation_factory=factory,
        parameter_layout=layout,
        likelihood=temporary_likelihood,
    )

    truth_images = generator.render_models(
        truth
    )

    assert truth_images["g"].shape == (
        61,
        61,
    )
    assert truth_images["r"].shape == (
        47,
        73,
    )

    likelihood = MultiObservationLikelihood(
        {
            "g": GaussianImageLikelihood(
                data=truth_images["g"],
                noise=np.ones(
                    (61, 61)
                ),
            ),
            "r": GaussianImageLikelihood(
                data=truth_images["r"],
                noise=np.ones(
                    (47, 73)
                ),
            ),
        }
    )

    problem = MultiObservationInferenceProblem(
        representation_factory=factory,
        parameter_layout=layout,
        likelihood=likelihood,
    )

    truth_logl = problem.log_likelihood(
        truth
    )

    perturbed = truth.copy()

    ra_index = problem.parameter_names.index(
        "ra"
    )
    re_index = problem.parameter_names.index(
        "re_arcsec"
    )

    perturbed[ra_index] += (
        0.3
        / 3600.0
    )
    perturbed[re_index] = 1.6

    perturbed_logl = problem.log_likelihood(
        perturbed
    )

    assert np.isfinite(
        truth_logl
    )
    assert np.isfinite(
        perturbed_logl
    )
    assert truth_logl > perturbed_logl
