"""Numerical closure tests for multi-observation forward rendering."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting.model_factories import (
    make_single_sersic_psf_sky_model,
)
from baymorph.models.geometry import (
    get_model_geometry,
)
from baymorph.observation.geometry import (
    slice_wcs_to_cutout,
)
from baymorph.observation.model_parameter_transforms import (
    SkyModelParameterTransform,
)
from baymorph.simulation.psf import GaussianPSF
from baymorph.validation.multi_observation_workspace import (
    MultiObservationValidationWorkspace,
)


TRUTH = {
    "flux": 1200.0,
    "re_arcsec": 1.2,
    "n": 2.0,
    "q": 0.65,
    "position_angle": 0.35,
    "ra": 150.0,
    "dec": 30.0,
    "sky": 0.0,
}


@pytest.mark.parametrize(
    (
        "shape",
        "pixel_scale",
        "rotation_deg",
        "crpix",
        "psf_fwhm",
    ),
    [
        (
            (61, 61),
            0.20,
            0.0,
            (31.0, 31.0),
            2.0,
        ),
        (
            (47, 73),
            0.35,
            32.0,
            (38.0, 24.0),
            2.8,
        ),
    ],
)
def test_full_to_cutout_forward_rendering_closes(
    shape,
    pixel_scale,
    rotation_deg,
    crpix,
    psf_fwhm,
):
    geometry = get_model_geometry(
        "single_sersic"
    )

    wcs = MultiObservationValidationWorkspace._make_wcs(
        ra=TRUTH["ra"],
        dec=TRUTH["dec"],
        pixel_scale_arcsec=pixel_scale,
        rotation_deg=rotation_deg,
        crpix=crpix,
    )

    full_transform = SkyModelParameterTransform(
        wcs=wcs,
        geometry=geometry,
    )

    full_parameters = full_transform(
        TRUTH
    )

    psf = GaussianPSF(
        fwhm=psf_fwhm
    )

    full_model = make_single_sersic_psf_sky_model(
        psf=psf,
        **full_parameters,
    )

    full_image = full_model.render(
        shape=shape
    )

    size = min(
        shape
    )

    x0, y0 = wcs.world_to_pixel_values(
        TRUTH["ra"],
        TRUTH["dec"],
    )

    xmin = int(
        np.floor(
            x0 - size / 2
        )
    )
    ymin = int(
        np.floor(
            y0 - size / 2
        )
    )

    xmin = max(
        0,
        min(
            xmin,
            shape[1] - size,
        ),
    )
    ymin = max(
        0,
        min(
            ymin,
            shape[0] - size,
        ),
    )

    xmax = xmin + size
    ymax = ymin + size

    direct_cutout = full_image[
        ymin:ymax,
        xmin:xmax,
    ]

    cutout_wcs = slice_wcs_to_cutout(
        wcs,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
    )

    cutout_transform = SkyModelParameterTransform(
        wcs=cutout_wcs,
        geometry=geometry,
    )

    cutout_parameters = cutout_transform(
        TRUTH
    )

    cutout_model = make_single_sersic_psf_sky_model(
        psf=psf,
        **cutout_parameters,
    )

    rerendered_cutout = cutout_model.render(
        shape=direct_cutout.shape
    )

    difference = (
        rerendered_cutout
        - direct_cutout
    )

    assert np.max(
        np.abs(difference)
    ) < 1.0e-8

    assert np.sum(
        rerendered_cutout
    ) == pytest.approx(
        np.sum(direct_cutout),
        rel=1.0e-10,
        abs=1.0e-10,
    )
