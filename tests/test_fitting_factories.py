"""Tests for BayMorph model factories."""

import numpy as np

from baymorph.fitting import (
    make_double_sersic_psf_sky_model,
    make_single_sersic_psf_sky_model,
)



class DeltaPSF:
    """Identity PSF kernel used in model-factory tests."""

    def kernel(self):
        """Return a normalized delta-function kernel."""
        kernel = np.zeros(
            (9, 9),
            dtype=float,
        )
        kernel[4, 4] = 1.0

        return kernel


def test_make_single_sersic_psf_sky_model_renders_expected_shape():
    """The single-Sérsic factory should render the requested shape."""
    psf = DeltaPSF()

    model = make_single_sersic_psf_sky_model(
        flux=100.0,
        re=3.0,
        n=2.0,
        q=0.8,
        theta=0.0,
        x0=15.0,
        y0=15.0,
        sky=2.0,
        psf=psf,
    )

    image = model.render(
        shape=(30, 30)
    )

    assert image.shape == (30, 30)
    assert np.all(np.isfinite(image))


def test_make_single_sersic_psf_sky_model_adds_sky_level():
    """The single-Sérsic factory should add a constant sky level."""
    psf = DeltaPSF()

    model_no_sky = make_single_sersic_psf_sky_model(
        flux=100.0,
        re=3.0,
        n=2.0,
        q=0.8,
        theta=0.0,
        x0=15.0,
        y0=15.0,
        sky=0.0,
        psf=psf,
    )

    model_with_sky = make_single_sersic_psf_sky_model(
        flux=100.0,
        re=3.0,
        n=2.0,
        q=0.8,
        theta=0.0,
        x0=15.0,
        y0=15.0,
        sky=2.0,
        psf=psf,
    )

    image_no_sky = model_no_sky.render(
        shape=(30, 30)
    )
    image_with_sky = model_with_sky.render(
        shape=(30, 30)
    )

    assert np.allclose(
        image_with_sky - image_no_sky,
        2.0,
    )


def test_make_double_sersic_psf_sky_model_renders_expected_shape():
    """The double-Sérsic factory should render the requested shape."""
    psf = DeltaPSF()

    model = make_double_sersic_psf_sky_model(
        flux_bulge=100.0,
        re_bulge=3.0,
        n_bulge=2.0,
        q_bulge=0.8,
        theta_bulge=0.0,
        flux_disk=50.0,
        re_disk=8.0,
        n_disk=1.0,
        q_disk=0.8,
        theta_disk=0.0,
        x0=15.0,
        y0=15.0,
        sky=2.0,
        psf=psf,
    )

    image = model.render(
        shape=(30, 30)
    )

    assert image.shape == (30, 30)
    assert np.all(np.isfinite(image))


def test_make_double_sersic_psf_sky_model_adds_sky_level():
    """The double-Sérsic factory should add a constant sky level."""
    psf = DeltaPSF()

    model_no_sky = make_double_sersic_psf_sky_model(
        flux_bulge=100.0,
        re_bulge=3.0,
        n_bulge=2.0,
        q_bulge=0.8,
        theta_bulge=0.0,
        flux_disk=50.0,
        re_disk=8.0,
        n_disk=1.0,
        q_disk=0.8,
        theta_disk=0.0,
        x0=15.0,
        y0=15.0,
        sky=0.0,
        psf=psf,
    )

    model_with_sky = make_double_sersic_psf_sky_model(
        flux_bulge=100.0,
        re_bulge=3.0,
        n_bulge=2.0,
        q_bulge=0.8,
        theta_bulge=0.0,
        flux_disk=50.0,
        re_disk=8.0,
        n_disk=1.0,
        q_disk=0.8,
        theta_disk=0.0,
        x0=15.0,
        y0=15.0,
        sky=2.0,
        psf=psf,
    )

    image_no_sky = model_no_sky.render(
        shape=(30, 30)
    )
    image_with_sky = model_with_sky.render(
        shape=(30, 30)
    )

    assert np.allclose(
        image_with_sky - image_no_sky,
        2.0,
    )

def test_make_sersic_point_source_psf_sky_model_renders_components():
    """The factory must render extended and unresolved contributions."""
    from baymorph.fitting import make_sersic_point_source_psf_sky_model

    model = make_sersic_point_source_psf_sky_model(
        flux_sersic=100.0,
        re=3.0,
        n=2.0,
        q=0.8,
        theta=0.0,
        flux_point_source=25.0,
        x0=15.0,
        y0=15.0,
        sky=2.0,
        psf=DeltaPSF(),
    )

    image = model.render((31, 31))
    components = model.render_components((31, 31))

    assert image.shape == (31, 31)
    assert set(components) == {
        "sersic",
        "point_source",
    }

    np.testing.assert_allclose(
        image,
        components["sersic"]
        + components["point_source"]
        + 2.0,
        rtol=0.0,
        atol=1.0e-13,
    )


def test_make_bulge_disk_psf_sky_model_renders_expected_components():
    """The bulge-disc factory must render bulge and exponential disc."""
    from baymorph.fitting import make_bulge_disk_psf_sky_model

    model = make_bulge_disk_psf_sky_model(
        flux_bulge=100.0,
        re_bulge=3.0,
        n_bulge=3.0,
        q_bulge=0.9,
        theta_bulge=0.1,
        flux_disk=200.0,
        re_disk=10.0,
        q_disk=0.7,
        theta_disk=0.2,
        x0=15.0,
        y0=15.0,
        sky=2.0,
        psf=DeltaPSF(),
    )

    image = model.render((31, 31))
    components = model.render_components((31, 31))

    assert image.shape == (31, 31)
    assert set(components) == {
        "bulge",
        "disk",
    }

    np.testing.assert_allclose(
        image,
        components["bulge"]
        + components["disk"]
        + 2.0,
        rtol=0.0,
        atol=1.0e-13,
    )
