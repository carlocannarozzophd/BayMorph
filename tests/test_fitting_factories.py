"""Tests for BayMorph model factories."""

import pickle

import numpy as np
import pytest

from baymorph.fitting import (
    ObservedModelFactory,
    make_double_sersic_psf_model,
    make_single_sersic_psf_model,
    resolve_model_factory,
)
from baymorph.models import registered_models
from baymorph.observation.background import BackgroundAppliedModel


SINGLE_SERSIC = {
    "flux": 100.0,
    "re": 3.0,
    "n": 2.0,
    "q": 0.8,
    "theta": 0.0,
    "x0": 15.0,
    "y0": 15.0,
}


class DeltaPSF:
    """Identity PSF kernel used in model-factory tests."""

    includes_pixel_response = False

    def kernel(self):
        """Return a normalized delta-function kernel."""
        kernel = np.zeros(
            (9, 9),
            dtype=float,
        )
        kernel[4, 4] = 1.0

        return kernel


def test_make_single_sersic_psf_model_renders_expected_shape():
    """The single-Sérsic factory should render the requested shape."""
    model = make_single_sersic_psf_model(
        **SINGLE_SERSIC,
        psf=DeltaPSF(),
    )

    image = model.render(
        shape=(30, 30)
    )

    assert image.shape == (30, 30)
    assert np.all(np.isfinite(image))


def test_make_double_sersic_psf_model_renders_expected_shape():
    """The double-Sérsic factory should render the requested shape."""
    model = make_double_sersic_psf_model(
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
        psf=DeltaPSF(),
    )

    image = model.render(
        shape=(30, 30)
    )

    assert image.shape == (30, 30)
    assert np.all(np.isfinite(image))


def test_make_sersic_point_source_psf_model_renders_components():
    """The factory must render extended and unresolved contributions."""
    from baymorph.fitting import make_sersic_point_source_psf_model

    model = make_sersic_point_source_psf_model(
        flux_sersic=100.0,
        re=3.0,
        n=2.0,
        q=0.8,
        theta=0.0,
        flux_point_source=25.0,
        x0=15.0,
        y0=15.0,
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
        + components["point_source"],
        rtol=0.0,
        atol=1.0e-13,
    )


def test_make_bulge_disk_psf_model_renders_expected_components():
    """The bulge-disc factory must render bulge and exponential disc."""
    from baymorph.fitting import make_bulge_disk_psf_model

    model = make_bulge_disk_psf_model(
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
        + components["disk"],
        rtol=0.0,
        atol=1.0e-13,
    )


@pytest.mark.parametrize(
    "model_name",
    registered_models(),
)
def test_every_registered_model_resolves_a_factory(model_name):
    """Each registered morphology must map to one production factory."""
    assert callable(
        resolve_model_factory(
            model_name
        )
    )


def test_observed_model_factory_applies_constant_background():
    """The production factory adds the residual background as an effect."""
    factory = ObservedModelFactory(
        "single_sersic",
        background_model="constant",
        psf=DeltaPSF(),
    )

    assert factory.parameter_names == (
        *SINGLE_SERSIC,
        "background_level",
    )

    observed = factory(
        **SINGLE_SERSIC,
        background_level=2.0,
    )

    assert isinstance(
        observed,
        BackgroundAppliedModel,
    )

    source = make_single_sersic_psf_model(
        **SINGLE_SERSIC,
        psf=DeltaPSF(),
    ).render(
        shape=(30, 30)
    )

    np.testing.assert_allclose(
        observed.render(
            shape=(30, 30)
        )
        - source,
        2.0,
    )


def test_observed_model_factory_applies_plane_background():
    """A plane background adds its gradients to the parameter contract."""
    factory = ObservedModelFactory(
        "single_sersic",
        background_model="plane",
        psf=DeltaPSF(),
    )

    assert factory.background_parameter_names == (
        "background_level",
        "background_gradient_x",
        "background_gradient_y",
    )

    observed = factory(
        **SINGLE_SERSIC,
        background_level=1.0,
        background_gradient_x=0.5,
        background_gradient_y=0.0,
    )

    background = observed.render_background(
        (5, 7)
    )

    assert background[2, 3] == pytest.approx(1.0)
    assert background[2, 4] == pytest.approx(1.5)


def test_observed_model_factory_takes_psf_at_call_time_for_joint_fits():
    """Joint inference injects each observation's own PSF at call time."""
    factory = ObservedModelFactory(
        "single_sersic",
        background_model="constant",
    )

    observed = factory(
        psf=DeltaPSF(),
        **SINGLE_SERSIC,
        background_level=0.0,
    )

    assert np.all(
        np.isfinite(
            observed.render(
                shape=(20, 20)
            )
        )
    )

    with pytest.raises(
        ValueError,
        match="requires a PSF",
    ):
        factory(
            **SINGLE_SERSIC,
            background_level=0.0,
        )


def test_observed_model_factory_rejects_psf_override_and_contract_breaks():
    """The factory fails closed on ambiguous PSFs and wrong parameters."""
    factory = ObservedModelFactory(
        "single_sersic",
        psf=DeltaPSF(),
    )

    with pytest.raises(
        ValueError,
        match="bound at construction",
    ):
        factory(
            psf=DeltaPSF(),
            **SINGLE_SERSIC,
            background_level=0.0,
        )

    with pytest.raises(
        ValueError,
        match=r"Missing: \['background_level'\]. Extra: \['sky'\]",
    ):
        factory(
            **SINGLE_SERSIC,
            sky=0.0,
        )

    with pytest.raises(
        ValueError,
        match="Unsupported background model",
    ):
        ObservedModelFactory(
            "single_sersic",
            background_model="polynomial",
        )


def test_observed_model_factory_is_picklable():
    """Worker processes receive the factory by pickling."""
    factory = ObservedModelFactory(
        "double_sersic",
        background_model="plane",
    )

    restored = pickle.loads(
        pickle.dumps(
            factory
        )
    )

    assert restored.model_name == "double_sersic"
    assert restored.background_model == "plane"
    assert restored.parameter_names == factory.parameter_names
