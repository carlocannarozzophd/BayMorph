import numpy as np
import pytest

from baymorph import SyntheticSersic


def test_synthetic_sersic_shape():
    model = SyntheticSersic(
        flux=1000.0,
        re=10.0,
        n=4.0,
        q=0.7,
        theta=0.5,
        x0=50.0,
        y0=50.0,
    )

    image = model.render(shape=(100, 100))

    assert image.shape == (100, 100)


def test_synthetic_sersic_uses_physical_flux_normalization():
    model = SyntheticSersic(
        flux=1000.0,
        re=5.0,
        n=1.0,
        q=0.8,
        theta=0.4,
        x0=100.0,
        y0=100.0,
    )

    image = model.render(
        shape=(201, 201),
        integration_order=7,
    )

    assert image.sum() == pytest.approx(
        model.flux,
        rel=5.0e-5,
    )


def test_synthetic_sersic_is_cutout_invariant():
    model = SyntheticSersic(
        flux=1000.0,
        re=8.0,
        n=2.0,
        q=0.65,
        theta=0.7,
        x0=30.4,
        y0=29.7,
    )

    smaller = model.render(shape=(61, 61))
    larger = model.render(shape=(121, 121))

    np.testing.assert_allclose(
        smaller,
        larger[:61, :61],
        rtol=0.0,
        atol=0.0,
    )
