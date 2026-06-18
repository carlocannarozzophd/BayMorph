import numpy as np

from baymorph import (
    render_components_from_values,
)

from baymorph import (
    DoubleSersicModel,
    SersicModel,
)


def test_render_components_returns_three_images():
    bulge = SersicModel(
        flux=3000,
        re=3,
        n=4,
        q=0.8,
        theta=0.0,
        x0=30,
        y0=30,
    )

    disk = SersicModel(
        flux=7000,
        re=12,
        n=1,
        q=0.8,
        theta=0.0,
        x0=30,
        y0=30,
    )

    model = DoubleSersicModel(
        bulge=bulge,
        disk=disk,
    )

    values = [
        1000.0,
        5.0,
        2.0,
        0.8,
        0.0,
        500.0,
        15.0,
        1.0,
        0.8,
        0.0,
    ]

    inner, outer, total = render_components_from_values(
        model=model,
        values=values,
        shape=(51, 51),
    )

    assert inner.shape == (51, 51)
    assert outer.shape == (51, 51)
    assert total.shape == (51, 51)

    np.testing.assert_allclose(
        inner + outer,
        total,
        rtol=1e-8,
    )