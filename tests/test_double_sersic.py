import numpy as np

from baymorph import (
    CompositeModel,
    DoubleSersicModel,
    SersicModel,
)



def test_double_sersic_matches_composite():

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

    composite = CompositeModel(
        [
            bulge,
            disk,
        ]
    )

    double_sersic = DoubleSersicModel(
        bulge=bulge,
        disk=disk,
    )

    composite_image = composite.render(
        shape=(60, 60),
    )

    double_image = double_sersic.render(
        shape=(60, 60),
    )

    assert np.allclose(
        composite_image,
        double_image,
    )


def test_double_sersic_threads_renderer_kwarg_to_components():

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

    double_sersic = DoubleSersicModel(
        bulge=bulge,
        disk=disk,
    )

    point_sampled = double_sersic.render(
        shape=(60, 60),
        renderer="point",
    )

    reference = bulge.render(
        shape=(60, 60),
        renderer="point",
    ) + disk.render(
        shape=(60, 60),
        renderer="point",
    )

    assert np.allclose(
        point_sampled,
        reference,
    )

    assert not np.allclose(
        point_sampled,
        double_sersic.render(shape=(60, 60)),
    )