import numpy as np

from baymorph import (
    CompositeModel,
    SersicModel,
)



def test_composite_model_sums_components():

    component_1 = SersicModel(
        flux=1000,
        re=8,
        n=2,
        q=0.8,
        theta=0.0,
        x0=30,
        y0=30,
    )

    component_2 = SersicModel(
        flux=500,
        re=4,
        n=1,
        q=0.9,
        theta=0.0,
        x0=30,
        y0=30,
    )

    composite = CompositeModel(
        [
            component_1,
            component_2,
        ]
    )

    image_1 = component_1.render(
        shape=(60, 60),
    )

    image_2 = component_2.render(
        shape=(60, 60),
    )

    composite_image = composite.render(
        shape=(60, 60),
    )

    assert np.allclose(
        composite_image,
        image_1 + image_2,
    )