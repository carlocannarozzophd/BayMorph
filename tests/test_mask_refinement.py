import numpy as np

from baymorph import refine_mask


def test_refine_mask_source_action():

    yy, xx = np.indices((101, 101))

    image = np.exp(
        -(
            (xx - 50) ** 2 +
            (yy - 50) ** 2
        ) / (2 * 5**2)
    )

    mask = np.zeros_like(
        image,
        dtype=bool,
    )

    refined = refine_mask(
        image=image,
        mask=mask,
        actions=[
            {
                "type": "source",
                "x": 50,
                "y": 50,
            },
        ],
    )

    assert refined.sum() > 0

    assert refined[
        50,
        50,
    ]


def test_refine_mask_star_action():

    yy, xx = np.indices((101, 101))

    image = np.exp(
        -(
            (xx - 50) ** 2 +
            (yy - 50) ** 2
        ) / (2 * 5**2)
    )

    mask = np.zeros_like(
        image,
        dtype=bool,
    )

    refined = refine_mask(
        image=image,
        mask=mask,
        actions=[
            {
                "type": "star",
                "x": 50,
                "y": 50,
                "radius": 20,
            },
        ],
    )

    assert refined.sum() > 0

    assert refined[
        50,
        50,
    ]


def test_refine_mask_multiple_actions():

    yy, xx = np.indices((201, 201))

    image = (
        np.exp(
            -(
                (xx - 50) ** 2 +
                (yy - 50) ** 2
            ) / (2 * 5**2)
        )
        +
        np.exp(
            -(
                (xx - 150) ** 2 +
                (yy - 150) ** 2
            ) / (2 * 5**2)
        )
    )

    mask = np.zeros_like(
        image,
        dtype=bool,
    )

    refined = refine_mask(
        image=image,
        mask=mask,
        actions=[
            {
                "type": "source",
                "x": 50,
                "y": 50,
            },
            {
                "type": "star",
                "x": 150,
                "y": 150,
                "radius": 20,
            },
        ],
    )

    assert refined[
        50,
        50,
    ]

    assert refined[
        150,
        150,
    ]


def test_refine_mask_unknown_action():

    image = np.zeros(
        (100, 100),
    )

    mask = np.zeros_like(
        image,
        dtype=bool,
    )

    import pytest

    with pytest.raises(
        ValueError,
    ):
        refine_mask(
            image=image,
            mask=mask,
            actions=[
                {
                    "type": "banana",
                    "x": 50,
                    "y": 50,
                },
            ],
        )