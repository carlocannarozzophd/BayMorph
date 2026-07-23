import numpy as np

from baymorph.fitting import (
    Parameter,
    ParameterSet,
)



def test_parameter_transform():
    parameter = Parameter(
        name="flux",
        lower=100.0,
        upper=200.0,
    )

    assert np.isclose(
        parameter.transform(0.0),
        100.0,
    )

    assert np.isclose(
        parameter.transform(1.0),
        200.0,
    )

    assert np.isclose(
        parameter.transform(0.5),
        150.0,
    )


def test_parameter_set():
    parameters = ParameterSet(
        [
            Parameter(
                "flux",
                100.0,
                200.0,
            ),
            Parameter(
                "re",
                1.0,
                10.0,
            ),
        ]
    )

    assert parameters.names == [
        "flux",
        "re",
    ]

    assert parameters.ndim == 2

    assert parameters.bounds == {
        "flux": (
            100.0,
            200.0,
        ),
        "re": (
            1.0,
            10.0,
        ),
    }

    values = parameters.transform(
        [
            0.5,
            0.0,
        ]
    )

    assert np.allclose(
        values,
        np.array(
            [
                150.0,
                1.0,
            ]
        ),
    )