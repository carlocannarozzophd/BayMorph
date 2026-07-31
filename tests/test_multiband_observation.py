import numpy as np

import pytest

from astropy.io import fits

from baymorph.core import (
    ImageData,
    MultiBandObservation,
    Observation,
)


def make_observation():
    image = ImageData(
        data=np.zeros((10, 10)),
        header=fits.Header(),
    )

    return Observation.from_image(image)


def test_multiband_requires_at_least_one_band():
    with pytest.raises(
        ValueError,
    ):
        MultiBandObservation({})


def test_multiband_rejects_non_observation():
    with pytest.raises(
        TypeError,
    ):
        MultiBandObservation(
            {
                "g": object(),
            }
        )


def test_multiband_lookup():
    g = make_observation()
    r = make_observation()

    obs = MultiBandObservation(
        {
            "g": g,
            "r": r,
        }
    )

    assert obs["g"] is g
    assert obs["r"] is r


def test_multiband_filters():
    obs = MultiBandObservation(
        {
            "F814W": make_observation(),
            "F160W": make_observation(),
        }
    )

    assert obs.filters == (
        "F814W",
        "F160W",
    )


def test_multiband_iteration():
    obs = MultiBandObservation(
        {
            "g": make_observation(),
            "r": make_observation(),
        }
    )

    assert list(obs) == list(obs.bands.items())


def test_multiband_length():
    obs = MultiBandObservation(
        {
            "g": make_observation(),
            "r": make_observation(),
            "i": make_observation(),
        }
    )

    assert len(obs) == 3
