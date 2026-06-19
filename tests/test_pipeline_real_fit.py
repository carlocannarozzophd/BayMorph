import numpy as np

from baymorph.pipeline.real_fit import _build_parameter_set

def test_build_parameter_set():

    data = np.ones((20, 20))
    noise = np.ones((20, 20))
    mask = np.zeros((20, 20), dtype=bool)

    names, parameters, flux_total, sky_guess = (
        _build_parameter_set(
            data,
            noise,
            mask,
            target_x=10,
            target_y=10,
        )
    )

    assert len(names) == 11
    assert len(parameters.parameters) == 11

    assert flux_total > 0
    assert np.isfinite(sky_guess)