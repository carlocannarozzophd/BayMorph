import numpy as np

from baymorph.fitting.dynesty_generic import DynestyFitter
from baymorph.models import SersicModel


def test_single_sersic_fitter_no_noise_recovers_truth():
    shape = (51, 51)

    truth = {
        "flux": 1.0e5,
        "re": 6.0,
        "n": 2.0,
        "q": 0.7,
        "theta": np.deg2rad(25.0),
        "x0": 25.0,
        "y0": 25.0,
    }

    data = SersicModel(**truth).render(shape)
    noise = np.ones(shape)
    # In BayMorph, mask=True means excluded pixel.
    mask = np.zeros(shape, dtype=bool)

    def make_model(flux, re, n, q, theta, x0, y0):
        return SersicModel(
            flux=flux,
            re=re,
            n=n,
            q=q,
            theta=theta,
            x0=x0,
            y0=y0,
        )

    parameter_names = list(truth)

    prior_bounds = {
        "flux": (0.8e5, 1.2e5),
        "re": (4.0, 8.0),
        "n": (1.0, 3.0),
        "q": (0.5, 0.9),
        "theta": (-0.5 * np.pi, 0.5 * np.pi),
        "x0": (24.0, 26.0),
        "y0": (24.0, 26.0),
    }

    fitter = DynestyFitter(
        model_factory=make_model,
        data=data,
        noise=noise,
        parameter_names=parameter_names,
        prior_bounds=prior_bounds,
        mask=mask,
    )

    result = fitter.run(
        nlive=150,
        sample="rwalk",
        bound="multi",
        dlogz=0.1,
        seed=42,
        print_progress=False,
    )

    recovered = {
        name: np.median(result.samples[:, i])
        for i, name in enumerate(parameter_names)
    }

    assert np.isclose(recovered["flux"], truth["flux"], rtol=0.02)
    assert np.isclose(recovered["re"], truth["re"], rtol=0.05)
    assert np.isclose(recovered["n"], truth["n"], rtol=0.10)
    assert np.isclose(recovered["q"], truth["q"], rtol=0.05)
    assert np.isclose(recovered["theta"], truth["theta"], atol=np.deg2rad(3.0))
    assert np.isclose(recovered["x0"], truth["x0"], atol=0.2)
    assert np.isclose(recovered["y0"], truth["y0"], atol=0.2)
