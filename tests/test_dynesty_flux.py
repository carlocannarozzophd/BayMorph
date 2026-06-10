from baymorph import (
    DynestyFluxFitter,
    SersicModel,
    make_constant_noise_map,
)


def test_dynesty_flux_fit_runs():

    true_flux = 1000.0

    data = SersicModel(
        flux=true_flux,
        re=8,
        n=2,
        q=0.8,
        theta=0.0,
        x0=30,
        y0=30,
    ).render(
        shape=(60, 60)
    )

    noise = make_constant_noise_map(
        data.shape,
        sigma=1.0,
    )

    def model_factory(flux):
        return SersicModel(
            flux=flux,
            re=8,
            n=2,
            q=0.8,
            theta=0.0,
            x0=30,
            y0=30,
        )

    fitter = DynestyFluxFitter(
        model_factory=model_factory,
        data=data,
        noise=noise,
        flux_min=100,
        flux_max=2000,
    )

    result = fitter.run(
        nlive=50,
    )

    assert result.samples.shape[1] == 1