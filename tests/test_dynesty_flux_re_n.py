from baymorph import (
    DynestyFluxReNFitter,
    SersicModel,
    make_constant_noise_map,
)


def test_dynesty_flux_re_n_fit_runs():
    true_flux = 1000.0
    true_re = 8.0
    true_n = 2.0

    data = SersicModel(
        flux=true_flux,
        re=true_re,
        n=true_n,
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

    def model_factory(flux, re, n):
        return SersicModel(
            flux=flux,
            re=re,
            n=n,
            q=0.8,
            theta=0.0,
            x0=30,
            y0=30,
        )

    fitter = DynestyFluxReNFitter(
        model_factory=model_factory,
        data=data,
        noise=noise,
        flux_min=100,
        flux_max=2000,
        re_min=2,
        re_max=20,
        n_min=0.3,
        n_max=10,
    )

    result = fitter.run(
        nlive=50,
    )

    assert result.samples.shape[1] == 3