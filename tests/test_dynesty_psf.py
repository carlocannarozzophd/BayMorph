from baymorph import (
    DynestyFitter,
    GaussianPSF,
    SersicModel,
    make_constant_noise_map,
)


def test_dynesty_accepts_psf():

    data = SersicModel(
        flux=1000,
        re=8,
        n=2,
        q=0.8,
        theta=0,
        x0=30,
        y0=30,
    ).render(
        shape=(60, 60)
    )

    noise = make_constant_noise_map(
        data.shape,
        sigma=1.0,
    )

    psf = GaussianPSF(
        fwhm=3.0,
    )

    def model_factory(flux):
        return SersicModel(
            flux=flux,
            re=8,
            n=2,
            q=0.8,
            theta=0,
            x0=30,
            y0=30,
        )

    fitter = DynestyFitter(
        model_factory=model_factory,
        data=data,
        noise=noise,
        parameter_names=[
            "flux",
        ],
        prior_bounds={
            "flux": (
                100,
                2000,
            ),
        },
        psf=psf,
    )

    result = fitter.run(
        nlive=50,
        print_progress=False,
    )

    assert result.samples.shape[1] == 1