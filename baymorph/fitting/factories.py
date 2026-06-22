"""Model factories used by fitting workflows."""

from __future__ import annotations

from baymorph.models import (
    ConvolvedModel,
    DoubleSersicModel,
    SersicModel,
)


class SkyModelWrapper:
    """Add a constant sky level to a rendered model."""

    def __init__(
        self,
        model,
        sky,
    ):
        self.model = model
        self.sky = sky

    def render(
        self,
        shape,
    ):
        return self.model.render(shape) + self.sky

def make_single_sersic_psf_sky_model(
    flux,
    re,
    n,
    q,
    theta,
    x0,
    y0,
    sky,
    *,
    psf,
):
    """Build a PSF-convolved single-Sérsic model with constant sky."""
    galaxy = SersicModel(
        flux=flux,
        re=re,
        n=n,
        q=q,
        theta=theta,
        x0=x0,
        y0=y0,
    )

    convolved_galaxy = ConvolvedModel(
        model=galaxy,
        psf=psf,
    )

    return SkyModelWrapper(
        model=convolved_galaxy,
        sky=sky,
    )

def make_double_sersic_psf_sky_model(
    flux_bulge,
    re_bulge,
    n_bulge,
    flux_disk,
    re_disk,
    n_disk,
    q,
    theta,
    x0,
    y0,
    sky,
    *,
    psf,
):
    """Build a PSF-convolved double-Sérsic model with constant sky."""
    bulge = SersicModel(
        flux=flux_bulge,
        re=re_bulge,
        n=n_bulge,
        q=q,
        theta=theta,
        x0=x0,
        y0=y0,
    )

    disk = SersicModel(
        flux=flux_disk,
        re=re_disk,
        n=n_disk,
        q=q,
        theta=theta,
        x0=x0,
        y0=y0,
    )

    galaxy = DoubleSersicModel(
        bulge=bulge,
        disk=disk,
    )

    convolved_galaxy = ConvolvedModel(
        model=galaxy,
        psf=psf,
    )

    return SkyModelWrapper(
        model=convolved_galaxy,
        sky=sky,
    )