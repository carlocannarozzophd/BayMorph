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
        """Render the wrapped model and add the constant sky level."""
        return self.model.render(shape) + self.sky

    def render_components(
        self,
        shape,
    ):
        """Render the wrapped model components without sky."""
        return self.model.render_components(shape)


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


def _parse_double_sersic_arguments(args):
    """Parse old and new double-Sérsic positional argument conventions."""
    if len(args) == 13:
        return args

    if len(args) == 11:
        (
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
        ) = args

        return (
            flux_bulge,
            re_bulge,
            n_bulge,
            q,
            theta,
            flux_disk,
            re_disk,
            n_disk,
            q,
            theta,
            x0,
            y0,
            sky,
        )

    raise TypeError(
        "make_double_sersic_psf_sky_model expects either 11 old-style "
        "or 13 new-style positional parameters before psf"
    )


def make_double_sersic_psf_sky_model(
    *args,
    psf,
):
    """Build a PSF-convolved double-Sérsic model with constant sky."""
    (
        flux_bulge,
        re_bulge,
        n_bulge,
        q_bulge,
        theta_bulge,
        flux_disk,
        re_disk,
        n_disk,
        q_disk,
        theta_disk,
        x0,
        y0,
        sky,
    ) = _parse_double_sersic_arguments(args)

    bulge = SersicModel(
        flux=flux_bulge,
        re=re_bulge,
        n=n_bulge,
        q=q_bulge,
        theta=theta_bulge,
        x0=x0,
        y0=y0,
    )

    disk = SersicModel(
        flux=flux_disk,
        re=re_disk,
        n=n_disk,
        q=q_disk,
        theta=theta_disk,
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