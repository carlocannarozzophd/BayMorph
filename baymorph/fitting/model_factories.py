"""Model factories used by BayMorph fitting workflows."""

from __future__ import annotations

from baymorph.models import (
    BulgeDiskModel,
    DoubleSersicModel,
    ExponentialComponent,
    PointSourceModel,
    SersicModel,
    SersicPointSourceModel,
)
from baymorph.observation.composite import ObservedCompositeModel
from baymorph.observation.convolved import ConvolvedModel


class SkyModelWrapper:
    """Add a constant sky level to a rendered model.

    Parameters
    ----------
    model : object
        Wrapped model exposing a ``render(shape)`` method.
    sky : float
        Constant sky level added to every rendered pixel.
    """

    def __init__(
        self,
        model,
        sky: float,
    ) -> None:
        self.model = model
        self.sky = float(sky)

    def render(
        self,
        shape: tuple[int, int],
    ):
        """Render the wrapped model and add the constant sky level.

        Parameters
        ----------
        shape : tuple of int
            Output image shape as ``(ny, nx)``.

        Returns
        -------
        ndarray
            Rendered model image including the constant sky level.
        """
        return self.model.render(shape) + self.sky

    def render_components(
        self,
        shape: tuple[int, int],
    ):
        """Render the wrapped model components without sky.

        Parameters
        ----------
        shape : tuple of int
            Output image shape as ``(ny, nx)``.

        Returns
        -------
        object
            Component rendering returned by the wrapped model.
        """
        return self.model.render_components(shape)


def make_single_sersic_psf_sky_model(
    *,
    flux: float,
    re: float,
    n: float,
    q: float,
    theta: float,
    x0: float,
    y0: float,
    sky: float,
    psf,
):
    """Build a PSF-convolved single-Sérsic model with constant sky.

    Parameters
    ----------
    flux : float
        Total Sérsic flux.
    re : float
        Effective radius in pixels.
    n : float
        Sérsic index.
    q : float
        Axis ratio.
    theta : float
        Position angle.
    x0, y0 : float
        Model centroid in pixel coordinates.
    sky : float
        Constant sky level.
    psf : array-like
        Point-spread function kernel.

    Returns
    -------
    SkyModelWrapper
        PSF-convolved Sérsic model with additive constant sky.
    """
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
    *,
    flux_bulge: float,
    re_bulge: float,
    n_bulge: float,
    q_bulge: float,
    theta_bulge: float,
    flux_disk: float,
    re_disk: float,
    n_disk: float,
    q_disk: float,
    theta_disk: float,
    x0: float,
    y0: float,
    sky: float,
    psf,
):
    """Build a PSF-convolved double-Sérsic model with constant sky.

    Parameters
    ----------
    flux_bulge, flux_disk : float
        Total fluxes of the two Sérsic components.
    re_bulge, re_disk : float
        Effective radii in pixels.
    n_bulge, n_disk : float
        Sérsic indices.
    q_bulge, q_disk : float
        Axis ratios.
    theta_bulge, theta_disk : float
        Position angles.
    x0, y0 : float
        Shared centroid in pixel coordinates.
    sky : float
        Constant sky level.
    psf : array-like
        Point-spread function kernel.

    Returns
    -------
    SkyModelWrapper
        PSF-convolved double-Sérsic model with additive constant sky.
    """
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


def make_sersic_point_source_psf_sky_model(
    *,
    flux_sersic: float,
    re: float,
    n: float,
    q: float,
    theta: float,
    flux_point_source: float,
    x0: float,
    y0: float,
    sky: float,
    psf,
):
    """Build an observed Sérsic plus point-source model with sky.

    Parameters
    ----------
    flux_sersic : float
        Total intrinsic Sérsic flux.
    re : float
        Sérsic effective radius in pixels.
    n : float
        Sérsic index.
    q : float
        Sérsic axis ratio.
    theta : float
        Sérsic position angle in radians.
    flux_point_source : float
        Total intrinsic unresolved-source flux.
    x0, y0 : float
        Shared source centre in pixel coordinates.
    sky : float
        Constant additive sky level.
    psf : object
        Observation PSF.

    Returns
    -------
    SkyModelWrapper
        Observed Sérsic plus point-source model with additive sky.
    """
    sersic = SersicModel(
        flux=flux_sersic,
        re=re,
        n=n,
        q=q,
        theta=theta,
        x0=x0,
        y0=y0,
    )

    point_source = PointSourceModel(
        flux=flux_point_source,
        x0=x0,
        y0=y0,
    )

    morphology = SersicPointSourceModel(
        sersic=sersic,
        point_source=point_source,
    )

    observed = ObservedCompositeModel.from_sersic_point_source(
        morphology,
        psf=psf,
    )

    return SkyModelWrapper(
        model=observed,
        sky=sky,
    )


def make_bulge_disk_psf_sky_model(
    *,
    flux_bulge: float,
    re_bulge: float,
    n_bulge: float,
    q_bulge: float,
    theta_bulge: float,
    flux_disk: float,
    re_disk: float,
    q_disk: float,
    theta_disk: float,
    x0: float,
    y0: float,
    sky: float,
    psf,
):
    """Build a PSF-convolved Sérsic-bulge plus exponential-disc model."""
    bulge = SersicModel(
        flux=flux_bulge,
        re=re_bulge,
        n=n_bulge,
        q=q_bulge,
        theta=theta_bulge,
        x0=x0,
        y0=y0,
    )

    disk = ExponentialComponent(
        flux=flux_disk,
        re=re_disk,
        q=q_disk,
        theta=theta_disk,
        x0=x0,
        y0=y0,
    )

    morphology = BulgeDiskModel(
        bulge=bulge,
        disk=disk,
    )

    observed = ConvolvedModel(
        model=morphology,
        psf=psf,
    )

    return SkyModelWrapper(
        model=observed,
        sky=sky,
    )

