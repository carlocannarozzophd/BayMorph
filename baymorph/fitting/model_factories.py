"""Model factories used by BayMorph fitting workflows.

The factories in this module build the *observed source* model of one
registered morphology: intrinsic components composed with the observation
PSF. They never add a background. The residual additive background is an
observation effect owned by :mod:`baymorph.observation.background` and is
applied by :class:`ObservedModelFactory`, which is the single production
factory used by both the single-observation and the joint multi-observation
inference paths (AD-031, AD-044).
"""

from __future__ import annotations

from typing import Any

from baymorph.models import (
    BulgeDiskModel,
    CoreSersicComponent,
    DoubleSersicModel,
    ExponentialComponent,
    GeneralizedSersicComponent,
    PointSourceModel,
    SersicModel,
    SersicPointSourceModel,
    get_model,
)
from baymorph.observation.background import (
    BackgroundAppliedModel,
    background_parameter_names,
    make_background_effect,
)
from baymorph.observation.composite import ObservedCompositeModel
from baymorph.observation.convolved import ConvolvedModel


def make_single_sersic_psf_model(
    *,
    flux: float,
    re: float,
    n: float,
    q: float,
    theta: float,
    x0: float,
    y0: float,
    psf,
):
    """Build a PSF-convolved single-Sérsic model.

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
    psf : object
        Observation PSF.

    Returns
    -------
    ConvolvedModel
        PSF-convolved Sérsic model without additive background.
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

    return ConvolvedModel(
        model=galaxy,
        psf=psf,
    )


def make_double_sersic_psf_model(
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
    psf,
):
    """Build a PSF-convolved labelled two-Sérsic model.

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
    psf : object
        Observation PSF.

    Returns
    -------
    ConvolvedModel
        PSF-convolved two-Sérsic model without additive background.
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

    return ConvolvedModel(
        model=galaxy,
        psf=psf,
    )


def make_sersic_point_source_psf_model(
    *,
    flux_sersic: float,
    re: float,
    n: float,
    q: float,
    theta: float,
    flux_point_source: float,
    x0: float,
    y0: float,
    psf,
):
    """Build an observed Sérsic plus point-source model.

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
    psf : object
        Observation PSF.

    Returns
    -------
    ObservedCompositeModel
        Observed Sérsic plus point-source model without additive background.
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

    return ObservedCompositeModel.from_sersic_point_source(
        morphology,
        psf=psf,
    )


def make_bulge_disk_psf_model(
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
    psf,
):
    """Build a PSF-convolved Sérsic-bulge plus exponential-disc model.

    Returns
    -------
    ConvolvedModel
        PSF-convolved bulge-disc model without additive background.
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

    return ConvolvedModel(
        model=morphology,
        psf=psf,
    )


def make_boxy_sersic_psf_model(
    *,
    flux: float,
    re: float,
    n: float,
    q: float,
    c0: float,
    theta: float,
    x0: float,
    y0: float,
    psf,
):
    """Build a PSF-convolved Sérsic model on generalized ellipses (AD-048).

    Parameters
    ----------
    flux : float
        Total flux of the profile.
    re : float
        Semi-major radius of the half-light isophote, in pixels.
    n : float
        Sérsic index.
    q : float
        Axis ratio of the isophotes.
    c0 : float
        Isophote shape: positive for boxy, negative for discy isophotes.
    theta : float
        Position angle.
    x0, y0 : float
        Centre in pixel coordinates.
    psf : object
        Observation PSF.

    Returns
    -------
    ConvolvedModel
        PSF-convolved model without additive background.
    """
    galaxy = GeneralizedSersicComponent(
        flux=flux,
        re=re,
        n=n,
        q=q,
        c0=c0,
        theta=theta,
        x0=x0,
        y0=y0,
    )

    return ConvolvedModel(
        model=galaxy,
        psf=psf,
    )


def make_core_sersic_psf_model(
    *,
    flux: float,
    re: float,
    n: float,
    rb: float,
    gamma: float,
    alpha: float,
    q: float,
    theta: float,
    x0: float,
    y0: float,
    psf,
):
    """Build a PSF-convolved core-Sérsic model (AD-048).

    Parameters
    ----------
    flux : float
        Total flux of the profile.
    re : float
        Half-light radius of the outer Sérsic profile, in pixels.
    n : float
        Sérsic index of the outer profile.
    rb : float
        Break radius, in pixels.
    gamma : float
        Slope of the inner power law.
    alpha : float
        Sharpness of the transition.
    q : float
        Axis ratio.
    theta : float
        Position angle.
    x0, y0 : float
        Centre in pixel coordinates.
    psf : object
        Observation PSF.

    Returns
    -------
    ConvolvedModel
        PSF-convolved model without additive background.
    """
    galaxy = CoreSersicComponent(
        flux=flux,
        re=re,
        n=n,
        rb=rb,
        gamma=gamma,
        alpha=alpha,
        q=q,
        theta=theta,
        x0=x0,
        y0=y0,
    )

    return ConvolvedModel(
        model=galaxy,
        psf=psf,
    )


_MODEL_FACTORIES = {
    "SingleSersicModelFactory": make_single_sersic_psf_model,
    "DoubleSersicModelFactory": make_double_sersic_psf_model,
    "SersicPointSourceModelFactory": make_sersic_point_source_psf_model,
    "BulgeDiskModelFactory": make_bulge_disk_psf_model,
    "BoxySersicModelFactory": make_boxy_sersic_psf_model,
    "CoreSersicModelFactory": make_core_sersic_psf_model,
}


def resolve_model_factory(
    model_name: str,
):
    """Return the production observed-source factory of a registered model.

    Parameters
    ----------
    model_name : str
        Registered BayMorph model name.

    Returns
    -------
    callable
        Keyword-only factory accepting the model morphology parameters and
        ``psf`` and returning an observed source model.

    Raises
    ------
    ValueError
        If the registered definition names an unknown factory.
    """
    model_definition = get_model(
        model_name
    )

    try:
        return _MODEL_FACTORIES[
            model_definition.factory_name
        ]
    except KeyError as exc:
        raise ValueError(
            "Unsupported model factory: "
            f"{model_definition.factory_name}"
        ) from exc


class ObservedModelFactory:
    """Pickle-safe factory of one observed source-plus-background model.

    This is the production model factory. It composes the registered
    morphology with the observation PSF and applies the declared residual
    background as an observation effect (AD-031). The same factory serves
    single-observation inference, where the PSF is bound at construction,
    and joint multi-observation inference, where each observation injects
    its own PSF at call time through the representation factory (AD-044).

    Parameters
    ----------
    model_name : str
        Registered BayMorph morphology model.
    background_model : {"constant", "plane"}, default="constant"
        Residual additive background family.
    psf : object, optional
        Observation PSF bound at construction. When omitted, ``psf`` must be
        passed at call time.

    Attributes
    ----------
    morphology_parameter_names : tuple of str
        Ordered morphology parameters of the registered model.
    background_parameter_names : tuple of str
        Ordered background parameters of the declared background family.
    parameter_names : tuple of str
        Complete ordered parameter contract expected by ``__call__``.
    """

    def __init__(
        self,
        model_name: str,
        *,
        background_model: str = "constant",
        psf: Any = None,
    ) -> None:
        self.model_name = str(
            model_name
        )
        self.background_model = str(
            background_model
        ).strip().lower()
        self.psf = psf

        model_definition = get_model(
            self.model_name
        )

        self.morphology_parameter_names = tuple(
            model_definition.parameter_names
        )
        self.background_parameter_names = tuple(
            background_parameter_names(
                self.background_model
            )
        )
        self.parameter_names = (
            self.morphology_parameter_names
            + self.background_parameter_names
        )

        # Resolve eagerly so an unsupported registered factory fails at
        # construction rather than inside the sampler.
        resolve_model_factory(
            self.model_name
        )

    def __call__(
        self,
        *,
        psf: Any = None,
        **parameters: float,
    ) -> BackgroundAppliedModel:
        """Build one observed source-plus-background model.

        Parameters
        ----------
        psf : object, optional
            Observation PSF for this call. Required when no PSF was bound at
            construction and forbidden otherwise.
        **parameters : float
            Exactly the morphology and background parameters listed in
            ``parameter_names``.

        Returns
        -------
        BackgroundAppliedModel
            Observed source model with the residual background applied.
        """
        if psf is None:
            psf = self.psf
        elif self.psf is not None:
            raise ValueError(
                "ObservedModelFactory received a PSF at call time although "
                "one was bound at construction."
            )

        if psf is None:
            raise ValueError(
                "ObservedModelFactory requires a PSF, either bound at "
                "construction or supplied at call time."
            )

        supplied = set(
            parameters
        )
        expected = set(
            self.parameter_names
        )

        if supplied != expected:
            missing = sorted(
                expected.difference(
                    supplied
                )
            )
            extra = sorted(
                supplied.difference(
                    expected
                )
            )

            raise ValueError(
                f"ObservedModelFactory for model {self.model_name!r} with "
                f"{self.background_model!r} background expects exactly "
                f"{list(self.parameter_names)}. "
                f"Missing: {missing}. Extra: {extra}."
            )

        source_model = resolve_model_factory(
            self.model_name
        )(
            **{
                name: parameters[
                    name
                ]
                for name in self.morphology_parameter_names
            },
            psf=psf,
        )

        background = make_background_effect(
            self.background_model,
            level=parameters[
                "background_level"
            ],
            gradient_x=parameters.get(
                "background_gradient_x",
                0.0,
            ),
            gradient_y=parameters.get(
                "background_gradient_y",
                0.0,
            ),
        )

        return BackgroundAppliedModel(
            model=source_model,
            background=background,
        )


__all__ = [
    "ObservedModelFactory",
    "make_boxy_sersic_psf_model",
    "make_bulge_disk_psf_model",
    "make_core_sersic_psf_model",
    "make_double_sersic_psf_model",
    "make_sersic_point_source_psf_model",
    "make_single_sersic_psf_model",
    "resolve_model_factory",
]
