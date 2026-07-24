"""Native nested-sampling backend for BayMorph morphology fits."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from baymorph.fitting.geometry import (
    GeometryModel,
    IdentityTransform,
    LogTransform,
    ParameterGeometry,
)
from baymorph.fitting.geometry.nested import (
    GeometryNestedSampler,
    NestedSamplerConfig,
)
from baymorph.fitting.parameters import ParameterSet
from baymorph.likelihood import ImageLikelihood

if TYPE_CHECKING:
    from baymorph.results import PosteriorResult


class NativeFitter:
    """Fit a BayMorph image model with the native nested sampler.

    Parameters
    ----------
    model_factory : callable
        Callable receiving physical model parameters as keyword arguments.
    parameters : ParameterSet
        Ordered fitted-parameter definitions.
    likelihood : ImageLikelihood
        Configured image likelihood.
    config : mapping, optional
        Native-sampler configuration supplied by the pipeline.
    """

    def __init__(
        self,
        *,
        model_factory,
        parameters: ParameterSet,
        likelihood: ImageLikelihood,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        if not callable(model_factory):
            raise TypeError(
                "model_factory must be callable."
            )

        if not isinstance(
            parameters,
            ParameterSet,
        ):
            raise TypeError(
                "parameters must be a ParameterSet instance."
            )

        if not isinstance(
            likelihood,
            ImageLikelihood,
        ):
            raise TypeError(
                "likelihood must be an ImageLikelihood instance."
            )

        if parameters.ndim < 1:
            raise ValueError(
                "ParameterSet must contain at least one parameter."
            )

        self.model_factory = model_factory
        self.parameters = parameters
        self.likelihood = likelihood
        self.config = dict(
            config or {}
        )
        self.geometry = make_morphology_geometry(
            parameters
        )

    def log_likelihood(
        self,
        values: np.ndarray,
    ) -> float:
        """Evaluate the image likelihood for physical parameters."""
        parameter_values = {
            name: float(value)
            for name, value in zip(
                self.parameters.names,
                values,
                strict=True,
            )
        }

        model = self.model_factory(
            **parameter_values
        )

        model_image = model.render(
            self.likelihood.image_shape
        )

        return float(
            self.likelihood(model_image)
        )

    def run(
        self,
        nlive: int | None = None,
        sample: str | None = None,
        bound: str = "multi",
        dlogz: float | None = None,
        bootstrap: int | None = None,
        print_progress: bool = True,
        n_processes: int = 1,
        queue_size: int | None = None,
        seed: int | None = None,
        progress_label: str | None = None,
    ) -> PosteriorResult:
        """Run the native sampler through the common fitter API.

        Parameters matching Dynesty but not yet used by the native backend
        are accepted to preserve pipeline compatibility.
        """
        del bound
        del bootstrap
        del print_progress
        del progress_label

        sample = str(
            sample
            if sample is not None
            else self.config.get(
                "sample",
                "auto",
            )
        )

        nlive = int(
            nlive
            if nlive is not None
            else self.config.get(
                "nlive",
                500,
            )
        )

        seed = (
            seed
            if seed is not None
            else self.config.get(
                "seed"
            )
        )

        if sample not in {
            "auto",
            "prior",
        }:
            raise ValueError(
                "The native backend currently supports only "
                "sample='auto' or sample='prior'."
            )

        if n_processes != 1:
            raise ValueError(
                "The native backend currently requires "
                "n_processes=1."
            )

        if queue_size not in {
            None,
            1,
        }:
            raise ValueError(
                "The native backend does not currently support "
                "queue_size > 1."
            )

        sampler_config = NestedSamplerConfig(
            nlive=nlive,
            dlogz=(
                float(dlogz)
                if dlogz is not None
                else float(
                    self.config.get(
                        "dlogz",
                        0.01,
                    )
                )
            ),
            max_iterations=self.config.get(
                "max_iterations"
            ),
            max_likelihood_calls=self.config.get(
                "max_likelihood_calls"
            ),
            seed=seed,
        )

        sampler = GeometryNestedSampler(
            geometry=self.geometry,
            parameter_set=self.parameters,
            likelihood=self.log_likelihood,
            config=sampler_config,
        )

        return sampler.run()


def make_morphology_geometry(
    parameters: ParameterSet,
) -> GeometryModel:
    """Build morphology geometry from BayMorph parameter names."""
    return GeometryModel(
        [
            _parameter_geometry(
                name
            )
            for name in parameters.names
        ]
    )


def _parameter_geometry(
    name: str,
) -> ParameterGeometry:
    """Return morphology-aware geometry for one parameter."""
    if name in {
        "x0",
        "y0",
    }:
        return ParameterGeometry(
            name=name,
            transform=IdentityTransform(),
            block="position",
        )

    if name == "sky":
        return ParameterGeometry(
            name=name,
            transform=IdentityTransform(),
            block="background",
        )

    if (
        name == "flux"
        or name.startswith("flux_")
    ):
        return ParameterGeometry(
            name=name,
            transform=LogTransform(),
            block=_component_block(
                "photometry",
                name,
            ),
        )

    if (
        name == "re"
        or name.startswith("re_")
    ):
        return ParameterGeometry(
            name=name,
            transform=LogTransform(),
            block=_component_block(
                "structure",
                name,
            ),
        )

    if (
        name == "n"
        or name.startswith("n_")
    ):
        return ParameterGeometry(
            name=name,
            transform=LogTransform(),
            block=_component_block(
                "structure",
                name,
            ),
        )

    if (
        name == "q"
        or name.startswith("q_")
    ):
        return ParameterGeometry(
            name=name,
            transform=IdentityTransform(),
            block=_component_block(
                "shape",
                name,
            ),
        )

    if (
        name == "theta"
        or name.startswith("theta_")
    ):
        return ParameterGeometry(
            name=name,
            transform=IdentityTransform(),
            block=_component_block(
                "shape",
                name,
            ),
            period=float(
                np.pi
            ),
        )

    raise ValueError(
        "The native sampler has no morphology geometry "
        f"for parameter {name!r}."
    )


def _component_block(
    prefix: str,
    name: str,
) -> str:
    """Return a shared or component-specific proposal block."""
    if "_" not in name:
        return prefix

    component = name.split(
        "_",
        1,
    )[1]

    return f"{prefix}_{component}"


__all__ = [
    "NativeFitter",
    "make_morphology_geometry",
]
