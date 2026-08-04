"""Backend-neutral inference problem for one astronomical image."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from baymorph.fitting.parameters import ParameterSet
from baymorph.fitting.problem import InferenceProblem
from baymorph.likelihood import ImageLikelihood
from baymorph.simulation.psf import convolve_with_psf

FloatArray = NDArray[np.float64]


class ImageInferenceProblem(InferenceProblem):
    """Bayesian inference problem for one image and one morphology model."""

    def __init__(
        self,
        *,
        model_factory,
        parameters: ParameterSet,
        likelihood: ImageLikelihood,
        psf=None,
    ) -> None:
        if not callable(model_factory):
            raise TypeError("model_factory must be callable.")

        if not isinstance(parameters, ParameterSet):
            raise TypeError(
                "parameters must be a ParameterSet instance."
            )

        if not isinstance(likelihood, ImageLikelihood):
            raise TypeError(
                "likelihood must be an ImageLikelihood instance."
            )

        if parameters.ndim < 1:
            raise ValueError(
                "ParameterSet must contain at least one parameter."
            )

        if len(set(parameters.names)) != parameters.ndim:
            raise ValueError("Parameter names must be unique.")

        self.model_factory = model_factory
        self.parameters = parameters
        self.likelihood = likelihood
        self.psf = psf

    @property
    def ndim(self) -> int:
        """Return the number of sampled parameters."""
        return self.parameters.ndim

    @property
    def parameter_names(self) -> list[str]:
        """Return parameter names in sampling order."""
        return self.parameters.names

    @property
    def image_shape(self) -> tuple[int, ...]:
        """Return the image shape required by the likelihood."""
        return self.likelihood.image_shape

    def prior_transform(
        self,
        cube: FloatArray,
    ) -> FloatArray:
        """Transform one unit-cube vector to physical parameters."""
        unit_values = np.asarray(
            cube,
            dtype=float,
        )

        if unit_values.shape != (self.ndim,):
            raise ValueError(
                "Unit-cube sample must contain one value per parameter."
            )

        if not np.all(np.isfinite(unit_values)):
            raise ValueError(
                "Unit-cube sample values must be finite."
            )

        if np.any(
            (unit_values < 0.0)
            | (unit_values > 1.0)
        ):
            raise ValueError(
                "Unit-cube sample values must lie between zero and one."
            )

        transformed = np.asarray(
            self.parameters.transform(unit_values),
            dtype=float,
        )

        if transformed.shape != (self.ndim,):
            raise ValueError(
                "ParameterSet.transform() must return one value per "
                "parameter."
            )

        if not np.all(np.isfinite(transformed)):
            raise ValueError(
                "Transformed parameter values must be finite."
            )

        return transformed

    def parameter_values(
        self,
        theta: FloatArray,
    ) -> dict[str, float]:
        """Convert one parameter vector to named physical values."""
        theta_array = np.asarray(
            theta,
            dtype=float,
        )

        if theta_array.shape != (self.ndim,):
            raise ValueError(
                "Parameter vector must contain one value per parameter."
            )

        if not np.all(np.isfinite(theta_array)):
            raise ValueError(
                "Parameter values must be finite."
            )

        return {
            name: float(value)
            for name, value in zip(
                self.parameter_names,
                theta_array,
                strict=True,
            )
        }

    def render_model(
        self,
        theta: FloatArray,
    ) -> FloatArray:
        """Render the expected image for one physical parameter vector."""
        model = self.model_factory(
            **self.parameter_values(theta)
        )

        render = getattr(
            model,
            "render",
            None,
        )

        if not callable(render):
            raise TypeError(
                "model_factory must return an object with a callable "
                "render(shape=...) method."
            )

        model_image = np.asarray(
            render(shape=self.image_shape),
            dtype=float,
        )

        if self.psf is not None:
            model_image = np.asarray(
                convolve_with_psf(
                    model_image,
                    self.psf,
                ),
                dtype=float,
            )

        if model_image.shape != self.image_shape:
            raise ValueError(
                "Rendered model must have the same shape as the "
                "likelihood data."
            )

        if not np.all(np.isfinite(model_image)):
            raise ValueError(
                "Rendered model must contain only finite values."
            )

        return model_image

    def log_likelihood(
        self,
        theta: FloatArray,
    ) -> float:
        """Evaluate the image log-likelihood."""
        return float(
            self.likelihood(
                self.render_model(theta)
            )
        )
