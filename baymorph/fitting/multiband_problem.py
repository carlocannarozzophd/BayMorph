"""Backend-neutral inference problem for simultaneous multi-band fitting."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from baymorph.fitting.multiband import MultiBandParameterLayout
from baymorph.fitting.problem import InferenceProblem
from baymorph.fitting.representations import RepresentationFactory
from baymorph.likelihood.multiband import MultiBandLikelihood

FloatArray = NDArray[np.float64]


class MultiBandInferenceProblem(InferenceProblem):
    """Inference problem with configurable parameter coupling across bands."""

    def __init__(
        self,
        *,
        representation_factory: RepresentationFactory,
        parameter_layout: MultiBandParameterLayout,
        likelihood: MultiBandLikelihood,
    ) -> None:
        if not isinstance(
            representation_factory,
            RepresentationFactory,
        ):
            raise TypeError(
                "representation_factory must be a RepresentationFactory."
            )

        if not isinstance(
            parameter_layout,
            MultiBandParameterLayout,
        ):
            raise TypeError(
                "parameter_layout must be a MultiBandParameterLayout."
            )

        if not isinstance(
            likelihood,
            MultiBandLikelihood,
        ):
            raise TypeError(
                "likelihood must be a MultiBandLikelihood."
            )

        if representation_factory.observations != parameter_layout.bands:
            raise ValueError(
                "Representation factory and parameter layout bands must "
                "match in both names and order."
            )

        if parameter_layout.bands != likelihood.bands:
            raise ValueError(
                "Parameter layout and likelihood bands must match "
                "in both names and order."
            )

        self.representation_factory = representation_factory
        self.parameter_layout = parameter_layout
        self.likelihood = likelihood

    @property
    def ndim(self) -> int:
        """Return the global sampled dimension."""
        return self.parameter_layout.ndim

    @property
    def parameter_names(self) -> list[str]:
        """Return global parameter names in sampling order."""
        return self.parameter_layout.parameter_set.names

    def prior_transform(
        self,
        cube: FloatArray,
    ) -> FloatArray:
        """Transform unit-cube coordinates into physical parameters."""
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
            self.parameter_layout.parameter_set.transform(unit_values),
            dtype=float,
        )

        if transformed.shape != (self.ndim,):
            raise ValueError(
                "Prior transform must return one value per parameter."
            )

        if not np.all(np.isfinite(transformed)):
            raise ValueError(
                "Transformed parameter values must be finite."
            )

        return transformed

    def routed_parameters(
        self,
        theta: FloatArray,
    ) -> dict[str, dict[str, float]]:
        """Route one global vector to parameters required by each band."""
        values_array = np.asarray(
            theta,
            dtype=float,
        )

        if values_array.shape != (self.ndim,):
            raise ValueError(
                "Parameter vector must contain one value per parameter."
            )

        if not np.all(np.isfinite(values_array)):
            raise ValueError(
                "Parameter values must be finite."
            )

        global_values = {
            name: float(value)
            for name, value in zip(
                self.parameter_names,
                values_array,
                strict=True,
            )
        }

        return self.parameter_layout.route(global_values)

    def render_models(
        self,
        theta: FloatArray,
    ) -> dict[str, np.ndarray]:
        """Build and render one representation for every band."""
        representations = self.representation_factory.build_all(
            self.routed_parameters(theta)
        )

        rendered: dict[str, np.ndarray] = {}

        for band, representation in representations.items():
            render = getattr(
                representation,
                "render",
                None,
            )

            if not callable(render):
                raise TypeError(
                    f"Representation for band '{band}' must expose a "
                    "callable render(shape=...) method."
                )

            expected_shape = self.likelihood.image_shapes[band]

            model_image = np.asarray(
                render(shape=expected_shape),
                dtype=float,
            )

            if model_image.shape != expected_shape:
                raise ValueError(
                    f"Rendered representation for band '{band}' has an "
                    "incompatible shape."
                )

            if not np.all(np.isfinite(model_image)):
                raise ValueError(
                    f"Rendered representation for band '{band}' contains "
                    "non-finite values."
                )

            rendered[band] = model_image

        return rendered

    def log_likelihood(
        self,
        theta: FloatArray,
    ) -> float:
        """Evaluate the joint multi-band log-likelihood."""
        return self.likelihood(
            self.render_models(theta)
        )
