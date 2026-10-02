"""Backend-neutral inference problem for multi-observation fitting."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from baymorph.fitting.multi_observation import (
    MultiObservationParameterLayout,
)
from baymorph.fitting.problem import InferenceProblem
from baymorph.fitting.representations import RepresentationFactory
from baymorph.likelihood.multi_observation import (
    MultiObservationLikelihood,
)

FloatArray = NDArray[np.float64]


class MultiObservationInferenceProblem(InferenceProblem):
    """Inference problem with explicit coupling across observations."""

    def __init__(
        self,
        *,
        representation_factory: RepresentationFactory,
        parameter_layout: MultiObservationParameterLayout,
        likelihood: MultiObservationLikelihood,
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
            MultiObservationParameterLayout,
        ):
            raise TypeError(
                "parameter_layout must be a "
                "MultiObservationParameterLayout."
            )

        if not isinstance(
            likelihood,
            MultiObservationLikelihood,
        ):
            raise TypeError(
                "likelihood must be a MultiObservationLikelihood."
            )

        if (
            representation_factory.observations
            != parameter_layout.observations
        ):
            raise ValueError(
                "Representation factory and parameter layout observations "
                "must match in both names and order."
            )

        if (
            parameter_layout.observations
            != likelihood.observations
        ):
            raise ValueError(
                "Parameter layout and likelihood observations must match "
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
        """Return sampled global parameter names in sampling order."""
        return self.parameter_layout.parameter_set.names

    def prior_transform(
        self,
        cube: FloatArray,
    ) -> FloatArray:
        """Transform unit-cube coordinates into sampled physical values."""
        unit_values = np.asarray(
            cube,
            dtype=float,
        )

        if unit_values.shape != (
            self.ndim,
        ):
            raise ValueError(
                "Unit-cube sample must contain one value per "
                "sampled parameter."
            )

        if not np.all(
            np.isfinite(
                unit_values
            )
        ):
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
            self.parameter_layout.parameter_set.transform(
                unit_values
            ),
            dtype=float,
        )

        if transformed.shape != (
            self.ndim,
        ):
            raise ValueError(
                "Prior transform must return one value per "
                "sampled parameter."
            )

        if not np.all(
            np.isfinite(
                transformed
            )
        ):
            raise ValueError(
                "Transformed parameter values must be finite."
            )

        return transformed

    def routed_parameters(
        self,
        theta: FloatArray,
    ) -> dict[str, dict[str, float]]:
        """Route one sampled vector to every observation."""
        values_array = np.asarray(
            theta,
            dtype=float,
        )

        if values_array.shape != (
            self.ndim,
        ):
            raise ValueError(
                "Parameter vector must contain one value per "
                "sampled parameter."
            )

        if not np.all(
            np.isfinite(
                values_array
            )
        ):
            raise ValueError(
                "Parameter values must be finite."
            )

        sampled_values = {
            name: float(value)
            for name, value in zip(
                self.parameter_names,
                values_array,
                strict=True,
            )
        }

        return self.parameter_layout.route(
            sampled_values
        )

    def render_models(
        self,
        theta: FloatArray,
    ) -> dict[str, np.ndarray]:
        """Build and render one representation for every observation."""
        representations = (
            self.representation_factory.build_all(
                self.routed_parameters(
                    theta
                )
            )
        )

        rendered: dict[
            str,
            np.ndarray,
        ] = {}

        for observation, representation in representations.items():
            render = getattr(
                representation,
                "render",
                None,
            )

            if not callable(
                render
            ):
                raise TypeError(
                    f"Representation for observation '{observation}' "
                    "must expose a callable render(shape=...) method."
                )

            expected_shape = (
                self.likelihood.image_shapes[
                    observation
                ]
            )

            rendered[
                observation
            ] = self._validated_image(
                observation,
                render(
                    shape=expected_shape
                ),
            )

        return rendered

    def render_predictions(
        self,
        theta: FloatArray,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Render every expected image and its source part (AD-047).

        The source part excludes the residual background; a representation
        without a residual background is its own source part.
        """
        representations = (
            self.representation_factory.build_all(
                self.routed_parameters(
                    theta
                )
            )
        )

        models: dict[str, np.ndarray] = {}
        sources: dict[str, np.ndarray] = {}

        for observation, representation in representations.items():
            expected_shape = (
                self.likelihood.image_shapes[
                    observation
                ]
            )
            render_prediction = getattr(
                representation,
                "render_prediction",
                None,
            )

            if callable(render_prediction):
                model_image, source_image = render_prediction(
                    shape=expected_shape
                )
            else:
                render = getattr(
                    representation,
                    "render",
                    None,
                )

                if not callable(render):
                    raise TypeError(
                        f"Representation for observation '{observation}' "
                        "must expose a callable render(shape=...) method."
                    )

                model_image = render(shape=expected_shape)
                source_image = model_image

            models[observation] = self._validated_image(
                observation,
                model_image,
            )
            sources[observation] = self._validated_image(
                observation,
                source_image,
            )

        return models, sources

    def _validated_image(
        self,
        observation: str,
        image,
    ) -> np.ndarray:
        """Validate the shape and values of one rendered image."""
        model_image = np.asarray(
            image,
            dtype=float,
        )

        if model_image.shape != self.likelihood.image_shapes[observation]:
            raise ValueError(
                f"Rendered representation for observation "
                f"'{observation}' has an incompatible shape."
            )

        if not np.all(
            np.isfinite(
                model_image
            )
        ):
            raise ValueError(
                f"Rendered representation for observation "
                f"'{observation}' contains non-finite values."
            )

        return model_image

    def log_likelihood(
        self,
        theta: FloatArray,
    ) -> float:
        """Evaluate the joint multi-observation log-likelihood."""
        if self.likelihood.requires_source:
            models, sources = self.render_predictions(
                theta
            )

            return float(
                self.likelihood(
                    models,
                    source_images=sources,
                )
            )

        return float(
            self.likelihood(
                self.render_models(
                    theta
                )
            )
        )
