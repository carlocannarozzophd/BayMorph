"""Production workflow for joint multi-observation image inference."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from baymorph.core.observation import Observation
from baymorph.fitting import make_sampler
from baymorph.fitting.multiband_problem import (
    MultiBandInferenceProblem,
)
from baymorph.fitting.representations import (
    RepresentationFactory,
)
from baymorph.likelihood import make_likelihood
from baymorph.likelihood.multiband import (
    MultiBandLikelihood,
)
from baymorph.models.geometry import get_model_geometry
from baymorph.observation.geometry import (
    slice_wcs_to_cutout,
)
from baymorph.observation.model_parameter_transforms import (
    SkyModelParameterTransform,
)
from baymorph.pipeline.image_preparation import (
    load_image_preparation_products,
)
from baymorph.pipeline.multiband_config import (
    multiband_parameter_frame,
    resolve_multiband_config,
)
from baymorph.pipeline.real_fit import (
    _load_psf_from_config,
    _resolve_model_factory,
)
from baymorph.results import InferenceResult


def build_multiband_problem_from_config(
    config: str | Path | Mapping[str, Any],
) -> MultiBandInferenceProblem:
    """Build a production multi-observation inference problem.

    Parameters
    ----------
    config : mapping, str, or pathlib.Path
        Joint multi-observation configuration or path to its YAML file.

    Returns
    -------
    MultiBandInferenceProblem
        Fully constructed backend-neutral inference problem.

    Notes
    -----
    This function constructs the scientific inference problem but does not
    instantiate or execute an inference backend. It is therefore suitable
    for deterministic validation, likelihood inspection, and testing.
    """
    raw_config, base_dir = _load_multiband_config(
        config
    )

    return _build_multiband_problem(
        raw_config,
        base_dir=base_dir,
    )


def run_multiband_fit_from_config(
    config: str | Path | Mapping[str, Any],
) -> InferenceResult:
    """Run production joint multi-observation image inference.

    Parameters
    ----------
    config : mapping, str, or pathlib.Path
        Joint multi-observation configuration or path to its YAML file.

    Returns
    -------
    InferenceResult
        Canonical joint inference result produced by the configured backend.

    Raises
    ------
    TypeError
        If the configuration is not a mapping or filesystem path.
    ValueError
        If the joint configuration or required sky-frame geometry is invalid.
    FileNotFoundError
        If required image-preparation, PSF, or source-image products are
        unavailable.
    """
    raw_config, base_dir = _load_multiband_config(
        config
    )

    fit_config = raw_config["fit"]

    sampler_config = fit_config.get(
        "sampler"
    )

    if not isinstance(
        sampler_config,
        Mapping,
    ):
        raise ValueError(
            "Multi-observation fit configuration requires a "
            "'sampler' mapping."
        )

    models = fit_config["models"]
    model_name = models[0]

    problem = _build_multiband_problem(
        raw_config,
        base_dir=base_dir,
    )

    backend = make_sampler(
        sampler_config,
        problem=problem,
    )

    return backend.fit(
        progress_label=(
            f"multi_observation_{model_name}"
        ),
    )


def _build_multiband_problem(
    raw_config: Mapping[str, Any],
    *,
    base_dir: str | Path,
) -> MultiBandInferenceProblem:
    """Build a multi-observation problem from an already loaded config."""
    observation_configs, parameter_layout = (
        resolve_multiband_config(
            raw_config,
            base_dir=base_dir,
        )
    )

    fit_config = raw_config["fit"]

    models = fit_config["models"]
    model_name = models[0]

    likelihood_config = fit_config.get(
        "likelihood",
        {
            "type": "gaussian",
            "include_normalization": True,
        },
    )

    parameter_frame = multiband_parameter_frame(
        raw_config
    )

    image_products = {}
    psfs = {}

    for observation_name, observation_config in observation_configs.items():
        try:
            image_products[
                observation_name
            ] = load_image_preparation_products(
                observation_config
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "Image-preparation products are required for "
                f"observation '{observation_name}'."
            ) from exc

        try:
            psfs[
                observation_name
            ] = _load_psf_from_config(
                observation_config
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "PSF products are required for observation "
                f"'{observation_name}'."
            ) from exc

    likelihood = MultiBandLikelihood(
        {
            observation_name: make_likelihood(
                likelihood_config,
                data=products.prepared_data,
                noise=products.noise,
                mask=products.fitting_mask,
            )
            for observation_name, products in image_products.items()
        }
    )

    parameter_transforms = None

    if parameter_frame == "sky":
        parameter_transforms = _build_sky_parameter_transforms(
            observation_configs=observation_configs,
            image_products=image_products,
            model_name=model_name,
        )

    representation_factory = RepresentationFactory(
        factory=_resolve_model_factory(
            model_name
        ),
        fixed_kwargs={
            observation_name: {
                "psf": psfs[
                    observation_name
                ],
            }
            for observation_name in parameter_layout.bands
        },
        parameter_transforms=parameter_transforms,
    )

    return MultiBandInferenceProblem(
        representation_factory=representation_factory,
        parameter_layout=parameter_layout,
        likelihood=likelihood,
    )


def _build_sky_parameter_transforms(
    *,
    observation_configs: Mapping[str, Mapping[str, Any]],
    image_products: Mapping[str, Any],
    model_name: str,
) -> dict[str, SkyModelParameterTransform]:
    """Build fitted-grid sky-to-detector transforms for all observations."""
    geometry = get_model_geometry(
        model_name
    )

    transforms = {}

    for observation_name, observation_config in observation_configs.items():
        image_path = observation_config.get(
            "data",
            {},
        ).get(
            "image_path"
        )

        if not image_path:
            raise ValueError(
                "Sky-frame inference requires data.image_path for "
                f"observation '{observation_name}'."
            )

        observation = Observation.from_fits(
            image_path
        )

        wcs = observation.metadata.wcs

        if wcs is None:
            raise ValueError(
                "Sky-frame inference requires a valid celestial WCS for "
                f"observation '{observation_name}'."
            )

        bounds = image_products[
            observation_name
        ].cutout_bounds

        if not isinstance(
            bounds,
            Mapping,
        ):
            raise ValueError(
                "Sky-frame inference requires saved cutout bounds for "
                f"observation '{observation_name}'."
            )

        required_bounds = {
            "xmin",
            "xmax",
            "ymin",
            "ymax",
        }

        if set(bounds) != required_bounds:
            raise ValueError(
                "Sky-frame inference requires cutout bounds containing "
                "exactly xmin, xmax, ymin, and ymax for observation "
                f"'{observation_name}'."
            )

        cutout_wcs = slice_wcs_to_cutout(
            wcs,
            xmin=bounds["xmin"],
            xmax=bounds["xmax"],
            ymin=bounds["ymin"],
            ymax=bounds["ymax"],
        )

        transforms[
            observation_name
        ] = SkyModelParameterTransform(
            wcs=cutout_wcs,
            geometry=geometry,
        )

    return transforms


def _load_multiband_config(
    config: str | Path | Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    Path,
]:
    """Load a raw joint configuration and its path-resolution directory."""
    if isinstance(
        config,
        (str, Path),
    ):
        path = Path(
            config
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Multi-band config file not found: {path}"
            )

        with path.open(
            "r"
        ) as stream:
            loaded = yaml.safe_load(
                stream
            )

        if not isinstance(
            loaded,
            dict,
        ):
            raise TypeError(
                "Multi-band YAML top level must be a mapping."
            )

        return (
            loaded,
            path.parent,
        )

    if isinstance(
        config,
        Mapping,
    ):
        return (
            dict(config),
            Path.cwd(),
        )

    raise TypeError(
        "config must be a path or a mapping."
    )
