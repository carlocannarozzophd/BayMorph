"""Production workflow for joint multi-band image inference."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from baymorph.fitting import make_sampler
from baymorph.fitting.model_factories import (
    make_single_sersic_psf_sky_model,
)
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
from baymorph.pipeline.image_preparation import (
    load_image_preparation_products,
)
from baymorph.pipeline.multiband_config import (
    resolve_multiband_config,
)
from baymorph.pipeline.real_fit import (
    _load_psf_from_config,
)
from baymorph.results import PosteriorResult


def run_multiband_fit_from_config(
    config: str | Path | Mapping[str, Any],
) -> PosteriorResult:
    """Run a production joint multi-band single-Sersic fit.

    Parameters
    ----------
    config : mapping, str, or pathlib.Path
        Joint multi-band configuration or path to its YAML file.

    Returns
    -------
    PosteriorResult
        Joint posterior result produced by the configured sampler.

    Raises
    ------
    TypeError
        If the configuration is not a mapping or filesystem path.
    ValueError
        If the joint configuration is invalid or prepared images have
        incompatible shapes.
    FileNotFoundError
        If required image-preparation or PSF products are unavailable.
    """
    raw_config, base_dir = _load_multiband_config(
        config
    )

    observation_configs, parameter_layout = (
        resolve_multiband_config(
            raw_config,
            base_dir=base_dir,
        )
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
            "Multi-band fit configuration requires a 'sampler' mapping."
        )

    likelihood_config = fit_config.get(
        "likelihood",
        {
            "type": "gaussian",
            "include_normalization": True,
        },
    )

    image_products = {}
    psfs = {}

    for band, observation_config in observation_configs.items():
        try:
            image_products[band] = (
                load_image_preparation_products(
                    observation_config
                )
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "Image-preparation products are required for "
                f"observation '{band}'."
            ) from exc

        try:
            psfs[band] = _load_psf_from_config(
                observation_config
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"PSF products are required for observation '{band}'."
            ) from exc

    _validate_compatible_image_shapes(
        image_products
    )

    likelihood = MultiBandLikelihood(
        {
            band: make_likelihood(
                likelihood_config,
                data=products.prepared_data,
                noise=products.noise,
                mask=products.fitting_mask,
            )
            for band, products in image_products.items()
        }
    )

    representation_factory = RepresentationFactory(
        factory=make_single_sersic_psf_sky_model,
        fixed_kwargs={
            band: {
                "psf": psfs[band],
            }
            for band in parameter_layout.bands
        },
    )

    problem = MultiBandInferenceProblem(
        representation_factory=representation_factory,
        parameter_layout=parameter_layout,
        likelihood=likelihood,
    )

    sampler = make_sampler(
        sampler_config,
        problem=problem,
    )

    return sampler.run(
        nlive=sampler_config["nlive"],
        sample=sampler_config.get(
            "sample",
            "auto",
        ),
        dlogz=sampler_config.get(
            "dlogz"
        ),
        n_processes=sampler_config.get(
            "n_processes",
            1,
        ),
        queue_size=sampler_config.get(
            "queue_size"
        ),
        seed=sampler_config.get(
            "seed"
        ),
        progress_label="multi_band_single_sersic",
    )


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


def _validate_compatible_image_shapes(
    image_products: Mapping[str, Any],
) -> None:
    """Require the same prepared-image shape for every observation."""
    shapes = {
        band: tuple(
            products.prepared_data.shape
        )
        for band, products in image_products.items()
    }

    if len(
        set(
            shapes.values()
        )
    ) != 1:
        formatted = ", ".join(
            f"{band}={shape}"
            for band, shape in shapes.items()
        )

        raise ValueError(
            "The initial production multi-band workflow requires "
            "all prepared observations to have the same image shape. "
            f"Received: {formatted}."
        )
