"""Disk workspace for heterogeneous multi-observation validation cases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from astropy.io import fits
from astropy.wcs import WCS

from baymorph.fitting.model_factories import (
    make_single_sersic_psf_sky_model,
)
from baymorph.models.geometry import get_model_geometry
from baymorph.observation.model_parameter_transforms import (
    SkyModelParameterTransform,
)
from baymorph.simulation.psf import GaussianPSF
from baymorph.validation.multi_observation_truth import (
    MultiObservationGroundTruth,
)


@dataclass(frozen=True)
class MultiObservationValidationWorkspace:
    """Materialize one heterogeneous multi-observation validation case."""

    root: str | Path

    def __post_init__(self) -> None:
        root = Path(
            self.root
        )

        if root.exists() and not root.is_dir():
            raise ValueError(
                "Validation workspace root must be a directory."
            )

        object.__setattr__(
            self,
            "root",
            root,
        )

    def write_case(
        self,
        case_id: str,
        truth: MultiObservationGroundTruth,
        *,
        sampler: dict[str, Any] | None = None,
    ) -> Path:
        """Write synthetic FITS observations and joint validation configs."""
        case_id = self._validate_case_id(
            case_id
        )

        if truth.model_name != "single_sersic":
            raise ValueError(
                "Initial multi-observation validation supports "
                "only the single_sersic reference model."
            )

        case_dir = (
            self.root
            / case_id
        )
        input_dir = (
            case_dir
            / "input"
        )
        config_dir = (
            case_dir
            / "configs"
        )
        run_dir = (
            case_dir
            / "runs"
        )

        input_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        config_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        geometry = get_model_geometry(
            truth.model_name
        )

        observation_config_paths = {}

        for name, observation_truth in truth.observations.items():
            metadata = dict(
                observation_truth.metadata
            )

            rotation_deg = float(
                metadata.get(
                    "rotation_deg",
                    0.0,
                )
            )

            crpix = metadata.get(
                "crpix"
            )

            if crpix is None:
                ny, nx = observation_truth.image_shape
                crpix = (
                    0.5 * (nx + 1),
                    0.5 * (ny + 1),
                )

            wcs = self._make_wcs(
                ra=truth.parameter(
                    "ra"
                ),
                dec=truth.parameter(
                    "dec"
                ),
                pixel_scale_arcsec=(
                    observation_truth.pixel_scale
                ),
                rotation_deg=rotation_deg,
                crpix=(
                    float(crpix[0]),
                    float(crpix[1]),
                ),
            )

            transform = SkyModelParameterTransform(
                wcs=wcs,
                geometry=geometry,
            )

            local_parameters = self._observation_parameters(
                truth,
                name,
            )

            detector_parameters = transform(
                local_parameters
            )

            psf_fwhm = float(
                observation_truth.psf_metadata[
                    "fwhm_pixels"
                ]
            )

            psf = GaussianPSF(
                fwhm=psf_fwhm
            )

            model = make_single_sersic_psf_sky_model(
                psf=psf,
                **detector_parameters,
            )

            noiseless = np.asarray(
                model.render(
                    shape=observation_truth.image_shape
                ),
                dtype=float,
            )

            noise_sigma = float(
                observation_truth.noise_metadata.get(
                    "sigma",
                    0.0,
                )
            )

            seed = observation_truth.metadata.get(
                "seed"
            )

            if seed is None:
                seed = truth.random_seed

            rng = np.random.default_rng(
                seed
            )

            raw_background = float(
                observation_truth.metadata.get(
                    "raw_background",
                    0.0,
                )
            )

            image = (
                noiseless
                + raw_background
                + rng.normal(
                    loc=0.0,
                    scale=noise_sigma,
                    size=noiseless.shape,
                )
            )

            image_path = (
                input_dir
                / f"{name}.fits"
            )

            header = wcs.to_header()
            header["BUNIT"] = "adu"

            fits.writeto(
                image_path,
                image,
                header=header,
                overwrite=True,
            )

            target_x, target_y = (
                wcs.world_to_pixel_values(
                    truth.parameter(
                        "ra"
                    ),
                    truth.parameter(
                        "dec"
                    ),
                )
            )

            config = {
                "object": {
                    "name": (
                        f"{case_id}_{name}"
                    ),
                },
                "data": {
                    "image_path": str(
                        image_path
                    ),
                    "target_position": {
                        "x": float(
                            target_x
                        ),
                        "y": float(
                            target_y
                        ),
                    },
                    "cutout": {
                        "mode": "fixed",
                        "size": int(
                            min(
                                observation_truth.image_shape
                            )
                        ),
                    },
                },
                "outputs": {
                    "root": str(
                        run_dir
                        / name
                    ),
                    "overwrite": True,
                },
                "preparation": {
                    "background": {
                        "box_size": 20,
                        "filter_size": 3,
                    },
                    "noise": {},
                    "detection": {
                        "n_sigma": 3.0,
                        "n_pixels": 20,
                        "n_levels": 32,
                    },
                },
                "psf": {
                    "type": "gaussian",
                    "gaussian": {
                        "fwhm": psf_fwhm,
                    },
                },
                "fit": {
                    "models": [
                        truth.model_name
                    ],
                },
            }

            config_path = (
                config_dir
                / f"{name}.yaml"
            )

            with config_path.open(
                "w"
            ) as stream:
                yaml.safe_dump(
                    config,
                    stream,
                    sort_keys=False,
                )

            observation_config_paths[
                name
            ] = config_path

        if sampler is None:
            sampler_config = {
                "name": "dynesty",
                "mode": "static",
                "init": {
                    "nlive": 150,
                    "sample": "auto",
                },
                "run": {
                    "dlogz": 0.3,
                },
            }
        else:
            sampler_config = dict(
                sampler
            )

        joint_config = {
            "observations": {
                name: {
                    "config_path": str(
                        path.relative_to(
                            case_dir
                        )
                    ),
                }
                for name, path
                in observation_config_paths.items()
            },
            "fit": {
                "models": [
                    truth.model_name
                ],
                "sampler": sampler_config,
                "multi_observation": {
                    "parameter_frame": "sky",
                    "parameters": self._parameter_config(
                        truth
                    ),
                },
            },
        }

        joint_path = (
            case_dir
            / "joint.yaml"
        )

        with joint_path.open(
            "w"
        ) as stream:
            yaml.safe_dump(
                joint_config,
                stream,
                sort_keys=False,
            )

        return joint_path

    @staticmethod
    def _observation_parameters(
        truth: MultiObservationGroundTruth,
        observation: str,
    ) -> dict[str, float]:
        """Return local sky-frame parameters for one observation."""
        parameters = {}

        for name, value in truth.parameters.items():
            suffix = (
                f"__{observation}"
            )

            if name.endswith(
                suffix
            ):
                parameters[
                    name[: -len(suffix)]
                ] = float(
                    value
                )
            elif "__" not in name:
                parameters[
                    name
                ] = float(
                    value
                )

        return parameters

    @staticmethod
    def _parameter_config(
        truth: MultiObservationGroundTruth,
    ) -> dict[str, Any]:
        """Build the initial joint recovery prior contract."""
        observations = tuple(
            truth.observations
        )

        def uniform(
            lower,
            upper,
        ):
            return {
                "type": "uniform",
                "lower": float(lower),
                "upper": float(upper),
            }

        parameters = {
            "flux": {
                "coupling": "independent",
                "priors": {
                    name: uniform(
                        0.5
                        * truth.parameter(
                            f"flux__{name}"
                        ),
                        1.5
                        * truth.parameter(
                            f"flux__{name}"
                        ),
                    )
                    for name in observations
                },
            },
            "re_arcsec": {
                "coupling": "shared",
                "prior": uniform(
                    0.5
                    * truth.parameter(
                        "re_arcsec"
                    ),
                    1.5
                    * truth.parameter(
                        "re_arcsec"
                    ),
                ),
            },
            "n": {
                "coupling": "shared",
                "prior": uniform(
                    0.8,
                    4.0,
                ),
            },
            "q": {
                "coupling": "shared",
                "prior": uniform(
                    0.3,
                    1.0,
                ),
            },
            "position_angle": {
                "coupling": "shared",
                "prior": uniform(
                    -1.5,
                    1.5,
                ),
            },
            "ra": {
                "coupling": "shared",
                "prior": uniform(
                    truth.parameter(
                        "ra"
                    )
                    - 1.0 / 3600.0,
                    truth.parameter(
                        "ra"
                    )
                    + 1.0 / 3600.0,
                ),
            },
            "dec": {
                "coupling": "shared",
                "prior": uniform(
                    truth.parameter(
                        "dec"
                    )
                    - 1.0 / 3600.0,
                    truth.parameter(
                        "dec"
                    )
                    + 1.0 / 3600.0,
                ),
            },
            "sky": {
                "coupling": "independent",
                "priors": {
                    name: uniform(
                        truth.parameter(
                            f"sky__{name}"
                        )
                        - 20.0,
                        truth.parameter(
                            f"sky__{name}"
                        )
                        + 20.0,
                    )
                    for name in observations
                },
            },
        }

        return parameters

    @staticmethod
    def _make_wcs(
        *,
        ra: float,
        dec: float,
        pixel_scale_arcsec: float,
        rotation_deg: float,
        crpix: tuple[float, float],
    ) -> WCS:
        """Build one synthetic celestial WCS."""
        angle = np.deg2rad(
            rotation_deg
        )
        scale = (
            pixel_scale_arcsec
            / 3600.0
        )

        rotation = np.array(
            [
                [
                    np.cos(angle),
                    -np.sin(angle),
                ],
                [
                    np.sin(angle),
                    np.cos(angle),
                ],
            ]
        )

        reflection = np.array(
            [
                [-1.0, 0.0],
                [0.0, 1.0],
            ]
        )

        wcs = WCS(
            naxis=2
        )
        wcs.wcs.crpix = list(
            crpix
        )
        wcs.wcs.crval = [
            ra,
            dec,
        ]
        wcs.wcs.ctype = [
            "RA---TAN",
            "DEC--TAN",
        ]
        wcs.wcs.cunit = [
            "deg",
            "deg",
        ]
        wcs.wcs.cd = (
            scale
            * rotation
            @ reflection
        )

        return wcs

    @staticmethod
    def _validate_case_id(
        case_id: str,
    ) -> str:
        """Validate one case identifier."""
        if (
            not isinstance(
                case_id,
                str,
            )
            or not case_id.strip()
        ):
            raise ValueError(
                "Case ID must be a non-empty string."
            )

        normalized = case_id.strip()

        if Path(normalized).name != normalized:
            raise ValueError(
                "Case ID must not contain path components."
            )

        return normalized
