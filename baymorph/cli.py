"""Command-line interface for BayMorph."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from collections.abc import Sequence
from importlib import metadata, resources
from pathlib import Path
from typing import Any

from baymorph.core.correlated_noise import SKY_CORRELATION_ESTIMATORS
from baymorph.core.header_metadata import PHOTOMETRIC_SYSTEMS
from baymorph.observation.background import BACKGROUND_PARAMETER_NAMES
from baymorph.pipeline.advisor import ObservationAdvisor
from baymorph.pipeline.config_maker import make_config_from_fits
from baymorph.pipeline.opening import (
    WHOLE_IMAGE_MAX_SIDE,
    SkyPosition,
    describe_working_region,
    read_observation_metadata,
    working_region,
)
from baymorph.runner import BayMorphRunner

_RESOURCE_PACKAGE = "baymorph.resources"


class UsageError(ValueError):
    """An invalid combination of command-line options.

    It is reported with the usage of the command and the exit status 2,
    like the errors that argparse finds itself.
    """


def _package_version() -> str:
    """Return the installed BayMorph version."""
    try:
        return metadata.version("baymorph")
    except metadata.PackageNotFoundError:
        return "unknown"


def _copy_resource_tree(
    resource_name: str, destination: Path, *, force: bool = False
) -> None:
    """Copy a packaged resource directory to a filesystem destination."""
    source = resources.files(_RESOURCE_PACKAGE).joinpath(resource_name)

    if not source.is_dir():
        raise RuntimeError(
            f"Packaged BayMorph resource directory not found: " f"{resource_name!r}."
        )

    destination.mkdir(parents=True, exist_ok=True)

    for resource in source.iterdir():
        output_path = destination / resource.name

        if output_path.exists() and not force:
            raise FileExistsError(
                f"Destination already exists: {output_path}. "
                "Use --force to overwrite it."
            )

        if resource.is_dir():
            with resources.as_file(resource) as source_path:
                shutil.copytree(source_path, output_path, dirs_exist_ok=force)
        else:
            with resources.as_file(resource) as source_path:
                shutil.copy2(source_path, output_path)


def _command_examples(args: argparse.Namespace) -> int:
    """Copy packaged BayMorph example configurations."""
    destination = Path(args.destination).expanduser().resolve()

    _copy_resource_tree("configs", destination / "configs", force=args.force)

    print("BayMorph example configurations copied to:" f"\n{destination}")
    return 0


def _command_init(args: argparse.Namespace) -> int:
    """Create a new BayMorph project directory."""
    project_dir = Path(args.project_name).expanduser().resolve()

    if project_dir.exists() and any(project_dir.iterdir()) and not args.force:
        raise FileExistsError(
            f"Project directory is not empty: {project_dir}. "
            "Use --force to continue."
        )

    project_dir.mkdir(parents=True, exist_ok=True)

    for directory_name in ("configs", "notebooks", "data", "runs"):
        (project_dir / directory_name).mkdir(parents=True, exist_ok=True)

    _copy_resource_tree("configs", project_dir / "configs", force=args.force)

    if resources.files(_RESOURCE_PACKAGE).joinpath("notebooks").is_dir():
        _copy_resource_tree("notebooks", project_dir / "notebooks", force=args.force)

    print(f"BayMorph project created at:\n{project_dir}")
    return 0


def _target_position_from_args(args: argparse.Namespace):
    """Return the explicit target seed requested on the command line."""
    in_pixels = args.target_x is not None or args.target_y is not None
    on_sky = args.target_ra is not None or args.target_dec is not None

    if args.target_center:
        if in_pixels or on_sky:
            raise UsageError(
                "--target-center cannot be combined with --target-x, "
                "--target-y, --target-ra or --target-dec."
            )

        return "center"

    if in_pixels and on_sky:
        raise UsageError(
            "Give the target either in pixels, with --target-x and --target-y, "
            "or on the sky, with --target-ra and --target-dec."
        )

    if on_sky:
        if args.target_ra is None or args.target_dec is None:
            raise UsageError("--target-ra and --target-dec must be supplied together.")

        try:
            return SkyPosition.of(
                {"ra": _sky_value(args.target_ra), "dec": _sky_value(args.target_dec)}
            )
        except ValueError as error:
            raise UsageError(str(error)) from error

    if args.target_x is None and args.target_y is None:
        raise UsageError(
            "Specify either --target-center, both --target-x and --target-y, "
            "or both --target-ra and --target-dec."
        )

    if args.target_x is None or args.target_y is None:
        raise UsageError("--target-x and --target-y must be supplied together.")

    return (float(args.target_x), float(args.target_y))


def _sky_value(text: str) -> float | str:
    """Return a coordinate of the command line: degrees, or a text with units."""
    try:
        return float(text)
    except ValueError:
        return text


def _models_from_args(args: argparse.Namespace) -> list[str]:
    """Return the models requested on the command line, in order."""
    models = [str(name).strip() for name in (args.model or ["single_sersic"])]

    if any(not name for name in models):
        raise UsageError("Model names must be non-empty.")

    duplicates = sorted({name for name in models if models.count(name) > 1})

    if duplicates:
        raise UsageError(f"--model names {duplicates} more than once.")

    return models


def _model_priors_from_args(
    args: argparse.Namespace, models: list[str]
) -> dict[str, dict[str, dict[str, float | str]]] | None:
    """Build explicit uniform model priors supplied on the command line.

    A prior names its parameter, or ``MODEL.PARAMETER`` to name the model it
    belongs to. The plain name is allowed only when one model is requested.

    Returns
    -------
    dict or None
        The priors of each requested model that declares any, or None.
    """
    entries = args.uniform_prior

    if not entries:
        return None

    priors: dict[str, dict[str, dict[str, float | str]]] = {}

    for qualified_name, lower, upper in entries:
        text = str(qualified_name).strip()
        model_name, separator, parameter_name = text.rpartition(".")

        if not separator:
            if len(models) > 1:
                raise UsageError(
                    f"With several --model options, name the model of each "
                    f"prior: --uniform-prior MODEL.{text} LOWER UPPER."
                )

            model_name = models[0]
        elif model_name not in models:
            raise UsageError(
                f"The prior {text!r} names the model {model_name!r}, which "
                f"is not among the requested models {models}."
            )

        name = parameter_name.strip()

        if not name:
            raise UsageError("Prior parameter names must be non-empty.")

        model_priors = priors.setdefault(model_name, {})

        if name in model_priors:
            raise UsageError(
                f"Duplicate prior supplied for parameter {name!r} of model "
                f"{model_name!r}."
            )

        model_priors[name] = {
            "type": "uniform",
            "lower": float(lower),
            "upper": float(upper),
        }

    return priors


def _hdu_argument(value: str) -> int | str:
    """Return an HDU index, or an ``EXTNAME`` given as text."""
    text = str(value).strip()

    if not text:
        raise argparse.ArgumentTypeError("--hdu must be an HDU index or an EXTNAME.")

    return int(text) if text.isdigit() else text


def _finite_number(value: str) -> float:
    """Return a finite number given on the command line."""
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number.") from None

    if not math.isfinite(number):
        raise argparse.ArgumentTypeError(f"{value!r} is not finite.")

    return number


def _positive_number(value: str) -> float:
    """Return a positive finite number given on the command line."""
    number = _finite_number(value)

    if number <= 0.0:
        raise argparse.ArgumentTypeError(f"{value!r} is not positive.")

    return number


def _photometric_system(value: str) -> str:
    """Return a photometric system, whatever its case."""
    systems = {system.lower(): system for system in PHOTOMETRIC_SYSTEMS}
    system = systems.get(str(value).strip().lower())

    if system is None:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not one of " + ", ".join(PHOTOMETRIC_SYSTEMS) + "."
        )

    return system


def _image_unit(value: str) -> str:
    """Return a non-empty image unit."""
    unit = str(value).strip()

    if not unit:
        raise argparse.ArgumentTypeError("The image unit must be non-empty.")

    return unit


_CALIBRATION_OPTIONS = (
    ("image_unit", "--image-unit"),
    ("zeropoint", "--zeropoint"),
    ("photometric_system", "--photometric-system"),
    ("exposure_time", "--exposure-time"),
    ("gain", "--gain"),
    ("saturation", "--saturation"),
)
"""Quantities of ``data.calibration`` and the options that declare them."""


def _calibration_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    """Return the calibration declared on the command line, or None."""
    declared = {
        name: getattr(args, name)
        for name, _ in _CALIBRATION_OPTIONS
        if getattr(args, name) is not None
    }

    if "photometric_system" in declared and "zeropoint" not in declared:
        raise UsageError(
            "--photometric-system qualifies a declared zeropoint; declare "
            "--zeropoint with it."
        )

    return declared or None


def _preparation_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    """Return the preparation section requested on the command line."""
    if args.sky_correlation is None:
        return None

    return {"noise": {"sky_correlation": args.sky_correlation}}


def _region_from_args(args: argparse.Namespace) -> str | dict[str, float] | None:
    """Return the data.region that the options declare, if any."""
    if args.whole_image:
        return "whole"

    if args.region_size is not None:
        return {"size": args.region_size}

    return None


def _region_line(data: dict[str, Any]) -> str:
    """Say how much of the image BayMorph will read."""
    metadata = read_observation_metadata(data["image_path"], hdu=data.get("hdu"))

    return describe_working_region(working_region(data), metadata)


def _command_config(args: argparse.Namespace) -> int:
    """Generate a BayMorph YAML configuration from one FITS image."""
    fits_path = Path(args.fits_path).expanduser()

    if args.output is None:
        output_path = Path(f"{fits_path.stem}.yaml")
    else:
        output_path = Path(args.output).expanduser()

    target_position = _target_position_from_args(args)
    models = _models_from_args(args)
    model_priors = _model_priors_from_args(args, models)
    preparation = _preparation_from_args(args)
    calibration = _calibration_from_args(args)
    region = _region_from_args(args)

    if args.advise:
        return _advise_config(
            args,
            fits_path=fits_path,
            output_path=output_path,
            target_position=target_position,
            models=models,
            model_priors=model_priors,
            preparation=preparation,
            calibration=calibration,
            region=region,
        )

    for option, value in (
        ("--accept", args.accept),
        ("--offline", args.offline),
        ("--background-model", args.background_model),
        ("--resampling-variance-factor", args.resampling_variance_factor),
    ):
        if value is not None and value is not False:
            raise UsageError(f"{option} requires --advise.")

    if len(models) > 1:
        raise UsageError(
            "Several --model options require --advise: without it the "
            "configuration holds one model."
        )

    config = make_config_from_fits(
        fits_path,
        target_position=target_position,
        hdu=args.hdu,
        calibration=calibration,
        region=region,
        output_path=output_path,
        object_name=args.object_name,
        model_name=models[0],
        model_priors=(None if model_priors is None else model_priors[models[0]]),
        sampler=args.sampler,
        outputs_root=args.outputs_root,
        overwrite=args.overwrite,
        cutout_mode=args.cutout_mode,
        fixed_size=args.cutout_size,
        psf_type=args.psf,
        gaussian_fwhm_pixels=args.gaussian_fwhm_pixels,
        psf_max_peak=args.psf_max_peak,
        preparation=preparation,
    )

    print("BayMorph configuration written to:\n" f"{output_path.resolve()}")

    if isinstance(target_position, SkyPosition):
        target = config["data"]["target_position"]
        print(
            f"Target: RA {target_position.ra:.10g}, Dec {target_position.dec:.10g} "
            f"(ICRS), at pixel ({target['x']:.2f}, {target['y']:.2f}) of the "
            "whole image."
        )

    print(_region_line(config["data"]))

    if "effective_gain" not in config["fit"].get("likelihood", {}):
        print(
            "The sky of the image is below ten times its scale, so it cannot "
            "give the effective gain of AD-047. Declare "
            "fit.likelihood.effective_gain (declared, sky_level or absent) "
            "before running the fit."
        )

    return 0


def _advise_config(
    args: argparse.Namespace,
    *,
    fits_path: Path,
    output_path: Path,
    target_position,
    models: list[str],
    model_priors: dict[str, dict[str, dict[str, float | str]]] | None,
    preparation: dict[str, Any] | None,
    calibration: dict[str, Any] | None,
    region: str | dict[str, float] | None,
) -> int:
    """Propose a configuration with the observation advisor.

    The advisor reads the header with the declared calibration, asks a
    calibration service for a quantity the header lacks unless
    ``--offline``, prepares the image and builds its PSF in the run
    directory, where the fit reuses them, and proposes every prior. The
    configuration and its report are written only with ``--accept``, and
    not when a request to a calibration service that the proposal needed
    failed: no answer, or an answer that fails its checks.
    """
    advisor = ObservationAdvisor(
        fits_path.resolve(),
        target_position=target_position,
        hdu=args.hdu,
        calibration=calibration,
        online=not args.offline,
        region=region,
    )

    print(advisor.header_report())
    print()
    print(
        _region_line(
            {
                "image_path": fits_path.resolve(),
                "hdu": args.hdu,
                "region": region,
                "target_position": target_position,
            }
        )
    )
    print("Preparing the image and building its PSF, which the fit will reuse.")

    # Absolute paths, as the configuration maker writes them, keep the
    # products of the advisor and of the fit in one run directory.
    proposal = advisor.propose_config(
        output_path=output_path,
        models=models,
        model_priors=model_priors,
        background_model=args.background_model,
        resampling_variance_factor=args.resampling_variance_factor,
        object_name=args.object_name,
        sampler=args.sampler,
        outputs_root=Path(args.outputs_root).expanduser().resolve(),
        overwrite=args.overwrite,
        cutout_mode=args.cutout_mode,
        fixed_size=args.cutout_size,
        psf_type=args.psf,
        gaussian_fwhm_pixels=args.gaussian_fwhm_pixels,
        psf_max_peak=args.psf_max_peak,
        preparation=preparation,
    )

    print()
    print(proposal.summary())
    print()

    if not args.accept:
        print(
            "Review the proposal, then run the same command with --accept "
            "to write the configuration and its report."
        )

        return 0

    failed = [
        record["title"]
        for record in (proposal.provenance or {}).get("bridges", [])
        if record["status"] == "failed"
    ]

    if failed:
        raise RuntimeError(
            "The configuration and its report were not written: the "
            + " and the ".join(failed)
            + " did not answer, or gave an answer that fails its checks, so the "
            "proposal holds the priors that do "
            "without it. Run the command again when the service answers, or add "
            "--offline to write this proposal."
        )

    written_path = proposal.write(accept=True)

    print(
        "BayMorph configuration written to:\n"
        f"{written_path}\n"
        "Advisor report written to:\n"
        f"{proposal.report_path.expanduser().resolve()}"
    )

    return 0


def _command_run(args: argparse.Namespace) -> int:
    """Run a BayMorph analysis from a YAML configuration."""
    config_path = Path(args.config_path).expanduser()

    runner = BayMorphRunner(config_path)

    runner.run(diagnostics=not args.no_diagnostics)

    print("BayMorph analysis completed.")

    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the BayMorph argument parser."""
    parser = argparse.ArgumentParser(
        prog="baymorph",
        description=("Bayesian morphological modelling of astronomical images."),
    )

    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {_package_version()}"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    examples_parser = subparsers.add_parser(
        "examples", help="Copy packaged example YAML configurations."
    )
    examples_parser.add_argument(
        "destination",
        nargs="?",
        default="baymorph_examples",
        help=("Destination directory. " "Default: ./baymorph_examples"),
    )
    examples_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing example files."
    )
    examples_parser.set_defaults(
        handler=_command_examples, command_parser=examples_parser
    )

    init_parser = subparsers.add_parser(
        "init", help="Create a new BayMorph project directory."
    )
    init_parser.add_argument("project_name", help="Project directory to create.")
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into an existing non-empty directory.",
    )
    init_parser.set_defaults(handler=_command_init, command_parser=init_parser)

    config_parser = subparsers.add_parser(
        "config", help="Generate a BayMorph YAML configuration from a FITS image."
    )

    config_parser.add_argument("fits_path", help="Input astronomical FITS image.")

    config_parser.add_argument(
        "--output",
        help=(
            "Output YAML path. " "Default: <FITS stem>.yaml in the current directory."
        ),
    )

    config_parser.add_argument(
        "--object-name", help="Analysis object name. Default: FITS filename stem."
    )

    config_parser.add_argument(
        "--hdu",
        type=_hdu_argument,
        help=(
            "Index or EXTNAME of the HDU that holds the image. "
            "Default: selected by BayMorph."
        ),
    )

    region_options = config_parser.add_mutually_exclusive_group()

    region_options.add_argument(
        "--region-size",
        dest="region_size",
        type=_positive_number,
        help=(
            "Side, in arcsec, of the square about the target within which "
            "the image is read, written to data.region. Default: the whole "
            f"image up to {WHOLE_IMAGE_MAX_SIDE} pixels on a side, otherwise "
            f"a square of {WHOLE_IMAGE_MAX_SIDE} pixels about the target."
        ),
    )

    region_options.add_argument(
        "--whole-image",
        dest="whole_image",
        action="store_true",
        help="Read the whole image, whatever its size (data.region: whole).",
    )

    config_parser.add_argument(
        "--model",
        action="append",
        help=(
            "Registered morphology model. Repeat it with --advise to "
            "propose several models, each fitted separately. "
            "Default: single_sersic."
        ),
    )

    config_parser.add_argument(
        "--sampler",
        default="nautilus",
        help=("Inference sampler name. " "Default: nautilus."),
    )

    config_parser.add_argument(
        "--uniform-prior",
        action="append",
        nargs=3,
        metavar=("PARAMETER", "LOWER", "UPPER"),
        default=[],
        help=(
            "Explicit uniform scientific prior. "
            "Repeat for multiple parameters, for example "
            "'--uniform-prior flux 100 1000000'. With several --model "
            "options, name the model too: "
            "'--uniform-prior double_sersic.q_bulge 0.4 1'."
        ),
    )

    config_parser.add_argument(
        "--advise",
        action="store_true",
        help=(
            "Print what the header holds, prepare the image and build its "
            "PSF in the run directory, where the fit reuses them, and "
            "propose every prior from the header, physics and these fixed "
            "inputs. The configuration is written only with --accept."
        ),
    )

    config_parser.add_argument(
        "--accept",
        action="store_true",
        help="With --advise, write the reviewed proposal and its report beside it.",
    )

    config_parser.add_argument(
        "--offline",
        action="store_true",
        help=(
            "With --advise, ask no calibration service online. Without it, "
            "the advisor asks the ACS Zeropoint Calculator of STScI for the "
            "zeropoint of a drizzled ACS image whose header gives no usable "
            "one, unless two readings of it disagree, and for which none is "
            "declared."
        ),
    )

    config_parser.add_argument(
        "--background-model",
        choices=tuple(BACKGROUND_PARAMETER_NAMES),
        help=(
            "With --advise, the residual background model whose priors are "
            "proposed. Default: constant."
        ),
    )

    config_parser.add_argument(
        "--resampling-variance-factor",
        type=float,
        help=(
            "With --advise, the factor by which a resampling of the image "
            "multiplied the variance of its noise, which lets the gain "
            "keyword of a resampled image be proposed."
        ),
    )

    config_parser.add_argument(
        "--target-center",
        action="store_true",
        help=(
            "Explicitly use the detector centre as the target-identification " "seed."
        ),
    )

    config_parser.add_argument(
        "--target-x",
        type=float,
        help="Target-identification seed x coordinate in zero-based pixels.",
    )

    config_parser.add_argument(
        "--target-y",
        type=float,
        help="Target-identification seed y coordinate in zero-based pixels.",
    )

    config_parser.add_argument(
        "--target-ra",
        help=(
            "Right ascension of the target (ICRS), in degrees or as a text "
            "with its unit such as 00h14m05.69s; converted to pixels through "
            "the celestial WCS of the header."
        ),
    )

    config_parser.add_argument(
        "--target-dec",
        help=(
            "Declination of the target (ICRS), in degrees or as a text with "
            "its unit such as -30d22m20.7s."
        ),
    )

    config_parser.add_argument(
        "--cutout-mode",
        choices=("auto", "fixed"),
        default="auto",
        help="Cutout strategy. Default: auto.",
    )

    config_parser.add_argument(
        "--cutout-size",
        type=int,
        help="Fixed cutout size in pixels when --cutout-mode=fixed.",
    )

    config_parser.add_argument(
        "--psf",
        choices=("empirical", "gaussian"),
        default="empirical",
        help="PSF strategy. Default: empirical.",
    )

    config_parser.add_argument(
        "--gaussian-fwhm-pixels",
        type=float,
        help=(
            "Gaussian PSF FWHM in pixels. Required only when it cannot "
            "be derived from FITS metadata."
        ),
    )

    config_parser.add_argument(
        "--psf-max-peak",
        type=float,
        help=(
            "Brightest peak pixel accepted for an empirical-PSF star, in "
            "image units; keep it below the saturation level. With --advise "
            "and without this option, half the saturation level of the "
            "header is proposed when its unit is established."
        ),
    )

    calibration_group = config_parser.add_argument_group(
        "calibration declared in data.calibration",
        "Quantities that the header does not give, or gives wrongly. A "
        "declared value takes the place of the header reading, which stays "
        "recorded beside it.",
    )

    calibration_group.add_argument(
        "--image-unit",
        dest="image_unit",
        type=_image_unit,
        help="Unit of the pixel values, such as ADU/s or electron/s.",
    )

    calibration_group.add_argument(
        "--zeropoint",
        type=_finite_number,
        help=(
            "Zeropoint in magnitudes, with m = zeropoint - 2.5 log10(flux in "
            "image units)."
        ),
    )

    calibration_group.add_argument(
        "--photometric-system",
        dest="photometric_system",
        type=_photometric_system,
        help="System of the declared zeropoint: AB or Vega.",
    )

    calibration_group.add_argument(
        "--exposure-time",
        dest="exposure_time",
        type=_positive_number,
        help="Exposure time in seconds.",
    )

    calibration_group.add_argument(
        "--gain",
        type=_positive_number,
        help=(
            "Gain of the detector in electrons per ADU, the meaning of the "
            "GAIN keyword; not the effective gain of the noise model."
        ),
    )

    calibration_group.add_argument(
        "--saturation",
        type=_positive_number,
        help="Saturation level in the units of the image.",
    )

    config_parser.add_argument(
        "--sky-correlation",
        choices=tuple(SKY_CORRELATION_ESTIMATORS),
        help=(
            "Estimator of the sky correlation of the noise model. "
            "Default: autocovariance."
        ),
    )

    config_parser.add_argument(
        "--outputs-root", default="runs", help="BayMorph output root. Default: runs."
    )

    config_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting of existing BayMorph run products.",
    )

    config_parser.set_defaults(handler=_command_config, command_parser=config_parser)

    run_parser = subparsers.add_parser(
        "run", help="Run a BayMorph analysis from a YAML configuration."
    )

    run_parser.add_argument(
        "config_path", help="BayMorph YAML configuration to execute."
    )

    run_parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help=(
            "Run inference without generating the configured " "post-fit diagnostics."
        ),
    )

    run_parser.set_defaults(handler=_command_run, command_parser=run_parser)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the BayMorph command-line interface.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments after the program name. Default: those of the process.

    Returns
    -------
    int
        Exit status: 0 on success, 1 when the command fails. An invalid
        combination of options exits with status 2 and the usage of the
        command, as argparse does.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        return int(args.handler(args))
    except UsageError as error:
        args.command_parser.error(str(error))
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"baymorph: error: {error}", file=sys.stderr)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
