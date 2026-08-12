"""Command-line interface for BayMorph."""

from __future__ import annotations

import argparse
import shutil
from importlib import metadata, resources
from pathlib import Path

from baymorph.pipeline.config_maker import make_config_from_fits


_RESOURCE_PACKAGE = "baymorph.resources"


def _package_version() -> str:
    """Return the installed BayMorph version."""
    try:
        return metadata.version("baymorph")
    except metadata.PackageNotFoundError:
        return "unknown"


def _copy_resource_tree(
    resource_name: str,
    destination: Path,
    *,
    force: bool = False,
) -> None:
    """Copy a packaged resource directory to a filesystem destination."""
    source = resources.files(_RESOURCE_PACKAGE).joinpath(resource_name)

    if not source.is_dir():
        raise RuntimeError(
            f"Packaged BayMorph resource directory not found: "
            f"{resource_name!r}."
        )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    for resource in source.iterdir():
        output_path = destination / resource.name

        if output_path.exists() and not force:
            raise FileExistsError(
                f"Destination already exists: {output_path}. "
                "Use --force to overwrite it."
            )

        if resource.is_dir():
            with resources.as_file(resource) as source_path:
                shutil.copytree(
                    source_path,
                    output_path,
                    dirs_exist_ok=force,
                )
        else:
            with resources.as_file(resource) as source_path:
                shutil.copy2(
                    source_path,
                    output_path,
                )


def _command_examples(args: argparse.Namespace) -> int:
    """Copy BayMorph example notebooks and configurations."""
    destination = Path(args.destination).expanduser().resolve()

    _copy_resource_tree(
        "configs",
        destination / "configs",
        force=args.force,
    )
    _copy_resource_tree(
        "notebooks",
        destination / "notebooks",
        force=args.force,
    )

    print(f"BayMorph examples copied to:\n{destination}")
    return 0


def _command_init(args: argparse.Namespace) -> int:
    """Create a new BayMorph project directory."""
    project_dir = Path(args.project_name).expanduser().resolve()

    if project_dir.exists() and any(project_dir.iterdir()) and not args.force:
        raise FileExistsError(
            f"Project directory is not empty: {project_dir}. "
            "Use --force to continue."
        )

    project_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for directory_name in (
        "configs",
        "notebooks",
        "data",
        "runs",
    ):
        (project_dir / directory_name).mkdir(
            parents=True,
            exist_ok=True,
        )

    _copy_resource_tree(
        "configs",
        project_dir / "configs",
        force=args.force,
    )
    _copy_resource_tree(
        "notebooks",
        project_dir / "notebooks",
        force=args.force,
    )

    print(f"BayMorph project created at:\n{project_dir}")
    return 0


def _target_position_from_args(
    args: argparse.Namespace,
):
    """Return the explicit target seed requested on the command line."""
    if args.target_center:
        if (
            args.target_x is not None
            or args.target_y is not None
        ):
            raise ValueError(
                "--target-center cannot be combined with "
                "--target-x or --target-y."
            )

        return "center"

    if (
        args.target_x is None
        and args.target_y is None
    ):
        raise ValueError(
            "Specify either --target-center or both "
            "--target-x and --target-y."
        )

    if (
        args.target_x is None
        or args.target_y is None
    ):
        raise ValueError(
            "--target-x and --target-y must be supplied together."
        )

    return (
        float(args.target_x),
        float(args.target_y),
    )


def _command_config(
    args: argparse.Namespace,
) -> int:
    """Generate a BayMorph YAML configuration from one FITS image."""
    fits_path = Path(
        args.fits_path
    ).expanduser()

    if args.output is None:
        output_path = Path(
            f"{fits_path.stem}.yaml"
        )
    else:
        output_path = Path(
            args.output
        ).expanduser()

    target_position = _target_position_from_args(
        args
    )

    make_config_from_fits(
        fits_path,
        target_position=target_position,
        output_path=output_path,
        object_name=args.object_name,
        outputs_root=args.outputs_root,
        overwrite=not args.no_overwrite,
        cutout_mode=args.cutout_mode,
        fixed_size=args.cutout_size,
        psf_type=args.psf,
        gaussian_fwhm_pixels=args.gaussian_fwhm_pixels,
    )

    print(
        "BayMorph configuration written to:\n"
        f"{output_path.resolve()}"
    )

    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the BayMorph argument parser."""
    parser = argparse.ArgumentParser(
        prog="baymorph",
        description=(
            "Bayesian morphological modelling of astronomical images."
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_package_version()}",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    examples_parser = subparsers.add_parser(
        "examples",
        help="Copy packaged example notebooks and YAML configurations.",
    )
    examples_parser.add_argument(
        "destination",
        nargs="?",
        default="baymorph_examples",
        help=(
            "Destination directory. "
            "Default: ./baymorph_examples"
        ),
    )
    examples_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing example files.",
    )
    examples_parser.set_defaults(
        handler=_command_examples,
    )

    init_parser = subparsers.add_parser(
        "init",
        help="Create a new BayMorph project directory.",
    )
    init_parser.add_argument(
        "project_name",
        help="Project directory to create.",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into an existing non-empty directory.",
    )
    init_parser.set_defaults(
        handler=_command_init,
    )

    config_parser = subparsers.add_parser(
        "config",
        help="Generate a BayMorph YAML configuration from a FITS image.",
    )

    config_parser.add_argument(
        "fits_path",
        help="Input astronomical FITS image.",
    )

    config_parser.add_argument(
        "--output",
        help=(
            "Output YAML path. "
            "Default: <FITS stem>.yaml in the current directory."
        ),
    )

    config_parser.add_argument(
        "--object-name",
        help="Analysis object name. Default: FITS filename stem.",
    )

    config_parser.add_argument(
        "--target-center",
        action="store_true",
        help=(
            "Explicitly use the detector centre as the target-identification "
            "seed."
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
        "--cutout-mode",
        choices=(
            "auto",
            "fixed",
        ),
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
        choices=(
            "empirical",
            "gaussian",
        ),
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
        "--outputs-root",
        default="runs",
        help="BayMorph output root. Default: runs.",
    )

    config_parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Disable overwriting of existing BayMorph run products.",
    )

    config_parser.set_defaults(
        handler=_command_config,
    )

    return parser


def main() -> int:
    """Run the BayMorph command-line interface."""
    parser = _build_parser()
    args = parser.parse_args()

    try:
        return int(args.handler(args))
    except (
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        parser.error(str(error))

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
