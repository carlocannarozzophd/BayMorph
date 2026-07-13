"""Command-line interface for BayMorph."""

from __future__ import annotations

import argparse
import shutil
from importlib import metadata, resources
from pathlib import Path


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

    return parser


def main() -> int:
    """Run the BayMorph command-line interface."""
    parser = _build_parser()
    args = parser.parse_args()

    try:
        return int(args.handler(args))
    except (FileExistsError, RuntimeError) as error:
        parser.error(str(error))

    return 1


if __name__ == "__main__":
    raise SystemExit(main())