"""PSF-candidate selection utilities."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from baymorph.core.mask import no_data_mask
from baymorph.pipeline.checkpoint_signing import canonical_json_sha256
from baymorph.pipeline.empirical_psf_builder import window_holds
from baymorph.pipeline.config import load_resolved_config
from baymorph.pipeline.image_preparation import detect_real_fits_sources
from baymorph.pipeline.opening import open_configured_observation, working_region
from baymorph.pipeline.outputs import make_output_tree, pipeline_paths
from baymorph.pipeline.prepared_identity import file_content_identity
from baymorph.pipeline.psf_candidate_screen import (
    PSF_CANDIDATE_SCREEN_SCHEMA,
    PSF_CANDIDATE_SCREEN_SCHEMA_VERSION,
    PointSourceScreenChangeError,
    PointSourceScreenError,
    point_source_screen_marker,
    point_source_screen_rule,
    recorded_screen_rule,
    screen_change_message,
    screen_difference_note,
    screen_psf_candidates,
    screen_stop_message,
)

PSF_CANDIDATES_PROVENANCE_SCHEMA_VERSION = 1


@dataclass
class PSFCandidate:
    """Candidate point source for empirical PSF construction.

    Parameters
    ----------
    x, y : float
        Candidate centroid coordinates in image pixels.
    label : int, optional
        Segmentation label associated with the candidate.
    area : float, optional
        Segmented source area.
    max_value : float, optional
        Peak pixel value of the candidate source.
    flux : float, optional
        Segment flux measured by the source catalog.
    eccentricity : float, optional
        Source eccentricity.
    elongation : float, optional
        Source elongation.
    semimajor_sigma, semiminor_sigma : float, optional
        Source size estimates from the catalog.
    fwhm : float, optional
        Full width at half maximum estimate, if available.
    sharpness : float, optional
        Sharpness diagnostic, if available.
    roundness : float, optional
        Roundness diagnostic, if available.
    half_size : int, optional
        Half-size of the PSF cutout assigned to this candidate.
    """

    x: float
    y: float
    label: int | None = None
    area: float | None = None
    max_value: float | None = None
    flux: float | None = None
    eccentricity: float | None = None
    elongation: float | None = None
    semimajor_sigma: float | None = None
    semiminor_sigma: float | None = None
    fwhm: float | None = None
    sharpness: float | None = None
    roundness: float | None = None
    half_size: int | None = None


def save_psf_candidates(config: dict[str, Any], candidates: list[PSFCandidate]) -> Path:
    """Save PSF candidates to the configured pipeline output path.

    Parameters
    ----------
    config : dict
        Resolved pipeline configuration.
    candidates : list of PSFCandidate
        Candidate point sources to save.

    Returns
    -------
    pathlib.Path
        Path to the saved JSON file.
    """
    make_output_tree(config)
    path = pipeline_paths(config)["psf"]["candidates"]

    if path.exists() and not bool(config["outputs"].get("overwrite", False)):
        raise FileExistsError(
            "PSF candidate output already exists and overwrite is disabled: " f"{path}"
        )

    with path.open("w") as f:
        json.dump([asdict(candidate) for candidate in candidates], f, indent=2)

    return path


def build_psf_candidates_provenance(config: dict[str, Any]) -> dict[str, Any]:
    """Fingerprint the image and the settings that select the PSF candidates.

    Parameters
    ----------
    config : dict
        Resolved pipeline configuration.

    Returns
    -------
    dict
        The FITS file, with its HDU when one is declared and its working
        region when there is one, and the digests of the detection and
        candidate-selection settings.
    """
    image_path = Path(config["data"]["image_path"]).expanduser().resolve()
    input_fits = {
        "path": str(image_path),
        "sha256": file_content_identity(image_path)["sha256"],
    }

    if config["data"].get("hdu") is not None:
        input_fits["hdu"] = config["data"]["hdu"]

    # The stars are searched within the working region, so a region enters
    # the provenance; an image read whole keeps the provenance it had.
    region = working_region(config["data"])

    if region is not None:
        input_fits["region"] = region.as_dict()

    candidate_selection = config["psf"]["empirical"]["candidate_selection"]
    marker = point_source_screen_marker(point_source_screen_rule(candidate_selection))
    provenance = {
        "schema_version": PSF_CANDIDATES_PROVENANCE_SCHEMA_VERSION,
        "input_fits": input_fits,
        "detection_config_sha256": canonical_json_sha256(
            config["preparation"]["detection"]
        ),
        "candidate_selection_config_sha256": canonical_json_sha256(
            screened_selection_identity(candidate_selection)
        ),
    }

    # A screen enters with its rule and the digest of its constants, so
    # candidates selected by code that ignored the rule are refused; the
    # neutral rule leaves the provenance of earlier candidates unchanged.
    if marker is not None:
        provenance["point_source_screen"] = marker

    return provenance


def screened_selection_identity(candidate_selection: dict[str, Any]) -> dict[str, Any]:
    """Return the candidate-selection settings as they enter an identity.

    The neutral point-source screen ``none`` is left out, so that products
    selected before the screen existed keep their identity; any other rule
    enters by name.

    Parameters
    ----------
    candidate_selection : dict
        The section ``psf.empirical.candidate_selection``.

    Returns
    -------
    dict
        The settings that the digest of the selection covers.
    """
    rule = point_source_screen_rule(candidate_selection)
    settings = {
        key: value
        for key, value in candidate_selection.items()
        if key != "point_source_screen"
    }

    if rule != "none":
        settings["point_source_screen"] = rule

    return settings


def save_psf_candidates_provenance(config: dict[str, Any]) -> Path:
    """Record the image and settings from which the PSF candidates were selected.

    Parameters
    ----------
    config : dict
        Resolved pipeline configuration.

    Returns
    -------
    pathlib.Path
        Path to the saved provenance file.
    """
    path = pipeline_paths(config)["psf"]["candidates_provenance"]
    path.write_text(
        json.dumps(build_psf_candidates_provenance(config), indent=2) + "\n",
        encoding="utf-8",
    )

    return path


def psf_candidates_provenance_exists(config: dict[str, Any]) -> bool:
    """Return whether the saved PSF candidates record their provenance.

    Candidates saved by BayMorph 0.5.1 and earlier have none.

    Parameters
    ----------
    config : dict
        Resolved pipeline configuration.

    Returns
    -------
    bool
        True when the provenance file of the candidates is present.
    """
    return pipeline_paths(config)["psf"]["candidates_provenance"].is_file()


def validate_psf_candidates_provenance(config: dict[str, Any]) -> None:
    """Reject saved PSF candidates selected from other inputs or settings.

    Parameters
    ----------
    config : dict
        Resolved pipeline configuration.

    Raises
    ------
    PointSourceScreenChangeError
        If only the point-source screen differs; the message names the
        declaration that reuses the candidates.
    ValueError
        If the recorded provenance differs from that of the current image,
        HDU, detection settings or candidate-selection settings.
    """
    path = pipeline_paths(config)["psf"]["candidates_provenance"]
    recorded = json.loads(path.read_text(encoding="utf-8"))

    if recorded == build_psf_candidates_provenance(config):
        return

    check_screen_change(
        recorded,
        config,
        build_psf_candidates_provenance,
        products=f"PSF candidates ({path})",
    )
    rule = point_source_screen_rule(config["psf"]["empirical"]["candidate_selection"])

    raise ValueError(
        "The saved PSF candidates were selected from another image, HDU, "
        "working region, detection or candidate-selection setting than the "
        "current "
        f"configuration ({path}). Use a new output namespace, or set "
        "outputs.overwrite: true to select them again."
        + screen_difference_note(recorded_screen_rule(recorded), rule)
    )


def config_under_screen(config: Mapping[str, Any], rule: str) -> dict[str, Any]:
    """Return a copy of a configuration whose stars are screened by ``rule``.

    Parameters
    ----------
    config : mapping
        Resolved pipeline configuration.
    rule : str
        One of the point-source screens.

    Returns
    -------
    dict
        A deep copy whose ``point_source_screen`` is ``rule``.
    """
    copy = deepcopy(dict(config))
    psf = copy.setdefault("psf", {})
    empirical = psf.setdefault("empirical", {})
    selection = empirical.setdefault("candidate_selection", {})
    selection["point_source_screen"] = rule

    return copy


def check_screen_change(
    recorded: Mapping[str, Any],
    config: Mapping[str, Any],
    build_provenance: Callable[[Mapping[str, Any]], dict[str, Any]],
    *,
    products: str,
) -> None:
    """Refuse, with its own message, products that differ only in the screen.

    Products selected before the screen existed, or under another rule,
    are not reused under the rule of the configuration; when nothing else
    differs, the refusal says which declaration reuses them, rather than
    asking for an overwrite of accepted products.

    Parameters
    ----------
    recorded : mapping
        The recorded provenance, which differs from the current one.
    config : mapping
        Resolved pipeline configuration.
    build_provenance : callable
        The function that builds the provenance of these products from a
        configuration.
    products : str
        What was saved, for the message.

    Raises
    ------
    PointSourceScreenChangeError
        If the recorded provenance is the one the configuration would give
        under the screen it was made with.
    """
    selection = (config.get("psf") or {}).get("empirical", {})
    current = point_source_screen_rule(selection.get("candidate_selection") or {})
    made_with = recorded_screen_rule(recorded)

    if made_with is None or made_with == current:
        return

    if dict(recorded) == build_provenance(config_under_screen(config, made_with)):
        raise PointSourceScreenChangeError(
            screen_change_message(products, recorded=made_with, current=current)
        )


def save_psf_candidate_screen(config: dict[str, Any], record: dict[str, Any]) -> Path:
    """Save the record of the point-source screen of the PSF candidates.

    Parameters
    ----------
    config : dict
        Resolved pipeline configuration.
    record : dict
        The record returned by
        :func:`baymorph.pipeline.psf_candidate_screen.screen_psf_candidates`.

    Returns
    -------
    pathlib.Path
        Path to the saved JSON file.

    Raises
    ------
    FileExistsError
        If the record exists and ``outputs.overwrite`` is disabled.
    """
    make_output_tree(config)
    path = pipeline_paths(config)["psf"]["candidate_screen"]

    if path.exists() and not bool(config["outputs"].get("overwrite", False)):
        raise FileExistsError(
            "PSF candidate screen output already exists and overwrite is "
            f"disabled: {path}"
        )

    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    return path


def psf_candidate_screen_exists(config: dict[str, Any]) -> bool:
    """Return whether the record of the point-source screen exists.

    Parameters
    ----------
    config : dict
        Resolved pipeline configuration.

    Returns
    -------
    bool
        True when ``psf/psf_candidate_screen.json`` is present.
    """
    return pipeline_paths(config)["psf"]["candidate_screen"].is_file()


def load_psf_candidate_screen(config: dict[str, Any]) -> dict[str, Any]:
    """Load the record of the point-source screen of the saved candidates.

    Parameters
    ----------
    config : dict
        Resolved pipeline configuration whose point-source screen is not
        ``none``.

    Returns
    -------
    dict
        The record of the screen.

    Raises
    ------
    ValueError
        If the record is missing, or was written by another rule, another
        set of constants or another schema.
    """
    rule = point_source_screen_rule(config["psf"]["empirical"]["candidate_selection"])
    marker = point_source_screen_marker(rule)
    path = pipeline_paths(config)["psf"]["candidate_screen"]

    if marker is None:
        raise ValueError(
            "The PSF candidates are selected without a point-source screen "
            "(point_source_screen: none), so they have no screen record."
        )

    if not path.is_file():
        raise ValueError(
            "The saved PSF candidates were selected with the point-source "
            f"screen {rule}, but its record {path} is missing. Use a new "
            "output namespace, or set outputs.overwrite: true to select them "
            "again."
        )

    record = json.loads(path.read_text(encoding="utf-8"))

    if (
        not isinstance(record, dict)
        or record.get("schema") != PSF_CANDIDATE_SCREEN_SCHEMA
        or record.get("schema_version") != PSF_CANDIDATE_SCREEN_SCHEMA_VERSION
        or record.get("rule") != marker["rule"]
        or record.get("constants_sha256") != marker["constants_sha256"]
        or record.get("outcome") != "found"
    ):
        raise ValueError(
            f"The record of the point-source screen {path} was not written by "
            f"the screen {rule} of this version of BayMorph for the saved "
            "candidates. Use a new output namespace, or set outputs.overwrite: "
            "true to select them again."
        )

    return record


def validate_psf_candidate_screen(
    record: dict[str, Any], candidates: list[PSFCandidate]
) -> None:
    """Reject a screen record that does not list the saved candidates.

    The candidates that the screen kept, in its record, must be the saved
    candidates, in the same order and at the same positions, so that the
    record describes the stars that build the PSF.

    Parameters
    ----------
    record : dict
        Record returned by :func:`load_psf_candidate_screen`.
    candidates : list of PSFCandidate
        The saved candidates.

    Raises
    ------
    ValueError
        If the kept candidates of the record differ from the saved ones.
    """
    kept = [
        (entry.get("label"), entry.get("x"), entry.get("y"))
        for entry in record.get("candidates", [])
        if entry.get("kept")
    ]

    if kept != [(c.label, c.x, c.y) for c in candidates]:
        raise ValueError(
            "The record of the point-source screen does not list the saved PSF "
            "candidates as the stars it kept, so it does not describe them. Use "
            "a new output namespace, or set outputs.overwrite: true to select "
            "them again."
        )


def psf_candidates_exist(config: str | Path | dict[str, Any]) -> bool:
    """Return whether saved PSF candidates exist on disk.

    Parameters
    ----------
    config : dict, str, or pathlib.Path
        Resolved configuration dictionary or path to a YAML configuration file.

    Returns
    -------
    bool
        True when the candidate file is present.
    """
    if isinstance(config, (str, Path)):
        resolved = load_resolved_config(config)
    else:
        resolved = config

    return pipeline_paths(resolved)["psf"]["candidates"].is_file()


def load_psf_candidates(config: str | Path | dict[str, Any]) -> list[PSFCandidate]:
    """Load saved PSF candidates from disk.

    Parameters
    ----------
    config : dict, str, or pathlib.Path
        Resolved configuration dictionary or path to a YAML configuration file.

    Returns
    -------
    list of PSFCandidate
        Previously saved PSF candidates.
    """
    if isinstance(config, (str, Path)):
        resolved = load_resolved_config(config)
    else:
        resolved = config

    path = pipeline_paths(resolved)["psf"]["candidates"]

    with path.open("r") as f:
        raw_candidates = json.load(f)

    return [PSFCandidate(**candidate) for candidate in raw_candidates]


def run_psf_candidate_selection_from_config(
    config: str | Path | dict[str, Any],
) -> list[PSFCandidate]:
    """Select empirical PSF candidates from a pipeline configuration.

    Parameters
    ----------
    config : dict, str, or pathlib.Path
        Resolved configuration dictionary or path to a YAML configuration file.

    Returns
    -------
    list of PSFCandidate
        Selected PSF candidates.

    Raises
    ------
    RuntimeError
        If source segmentation or source-catalog construction fails.
    PointSourceScreenError
        If the point-source screen finds no stellar sequence among the
        candidates; nothing is saved, and the error carries the record.

    Notes
    -----
    Unless ``psf.empirical.candidate_selection.point_source_screen`` is
    ``none``, every candidate that passes the shape, isolation and no-data
    cuts is screened, and the first ``target_n_candidates`` members of the
    stellar sequence are kept; the record of the screen is saved in
    ``psf/psf_candidate_screen.json``.
    """
    if isinstance(config, (str, Path)):
        resolved = load_resolved_config(config)
    else:
        resolved = config

    make_output_tree(resolved)

    image = open_configured_observation(resolved["data"], declarations=False).image
    candidate_cfg = resolved["psf"]["empirical"]["candidate_selection"]
    rule = point_source_screen_rule(candidate_cfg)
    paths = pipeline_paths(resolved)["psf"]

    # Nothing is written unless everything can be: without overwrite, saved
    # candidates, or a record of a screen, are refused before the search,
    # so that a selection never leaves candidates beside a stale record.
    if not bool(resolved["outputs"].get("overwrite", False)):
        existing = [
            path
            for path in (paths["candidates"], paths["candidate_screen"])
            if path.exists()
        ]

        if existing:
            raise FileExistsError(
                "PSF candidate output already exists and overwrite is disabled: "
                + ", ".join(str(path) for path in existing)
            )
    region = image.region
    origin_x, origin_y = (0, 0) if region is None else (region.xmin, region.ymin)

    # Pixels without data take no part in the search (AD-053).
    no_data = no_data_mask(image.data, floating_point=image.floating_point)
    if not np.any(no_data):
        no_data = None

    detection_cfg = resolved["preparation"]["detection"]

    detection = detect_real_fits_sources(
        image.data,
        detection_sigma=detection_cfg["n_sigma"],
        detection_n_pixels=detection_cfg["n_pixels"],
        detection_n_levels=detection_cfg["n_levels"],
        no_data=no_data,
    )

    segmentation = detection.segmentation
    source_catalog = detection.catalog

    if segmentation is None:
        raise RuntimeError("No segmentation map was produced.")

    if source_catalog is None:
        raise RuntimeError("No source catalog was produced.")

    rows = source_catalog_to_candidate_rows(source_catalog)

    cuts = {
        "min_area": candidate_cfg["min_area"],
        "max_area": candidate_cfg["max_area"],
        "min_peak": candidate_cfg["min_peak"],
        "max_peak": candidate_cfg["max_peak"],
        "max_eccentricity": candidate_cfg["max_eccentricity"],
        "max_elongation": candidate_cfg["max_elongation"],
        "edge_margin": candidate_cfg["edge_margin"],
        "galaxy_exclusion_box": _box_in_region(
            candidate_cfg["galaxy_exclusion_box"], origin_x=origin_x, origin_y=origin_y
        ),
        "sigma_factor": candidate_cfg["sigma_factor"],
        "size_margin": candidate_cfg["size_margin"],
        "min_half_size": candidate_cfg["min_half_size"],
        "max_half_size": candidate_cfg["max_half_size"],
        "isolation_margin": candidate_cfg["isolation_margin"],
        "no_data": no_data,
    }
    screen_record = None

    if rule == "none":
        candidates = select_psf_candidates_from_rows(
            rows,
            segmentation.data,
            target_n_candidates=candidate_cfg["target_n_candidates"],
            **cuts,
        )
    else:
        # The whole pool, in the order of the selection, is screened before
        # the first candidates are kept, so a rejected source is replaced by
        # the next member of the stellar sequence.
        pool = select_psf_candidates_from_rows(
            rows, segmentation.data, target_n_candidates=max(1, len(rows)), **cuts
        )

        try:
            screen = screen_psf_candidates(
                image.data,
                pool,
                min_peak_snr=candidate_cfg["min_peak_snr"],
                target_n_candidates=int(candidate_cfg["target_n_candidates"]),
                origin=(origin_x, origin_y),
            )
        except PointSourceScreenError as error:
            # A stop saves nothing: the record travels with the error, whose
            # remedies name the region that was searched.
            searched = (
                "The whole image was searched."
                if region is None
                else "The stars were searched within the working region x "
                f"{region.xmin}-{region.xmax}, y {region.ymin}-{region.ymax} of "
                "the image."
            )

            raise PointSourceScreenError(
                screen_stop_message(
                    error.record, searched=searched, whole_image=region is None
                ),
                code=error.code,
                record=error.record,
                finding=error.finding,
            ) from error

        candidates = list(screen.kept)
        screen_record = screen.record

    # The stars are found in the array of the working region; their positions
    # are saved in pixels of the whole image.
    if region is not None:
        candidates = [
            replace(candidate, x=candidate.x + origin_x, y=candidate.y + origin_y)
            for candidate in candidates
        ]

    # The provenance is written last: it binds the candidates, and the
    # record of their screen, to the image and the settings.
    save_psf_candidates(resolved, candidates)

    if screen_record is not None:
        save_psf_candidate_screen(resolved, screen_record)
    else:
        # Unscreened candidates have no record; one left by an earlier
        # selection, which only an overwrite reaches, would describe others.
        paths["candidate_screen"].unlink(missing_ok=True)

    save_psf_candidates_provenance(resolved)

    return candidates


def _box_in_region(
    box: dict[str, float] | None, *, origin_x: int, origin_y: int
) -> dict[str, float] | None:
    """Move an exclusion box of the whole image into a working region."""
    if box is None:
        return None

    return {
        **box,
        "xmin": box["xmin"] - origin_x,
        "xmax": box["xmax"] - origin_x,
        "ymin": box["ymin"] - origin_y,
        "ymax": box["ymax"] - origin_y,
    }


def is_isolated_candidate(
    segmentation: np.ndarray, *, label: int, x: float, y: float, half_size: int
) -> bool:
    """Check whether a candidate is isolated within a local segmentation window.

    Parameters
    ----------
    segmentation : ndarray
        Integer segmentation map.
    label : int
        Segmentation label of the candidate source.
    x, y : float
        Candidate centroid coordinates.
    half_size : int
        Half-size of the local window inspected around the candidate.

    Returns
    -------
    bool
        True if no non-background, non-candidate labels are present in the
        local window.

    Raises
    ------
    ValueError
        If ``half_size`` is not strictly positive.
    """
    if half_size <= 0:
        raise ValueError("half_size must be positive.")

    x0 = int(round(x))
    y0 = int(round(y))

    y_min = max(0, y0 - half_size)
    y_max = min(segmentation.shape[0], y0 + half_size + 1)
    x_min = max(0, x0 - half_size)
    x_max = min(segmentation.shape[1], x0 + half_size + 1)

    local = segmentation[y_min:y_max, x_min:x_max]

    labels = set(np.unique(local))
    labels.discard(0)
    labels.discard(label)

    return len(labels) == 0


def select_psf_candidates_from_rows(
    rows: list[dict[str, Any]],
    segmentation: np.ndarray,
    *,
    target_n_candidates: int,
    min_area: float,
    max_area: float,
    min_peak: float,
    max_peak: float,
    max_eccentricity: float,
    max_elongation: float,
    edge_margin: int,
    galaxy_exclusion_box: dict[str, float] | None,
    sigma_factor: float,
    size_margin: int,
    min_half_size: int,
    max_half_size: int,
    isolation_margin: int,
    no_data: np.ndarray | None = None,
) -> list[PSFCandidate]:
    """Select high-quality empirical PSF candidates from source-catalog rows.

    Parameters
    ----------
    rows : list of dict
        Source-catalog rows converted to plain dictionaries.
    segmentation : ndarray
        Integer segmentation map.
    target_n_candidates : int
        Maximum number of candidates to return.
    min_area, max_area : float
        Allowed source-area range.
    min_peak, max_peak : float
        Allowed peak-pixel range.
    max_eccentricity : float
        Maximum allowed source eccentricity.
    max_elongation : float
        Maximum allowed source elongation.
    edge_margin : int
        Minimum distance from image edges.
    galaxy_exclusion_box : dict, optional
        Box inside which candidates are rejected.
    sigma_factor : float
        Multiplicative factor used to convert catalog sigmas to cutout size.
    size_margin : int
        Additional margin added to the estimated half-size.
    min_half_size, max_half_size : int
        Bounds applied to the estimated half-size.
    isolation_margin : int
        Extra margin used when checking candidate isolation.
    no_data : ndarray of bool, optional
        Pixels without data (AD-053), the shape of ``segmentation``. A
        candidate is rejected when its isolation window holds one.

    Returns
    -------
    list of PSFCandidate
        Selected candidates sorted by quality.

    Raises
    ------
    ValueError
        If ``target_n_candidates`` is not strictly positive.
    """
    if target_n_candidates <= 0:
        raise ValueError("target_n_candidates must be positive.")

    selected: list[PSFCandidate] = []

    for row in rows:
        if not candidate_row_passes_quality_cuts(
            row,
            segmentation.shape,
            min_area=min_area,
            max_area=max_area,
            min_peak=min_peak,
            max_peak=max_peak,
            max_eccentricity=max_eccentricity,
            max_elongation=max_elongation,
            edge_margin=edge_margin,
            galaxy_exclusion_box=galaxy_exclusion_box,
        ):
            continue

        half_size = estimate_candidate_half_size_from_sigmas(
            float(row["semimajor_sigma"]),
            float(row["semiminor_sigma"]),
            sigma_factor=sigma_factor,
            size_margin=size_margin,
            min_half_size=min_half_size,
            max_half_size=max_half_size,
        )

        isolation_half_size = half_size + int(isolation_margin)

        if not is_isolated_candidate(
            segmentation,
            label=int(row["label"]),
            x=float(row["x"]),
            y=float(row["y"]),
            half_size=isolation_half_size,
        ):
            continue

        if no_data is not None and window_holds(
            no_data, x=float(row["x"]), y=float(row["y"]), half_size=isolation_half_size
        ):
            continue

        selected.append(
            PSFCandidate(
                x=float(row["x"]),
                y=float(row["y"]),
                label=int(row["label"]),
                area=float(row["area"]),
                max_value=float(row["max_value"]),
                flux=row.get("flux"),
                eccentricity=float(row["eccentricity"]),
                elongation=float(row["elongation"]),
                semimajor_sigma=float(row["semimajor_sigma"]),
                semiminor_sigma=float(row["semiminor_sigma"]),
                half_size=half_size,
            )
        )

    selected = sorted(
        selected,
        key=lambda candidate: (
            candidate.eccentricity,
            candidate.elongation,
            -candidate.max_value,
        ),
    )

    return selected[:target_n_candidates]


def is_inside_exclusion_box(x: float, y: float, box: dict[str, float] | None) -> bool:
    """Return whether a position lies inside an exclusion box.

    Parameters
    ----------
    x, y : float
        Position coordinates.
    box : dict, optional
        Exclusion box with ``xmin``, ``xmax``, ``ymin``, and ``ymax`` entries.

    Returns
    -------
    bool
        True if the position lies inside the box.
    """
    if box is None:
        return False

    return box["xmin"] <= x <= box["xmax"] and box["ymin"] <= y <= box["ymax"]


def estimate_candidate_half_size_from_sigmas(
    semimajor_sigma: float,
    semiminor_sigma: float,
    *,
    sigma_factor: float,
    size_margin: int,
    min_half_size: int,
    max_half_size: int,
) -> int:
    """Estimate a candidate cutout half-size from source-catalog sigmas.

    Parameters
    ----------
    semimajor_sigma : float
        Semimajor-axis sigma estimate.
    semiminor_sigma : float
        Semiminor-axis sigma estimate.
    sigma_factor : float
        Multiplicative factor applied to the larger sigma.
    size_margin : int
        Additional margin added to the estimated half-size.
    min_half_size, max_half_size : int
        Bounds applied to the returned half-size.

    Returns
    -------
    int
        Clipped PSF cutout half-size.

    Raises
    ------
    ValueError
        If either sigma estimate is not strictly positive.
    """
    if semimajor_sigma <= 0 or semiminor_sigma <= 0:
        raise ValueError("semimajor_sigma and semiminor_sigma must be positive.")

    sigma = max(semimajor_sigma, semiminor_sigma)
    half_size = int(np.ceil(sigma_factor * sigma)) + int(size_margin)

    return int(np.clip(half_size, min_half_size, max_half_size))


def candidate_row_passes_quality_cuts(
    row: dict[str, Any],
    image_shape: tuple[int, int],
    *,
    min_area: float,
    max_area: float,
    min_peak: float,
    max_peak: float,
    max_eccentricity: float,
    max_elongation: float,
    edge_margin: int,
    galaxy_exclusion_box: dict[str, float] | None,
) -> bool:
    """Check whether a source-catalog row satisfies PSF-candidate cuts.

    Parameters
    ----------
    row : dict
        Source-catalog row.
    image_shape : tuple of int
        Image shape as ``(ny, nx)``.
    min_area, max_area : float
        Allowed source-area range.
    min_peak, max_peak : float
        Allowed peak-pixel range.
    max_eccentricity : float
        Maximum allowed source eccentricity.
    max_elongation : float
        Maximum allowed source elongation.
    edge_margin : int
        Minimum distance from image edges.
    galaxy_exclusion_box : dict, optional
        Box inside which candidates are rejected.

    Returns
    -------
    bool
        True if the row passes all cuts.
    """
    x = float(row["x"])
    y = float(row["y"])

    ny, nx = image_shape

    if x <= edge_margin or x >= nx - edge_margin:
        return False

    if y <= edge_margin or y >= ny - edge_margin:
        return False

    if is_inside_exclusion_box(x, y, galaxy_exclusion_box):
        return False

    area = float(row["area"])
    max_value = float(row["max_value"])
    eccentricity = float(row["eccentricity"])
    elongation = float(row["elongation"])

    return (
        min_area <= area <= max_area
        and min_peak <= max_value <= max_peak
        and eccentricity < max_eccentricity
        and elongation < max_elongation
    )


def _as_float(value):
    """Convert scalar-like catalog values to float."""
    if hasattr(value, "value"):
        value = value.value

    return float(value)


def source_catalog_to_candidate_rows(source_catalog) -> list[dict[str, Any]]:
    """Convert a Photutils source catalog to PSF-candidate rows.

    Parameters
    ----------
    source_catalog : photutils.segmentation.SourceCatalog
        Source catalog produced during image preparation.

    Returns
    -------
    list of dict
        Plain-Python source rows used by PSF-candidate selection.
    """
    rows: list[dict[str, Any]] = []

    for source in source_catalog:
        rows.append(
            {
                "label": int(source.label),
                "x": _as_float(source.x_centroid),
                "y": _as_float(source.y_centroid),
                "area": _as_float(source.area),
                "max_value": _as_float(source.max_value),
                "eccentricity": _as_float(source.eccentricity),
                "elongation": _as_float(source.elongation),
                "semimajor_sigma": _as_float(source.semimajor_axis),
                "semiminor_sigma": _as_float(source.semiminor_axis),
                "flux": _as_float(source.segment_flux),
            }
        )

    return rows
