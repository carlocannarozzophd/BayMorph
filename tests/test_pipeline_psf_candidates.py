import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from scipy.special import erf

from baymorph.pipeline import (
    is_isolated_candidate,
    load_psf_candidates,
    make_output_tree,
    pipeline_paths,
    PSFCandidate,
    resolve_config,
    save_psf_candidates,
    select_psf_candidates_from_rows,
    source_catalog_to_candidate_rows,
)
from baymorph.pipeline.checkpoint_signing import canonical_json_sha256
from baymorph.pipeline.image_preparation import detect_real_fits_sources
from baymorph.pipeline.prepared_identity import file_content_identity
from baymorph.pipeline.psf_candidate_screen import (
    SIZE_SEQUENCE_V1_SHA256,
    PointSourceScreenChangeError,
    PointSourceScreenError,
)
from baymorph.pipeline.psf_candidates import (
    build_psf_candidates_provenance,
    load_psf_candidate_screen,
    run_psf_candidate_selection_from_config,
    save_psf_candidates_provenance,
    validate_psf_candidates_provenance,
)



def make_candidate_row(
    label,
    x,
    y,
    max_value=1000.0,
    eccentricity=0.1,
    elongation=1.1,
):
    return {
        "label": label,
        "x": x,
        "y": y,
        "area": 50.0,
        "max_value": max_value,
        "eccentricity": eccentricity,
        "elongation": elongation,
        "semimajor_sigma": 2.0,
        "semiminor_sigma": 1.5,
    }


def test_save_and_load_psf_candidates_roundtrip(tmp_path):
    config = {
        "object": {"name": "example_galaxy"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": str(tmp_path / "outputs")},
    }

    resolved = resolve_config(config)

    candidates = [
        PSFCandidate(x=10.0, y=20.0, label=1, flux=1000.0, fwhm=2.5, half_size=12),
        PSFCandidate(x=30.0, y=40.0, label=2, flux=900.0, fwhm=2.7, half_size=12),
    ]

    path = save_psf_candidates(resolved, candidates)

    assert path == pipeline_paths(resolved)["psf"]["candidates"]
    assert path.exists()

    loaded = load_psf_candidates(resolved)

    assert loaded == candidates


def test_isolated_candidate_returns_true_without_neighbor_labels():
    segmentation = np.zeros((20, 20), dtype=int)
    segmentation[10, 10] = 3

    assert is_isolated_candidate(
        segmentation,
        label=3,
        x=10,
        y=10,
        half_size=3,
    )


def test_isolated_candidate_returns_false_with_neighbor_label():
    segmentation = np.zeros((20, 20), dtype=int)
    segmentation[10, 10] = 3
    segmentation[12, 12] = 7

    assert not is_isolated_candidate(
        segmentation,
        label=3,
        x=10,
        y=10,
        half_size=3,
    )


def test_select_psf_candidates_from_rows_keeps_isolated_sources():
    segmentation = np.zeros((50, 50), dtype=int)
    segmentation[10, 10] = 1
    segmentation[30, 30] = 2
    segmentation[32, 32] = 3

    rows = [
        make_candidate_row(1, 10, 10, max_value=100.0),
        make_candidate_row(2, 30, 30, max_value=200.0),
    ]

    candidates = select_psf_candidates_from_rows(
        rows,
        segmentation,
        target_n_candidates=5,
        min_area=20,
        max_area=300,
        min_peak=50,
        max_peak=50000,
        max_eccentricity=0.5,
        max_elongation=1.5,
        edge_margin=0,
        galaxy_exclusion_box=None,
        sigma_factor=1.0,
        size_margin=0,
        min_half_size=2,
        max_half_size=10,
        isolation_margin=0,
    )

    assert len(candidates) == 1
    assert candidates[0].label == 1


def test_select_psf_candidates_from_rows_limits_number_of_candidates():
    segmentation = np.zeros((100, 100), dtype=int)

    rows = [
        make_candidate_row(1, 10, 10, max_value=100.0, eccentricity=0.3),
        make_candidate_row(2, 30, 10, max_value=300.0, eccentricity=0.1),
        make_candidate_row(3, 50, 10, max_value=200.0, eccentricity=0.2),
    ]

    for row in rows:
        segmentation[int(row["y"]), int(row["x"])] = int(row["label"])

    candidates = select_psf_candidates_from_rows(
        rows,
        segmentation,
        target_n_candidates=2,
        min_area=20,
        max_area=300,
        min_peak=50,
        max_peak=50000,
        max_eccentricity=0.5,
        max_elongation=1.5,
        edge_margin=0,
        galaxy_exclusion_box=None,
        sigma_factor=1.0,
        size_margin=0,
        min_half_size=1,
        max_half_size=10,
        isolation_margin=0,
    )

    assert [candidate.label for candidate in candidates] == [2, 3]


def test_select_psf_candidates_from_rows_rejects_nonpositive_target_count():
    with pytest.raises(ValueError, match="target_n_candidates must be positive"):
        select_psf_candidates_from_rows(
            [],
            np.zeros((10, 10), dtype=int),
            target_n_candidates=0,
            min_area=20,
            max_area=300,
            min_peak=50,
            max_peak=50000,
            max_eccentricity=0.5,
            max_elongation=1.5,
            edge_margin=0,
            galaxy_exclusion_box=None,
            sigma_factor=1.0,
            size_margin=0,
            min_half_size=1,
            max_half_size=10,
            isolation_margin=0,
        )


def test_source_catalog_to_candidate_rows_is_callable():
    assert callable(source_catalog_to_candidate_rows)


# --- point-source screen ----------------------------------------------------

SCREEN_MARKER = {
    "rule": "size_sequence_v1",
    "constants_sha256": SIZE_SEQUENCE_V1_SHA256,
}


def _pixel_gaussian(image, x0, y0, sigma, flux, *, q=1.0):
    """Add a Gaussian integrated over the pixels, axis ratio q along y."""
    ny, nx = image.shape
    edges_x = np.arange(nx + 1) - 0.5
    edges_y = np.arange(ny + 1) - 0.5
    sigma_y = sigma * q
    fx = np.diff(0.5 * (1.0 + erf((edges_x - x0) / (np.sqrt(2.0) * sigma))))
    fy = np.diff(0.5 * (1.0 + erf((edges_y - y0) / (np.sqrt(2.0) * sigma_y))))
    image += flux * np.outer(fy, fx)


def _screen_field(path: Path, *, stars=20, galaxies=15, cosmic_rays=10):
    """Write isolated stars, compact galaxies and raw cosmic rays on a grid.

    Stars are Gaussians of FWHM 2.7 px; galaxies are 1.4 to 2.1 times wider;
    cosmic rays are one or two bright pixels. Returns the positions by kind.
    """
    rng = np.random.default_rng(11)
    n_side = 7
    image = rng.normal(0.0, 1.0, (60 * n_side + 60, 60 * n_side + 60))
    kinds = ["star"] * stars + ["galaxy"] * galaxies + ["cosmic_ray"] * cosmic_rays
    positions = {"star": [], "galaxy": [], "cosmic_ray": []}

    for index, kind in enumerate(kinds):
        row, column = divmod(index, n_side)
        x = 60.0 + 60.0 * column + rng.uniform(-0.5, 0.5)
        y = 60.0 + 60.0 * row + rng.uniform(-0.5, 0.5)

        if kind == "star":
            peak = float(np.exp(rng.uniform(np.log(80.0), np.log(800.0))))
            _pixel_gaussian(image, x, y, 1.15, peak * 2.0 * np.pi * 1.15**2)
        elif kind == "galaxy":
            sigma = 1.15 * rng.uniform(1.4, 2.1)
            peak = float(rng.uniform(40.0, 200.0))
            _pixel_gaussian(
                image,
                x,
                y,
                sigma,
                peak * 2.0 * np.pi * sigma**2,
                q=rng.uniform(0.8, 1.0),
            )
        else:
            x, y = float(round(x)), float(round(y))
            image[int(y), int(x)] += rng.uniform(150.0, 600.0)

            if index % 2:
                image[int(y), int(x) + 1] += rng.uniform(100.0, 400.0)

        positions[kind].append((x, y))

    fits.writeto(path, image.astype(np.float32))

    return positions


def _screen_config(tmp_path: Path, image: Path, *, screen=None, name="field"):
    """Return the resolved configuration of a selection on ``image``."""
    selection = {} if screen is None else {"point_source_screen": screen}

    return resolve_config(
        {
            "object": {"name": name},
            "data": {"image_path": str(image)},
            "psf": {"empirical": {"candidate_selection": selection}},
            "outputs": {"root": str(tmp_path / "runs")},
        }
    )


def _nearest(positions, candidate, *, tolerance=2.0):
    """Return whether a candidate lies on one of the positions."""
    return any(
        np.hypot(candidate.x - x, candidate.y - y) < tolerance for x, y in positions
    )


def test_selection_without_the_screen_writes_what_earlier_releases_wrote(tmp_path):
    """Under ``none`` the products are byte-identical to the shape selection."""
    _screen_field(tmp_path / "field.fits")
    config = _screen_config(tmp_path, tmp_path / "field.fits", screen="none")

    candidates = run_psf_candidate_selection_from_config(config)

    image = fits.getdata(tmp_path / "field.fits").astype(float)
    detection = config["preparation"]["detection"]
    found = detect_real_fits_sources(
        image,
        detection_sigma=detection["n_sigma"],
        detection_n_pixels=detection["n_pixels"],
        detection_n_levels=detection["n_levels"],
    )
    selection = dict(config["psf"]["empirical"]["candidate_selection"])
    expected = select_psf_candidates_from_rows(
        source_catalog_to_candidate_rows(found.catalog),
        found.segmentation.data,
        **{
            key: selection[key]
            for key in (
                "target_n_candidates",
                "min_area",
                "max_area",
                "min_peak",
                "max_peak",
                "max_eccentricity",
                "max_elongation",
                "edge_margin",
                "galaxy_exclusion_box",
                "sigma_factor",
                "size_margin",
                "min_half_size",
                "max_half_size",
                "isolation_margin",
            )
        },
    )
    selection.pop("point_source_screen")
    image_path = (tmp_path / "field.fits").resolve()
    provenance = {
        "schema_version": 1,
        "input_fits": {
            "path": str(image_path),
            "sha256": file_content_identity(image_path)["sha256"],
        },
        "detection_config_sha256": canonical_json_sha256(detection),
        "candidate_selection_config_sha256": canonical_json_sha256(selection),
    }
    paths = pipeline_paths(config)["psf"]

    assert candidates == expected
    assert paths["candidates"].read_text() == json.dumps(
        [asdict(candidate) for candidate in expected], indent=2
    )
    assert paths["candidates_provenance"].read_text(encoding="utf-8") == (
        json.dumps(provenance, indent=2) + "\n"
    )
    assert not paths["candidate_screen"].exists()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_selecting_the_whole_pool_and_truncating_it_keeps_the_order(seed):
    """The pool cut to 30 is the selection of 30, ties included."""
    rng = np.random.default_rng(seed)
    segmentation = np.zeros((400, 400), dtype=int)
    rows = []

    for label in range(1, 61):
        x, y = rng.uniform(40.0, 360.0, size=2)
        segmentation[int(round(y)), int(round(x))] = label
        rows.append(
            {
                **make_candidate_row(
                    label,
                    x,
                    y,
                    max_value=float(rng.choice([100.0, 200.0])),
                    eccentricity=float(rng.choice([0.1, 0.2, 0.3])),
                    elongation=float(rng.choice([1.1, 1.2])),
                ),
            }
        )

    cuts = dict(
        min_area=20,
        max_area=300,
        min_peak=50,
        max_peak=50000,
        max_eccentricity=0.5,
        max_elongation=1.5,
        edge_margin=0,
        galaxy_exclusion_box=None,
        sigma_factor=1.0,
        size_margin=0,
        min_half_size=1,
        max_half_size=1,
        isolation_margin=0,
    )

    pool = select_psf_candidates_from_rows(
        rows, segmentation, target_n_candidates=len(rows), **cuts
    )

    assert len(pool) == 60
    assert pool[:30] == select_psf_candidates_from_rows(
        rows, segmentation, target_n_candidates=30, **cuts
    )


def test_the_screen_keeps_only_the_stars_and_records_its_verdicts(tmp_path):
    """Galaxies and cosmic rays are rejected; the next members replace them."""
    positions = _screen_field(tmp_path / "field.fits")
    config = _screen_config(tmp_path, tmp_path / "field.fits")

    candidates = run_psf_candidate_selection_from_config(config)

    record = load_psf_candidate_screen(config)
    provenance = json.loads(
        pipeline_paths(config)["psf"]["candidates_provenance"].read_text()
    )
    kinds = {
        kind: sum(_nearest(points, c) for c in candidates)
        for kind, points in positions.items()
    }

    assert kinds == {"star": 20, "galaxy": 0, "cosmic_ray": 0}
    assert len(candidates) == min(30, record["n_members"]) == 20
    assert record["n_kept"] == 20
    assert record["counts"]["larger_than_stellar_sequence"] >= 10
    assert record["counts"]["sharper_than_point_source"] >= 5
    assert provenance["point_source_screen"] == SCREEN_MARKER
    assert provenance == build_psf_candidates_provenance(config)


def test_a_field_without_stars_stops_and_saves_nothing(tmp_path):
    """A stop writes no candidates, record or provenance."""
    _screen_field(tmp_path / "field.fits", stars=0, galaxies=20, cosmic_rays=10)
    config = _screen_config(tmp_path, tmp_path / "field.fits")

    with pytest.raises(PointSourceScreenError) as error:
        run_psf_candidate_selection_from_config(config)

    record = error.value.record
    paths = pipeline_paths(config)["psf"]

    assert error.value.code in {"too_few", "no_sequence"}
    assert "No PSF candidate was saved" in str(error.value)
    assert "The whole image was searched." in str(error.value)
    assert "a larger data.region" not in str(error.value).lower()
    assert len(record["candidates"]) == record["n_pool"] > 0
    assert all(candidate["reason"] for candidate in record["candidates"])
    assert not paths["candidates"].exists()
    assert not paths["candidate_screen"].exists()
    assert not paths["candidates_provenance"].exists()


# Candidate-selection settings of two real runs, a JWST mosaic and an HST
# image, copied from their resolved configurations; no data are read.
JWST_SELECTION = {
    "auto": True,
    "edge_margin": 30,
    "galaxy_exclusion_box": None,
    "isolation_margin": 2,
    "max_area": 800,
    "max_eccentricity": 0.8,
    "max_elongation": 2.5,
    "max_half_size": 30,
    "max_peak": 80000,
    "min_area": 10,
    "min_half_size": 12,
    "min_peak": 0.2138787869129393,
    "min_peak_snr": 30.0,
    "sigma_factor": 4.0,
    "size_margin": 4,
    "target_n_candidates": 30,
}
HST_SELECTION = {
    **JWST_SELECTION,
    "max_peak": 110.5,
    "min_peak": 0.25168533815837185,
}


@pytest.mark.parametrize(
    ("selection", "digest"),
    [
        (
            JWST_SELECTION,
            "597f5b6b64ab20b34f3816898626e7f0a02126f61cf603d88675db7c689976dd",
        ),
        (
            HST_SELECTION,
            "4476409dd5e15f2d642474d01e0e418cf0bffcf9ec5afc954b3713cd0bb8d110",
        ),
    ],
)
def test_the_neutral_screen_keeps_the_identity_of_saved_candidates(
    tmp_path, selection, digest
):
    """Under none the digest is that of earlier releases; the rule changes it."""
    image = tmp_path / "image.fits"
    fits.writeto(image, np.zeros((8, 8)))

    def provenance(screen):
        config = {
            "data": {"image_path": str(image)},
            "preparation": {"detection": {"n_sigma": 3.0}},
            "psf": {
                "empirical": {
                    "candidate_selection": {
                        **selection,
                        "point_source_screen": screen,
                    }
                }
            },
        }

        return build_psf_candidates_provenance(config)

    neutral = provenance("none")
    screened = provenance("size_sequence_v1")

    assert neutral["candidate_selection_config_sha256"] == digest
    assert "point_source_screen" not in neutral
    assert screened["candidate_selection_config_sha256"] != digest
    assert screened["point_source_screen"] == SCREEN_MARKER


def test_candidates_of_one_screen_are_refused_under_the_other(tmp_path):
    """A provenance without the marker, or with it, binds its own rule only."""
    image = tmp_path / "image.fits"
    fits.writeto(image, np.zeros((8, 8)))
    neutral = _screen_config(tmp_path, image, screen="none")
    screened = _screen_config(tmp_path, image)
    make_output_tree(neutral)

    save_psf_candidates_provenance(neutral)
    validate_psf_candidates_provenance(neutral)

    # Only the screen differs: the refusal names the declaration that reuses
    # the candidates, and does not ask for an overwrite.
    with pytest.raises(PointSourceScreenChangeError) as error:
        validate_psf_candidates_provenance(screened)

    assert "selected without the point-source screen" in str(error.value)
    assert "point_source_screen: none to reuse them unchanged" in str(error.value)
    assert "overwrite" not in str(error.value)

    save_psf_candidates_provenance(screened)
    validate_psf_candidates_provenance(screened)

    with pytest.raises(PointSourceScreenChangeError, match="size_sequence_v1 to"):
        validate_psf_candidates_provenance(neutral)

    # With another difference, the usual refusal names the screen as well.
    other = _screen_config(tmp_path, image, screen="none")
    other["preparation"]["detection"]["n_sigma"] += 1.0

    with pytest.raises(ValueError, match="candidate-selection setting") as error:
        validate_psf_candidates_provenance(other)

    assert not isinstance(error.value, PointSourceScreenChangeError)
    assert "also selected with the screen size_sequence_v1" in str(error.value)


def _ordered_field(path: Path, *, stars=36, galaxies=12):
    """Write round galaxies that sort before slightly elongated stars.

    The stars share an elongated PSF (axis ratio 0.85, eccentricity about
    0.5), and the galaxies are round, so selection by shape ranks every
    galaxy before the stars. Returns the positions by kind.
    """
    rng = np.random.default_rng(23)
    n_side = 7
    image = rng.normal(0.0, 1.0, (60 * n_side + 60, 60 * n_side + 60))
    kinds = ["galaxy"] * galaxies + ["star"] * stars
    positions = {"star": [], "galaxy": []}

    for index, kind in enumerate(kinds):
        row, column = divmod(index, n_side)
        x = 60.0 + 60.0 * column + rng.uniform(-0.5, 0.5)
        y = 60.0 + 60.0 * row + rng.uniform(-0.5, 0.5)

        if kind == "star":
            peak = float(np.exp(rng.uniform(np.log(80.0), np.log(800.0))))
            flux = peak * 2.0 * np.pi * 1.15**2 * 0.85
            _pixel_gaussian(image, x, y, 1.15, flux, q=0.85)
        else:
            sigma = 1.15 * rng.uniform(1.5, 2.0)
            flux = float(rng.uniform(150.0, 400.0)) * 2.0 * np.pi * sigma**2
            _pixel_gaussian(image, x, y, sigma, flux)

        positions[kind].append((x, y))

    fits.writeto(path, image.astype(np.float32))

    return positions


def test_members_beyond_the_first_by_shape_replace_rejected_sources(tmp_path):
    """Galaxies first in shape order give their places to later stars."""
    positions = _ordered_field(tmp_path / "field.fits")
    image = tmp_path / "field.fits"
    by_shape = run_psf_candidate_selection_from_config(
        _screen_config(tmp_path, image, screen="none", name="by_shape")
    )
    config = _screen_config(tmp_path, image, name="screened")
    screened = run_psf_candidate_selection_from_config(config)
    record = load_psf_candidate_screen(config)

    def stars(candidates):
        return [c for c in candidates if _nearest(positions["star"], c)]

    first_by_shape = {(c.x, c.y) for c in by_shape}
    replacements = [c for c in screened if (c.x, c.y) not in first_by_shape]

    galaxies_by_shape = len(by_shape) - len(stars(by_shape))

    # By shape, the round galaxies come first and take places among the 30.
    assert len(by_shape) == 30
    assert galaxies_by_shape >= 10
    # Screened, the 30 kept are stars: those among the first 30 by shape,
    # and as many stars ranked beyond 30 as there were galaxies there.
    assert len(screened) == record["n_kept"] == 30
    assert stars(screened) == screened
    assert len(replacements) == galaxies_by_shape
    assert record["n_members"] == 36
    assert record["counts"]["larger_than_stellar_sequence"] == galaxies_by_shape


def test_a_dense_group_of_galaxies_without_stars_stops(tmp_path):
    """Galaxies dense in size but not one size stop the screen."""
    rng = np.random.default_rng(29)
    n_side = 7
    pixels = rng.normal(0.0, 1.0, (60 * n_side + 60, 60 * n_side + 60))

    for index in range(42):
        row, column = divmod(index, n_side)
        x = 60.0 + 60.0 * column + rng.uniform(-0.5, 0.5)
        y = 60.0 + 60.0 * row + rng.uniform(-0.5, 0.5)
        sigma = 1.15 * float(np.exp(rng.uniform(np.log(1.3), np.log(2.2))))
        flux = float(rng.uniform(100.0, 400.0)) * 2.0 * np.pi * sigma**2
        _pixel_gaussian(pixels, x, y, sigma, flux, q=rng.uniform(0.9, 1.0))

    fits.writeto(tmp_path / "field.fits", pixels.astype(np.float32))
    config = _screen_config(tmp_path, tmp_path / "field.fits")

    with pytest.raises(PointSourceScreenError) as error:
        run_psf_candidate_selection_from_config(config)

    record = error.value.record
    paths = pipeline_paths(config)["psf"]

    assert error.value.code == "no_sequence"
    assert record["detail"] in {"not_standing_out", "too_wide"}
    assert record["sequence"]["n_core"] >= 6
    assert record["n_search"] >= 30
    assert not any(path.exists() for path in paths.values() if path.suffix)


def test_a_stale_screen_record_stops_the_selection_before_anything_is_written(
    tmp_path,
):
    """Without overwrite, nothing is written beside an earlier record."""
    _screen_field(tmp_path / "field.fits")
    config = _screen_config(tmp_path, tmp_path / "field.fits")
    paths = pipeline_paths(config)["psf"]
    make_output_tree(config)
    paths["candidate_screen"].write_text("{}\n")

    with pytest.raises(FileExistsError, match="psf_candidate_screen.json"):
        run_psf_candidate_selection_from_config(config)

    assert not paths["candidates"].exists()
    assert not paths["candidates_provenance"].exists()
    assert paths["candidate_screen"].read_text() == "{}\n"


def test_selecting_by_shape_again_removes_an_earlier_screen_record(tmp_path):
    """Overwritten without the screen, the candidates have no record."""
    _screen_field(tmp_path / "field.fits")
    image = tmp_path / "field.fits"
    run_psf_candidate_selection_from_config(_screen_config(tmp_path, image))
    config = _screen_config(tmp_path, image, screen="none")
    config["outputs"]["overwrite"] = True

    run_psf_candidate_selection_from_config(config)

    paths = pipeline_paths(config)["psf"]

    assert not paths["candidate_screen"].exists()
    assert "point_source_screen" not in json.loads(
        paths["candidates_provenance"].read_text()
    )


def test_a_stop_in_a_working_region_names_the_region(tmp_path):
    """The remedies say where the stars were searched."""
    _screen_field(tmp_path / "field.fits", stars=0, galaxies=20, cosmic_rays=10)
    config = resolve_config(
        {
            "object": {"name": "region"},
            "data": {
                "image_path": str(tmp_path / "field.fits"),
                "pixel_scale": 0.1,
                "region": {"size": 30.0},
                "target_position": {"x": 240.0, "y": 240.0},
            },
            "outputs": {"root": str(tmp_path / "runs")},
        }
    )

    with pytest.raises(PointSourceScreenError) as error:
        run_psf_candidate_selection_from_config(config)

    message = str(error.value)

    assert "The stars were searched within the working region x 90-390, y 90-390" in (
        message
    )
    assert "data.region" in message.split("of the image.")[1]
    assert not pipeline_paths(config)["psf"]["candidates"].exists()
