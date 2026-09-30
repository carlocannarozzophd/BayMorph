import numpy as np
import pytest

from baymorph.pipeline import (
    is_isolated_candidate,
    load_psf_candidates,
    pipeline_paths,
    PSFCandidate,
    resolve_config,
    save_psf_candidates,
    select_psf_candidates_from_rows,
    source_catalog_to_candidate_rows,
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
