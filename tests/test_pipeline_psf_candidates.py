import numpy as np
import pytest

from baymorph import (
    PSFCandidate,
    estimate_candidate_half_size,
    is_isolated_candidate,
    load_psf_candidates,
    pipeline_paths,
    resolve_config,
    save_psf_candidates,
    select_psf_candidates_from_rows
)


def test_save_and_load_psf_candidates_roundtrip(tmp_path):
    config = {
        "object": {"name": "NGC1234"},
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


def test_estimate_candidate_half_size_uses_limits():
    assert estimate_candidate_half_size(2.0, scale_factor=4.0, min_half_size=12, max_half_size=40) == 12
    assert estimate_candidate_half_size(5.0, scale_factor=4.0, min_half_size=12, max_half_size=40) == 20
    assert estimate_candidate_half_size(20.0, scale_factor=4.0, min_half_size=12, max_half_size=40) == 40


def test_estimate_candidate_half_size_rejects_nonpositive_fwhm():
    with pytest.raises(ValueError, match="fwhm must be positive"):
        estimate_candidate_half_size(0.0)


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


def test_select_psf_candidates_from_rows_keeps_isolated_bright_sources():
    segmentation = np.zeros((50, 50), dtype=int)
    segmentation[10, 10] = 1
    segmentation[30, 30] = 2
    segmentation[32, 32] = 3

    rows = [
        {"label": 1, "x": 10, "y": 10, "flux": 100.0, "fwhm": 2.0},
        {"label": 2, "x": 30, "y": 30, "flux": 200.0, "fwhm": 2.0},
    ]

    candidates = select_psf_candidates_from_rows(
        rows,
        segmentation,
        target_n_candidates=5,
        isolation_scale_factor=1.0,
        min_half_size=2,
        max_half_size=10,
    )

    assert len(candidates) == 1
    assert candidates[0].label == 1


def test_select_psf_candidates_from_rows_limits_number_of_candidates():
    segmentation = np.zeros((100, 100), dtype=int)
    rows = []

    for label, flux, x in [
        (1, 100.0, 10),
        (2, 300.0, 30),
        (3, 200.0, 50),
    ]:
        segmentation[10, x] = label
        rows.append(
            {
                "label": label,
                "x": x,
                "y": 10,
                "flux": flux,
                "fwhm": 2.0,
            }
        )

    candidates = select_psf_candidates_from_rows(
        rows,
        segmentation,
        target_n_candidates=2,
        isolation_scale_factor=1.0,
        min_half_size=1,
        max_half_size=10,
    )

    assert [candidate.label for candidate in candidates] == [2, 3]


def test_select_psf_candidates_from_rows_rejects_nonpositive_target_count():
    with pytest.raises(ValueError, match="target_n_candidates must be positive"):
        select_psf_candidates_from_rows(
            [],
            np.zeros((10, 10), dtype=int),
            target_n_candidates=0,
            isolation_scale_factor=1.0,
            min_half_size=1,
            max_half_size=10,
        )