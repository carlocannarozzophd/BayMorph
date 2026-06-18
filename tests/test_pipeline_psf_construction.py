import numpy as np

import pytest

from baymorph import (
    load_empirical_psf,
    pipeline_paths,
    resolve_config,
    save_empirical_psf,
    extract_centered_cutout,
    normalize_psf,
    stack_psf_cutouts,
    subtract_outer_background,
    centroid_from_flux,
    recenter_cutout,
)


def test_save_and_load_empirical_psf_roundtrip(tmp_path):
    config = {
        "object": {"name": "NGC1234"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": str(tmp_path / "outputs"), "overwrite": True},
    }

    resolved = resolve_config(config)

    psf = np.ones((9, 9))
    psf /= psf.sum()

    path = save_empirical_psf(resolved, psf)

    assert path == pipeline_paths(resolved)["psf"]["psf_fits"]
    assert path.exists()

    loaded = load_empirical_psf(resolved)

    assert np.allclose(loaded, psf)


def test_normalize_psf_returns_unit_sum():
    psf = np.ones((3, 3))

    normalized = normalize_psf(psf)

    assert np.isclose(normalized.sum(), 1.0)


def test_normalize_psf_rejects_zero_sum():
    with pytest.raises(ValueError, match="positive and finite"):
        normalize_psf(np.zeros((3, 3)))


def test_extract_centered_cutout_returns_expected_shape():
    image = np.arange(100).reshape(10, 10)

    cutout = extract_centered_cutout(image, x=5, y=5, half_size=2)

    assert cutout.shape == (5, 5)
    assert cutout[2, 2] == image[5, 5]


def test_extract_centered_cutout_raises_at_edge():
    image = np.zeros((10, 10))

    with pytest.raises(ValueError, match="outside the image boundaries"):
        extract_centered_cutout(image, x=1, y=1, half_size=3)


def test_subtract_outer_background_removes_border_level():
    cutout = np.ones((9, 9)) * 5
    cutout[4, 4] = 15

    corrected = subtract_outer_background(cutout, border_width=2)

    assert np.isclose(np.median(corrected[:2, :]), 0.0)
    assert np.isclose(corrected[4, 4], 10.0)


def test_stack_psf_cutouts_returns_normalized_stack():
    cutouts = [
        np.ones((5, 5)),
        np.ones((5, 5)) * 2,
        np.ones((5, 5)) * 3,
    ]

    psf = stack_psf_cutouts(cutouts)

    assert psf.shape == (5, 5)
    assert np.isclose(psf.sum(), 1.0)


def test_centroid_from_flux_recovers_peak_position():
    cutout = np.zeros((7, 7))
    cutout[2, 4] = 10.0

    x, y = centroid_from_flux(cutout)

    assert np.isclose(x, 4.0)
    assert np.isclose(y, 2.0)


def test_centroid_from_flux_rejects_empty_cutout():
    with pytest.raises(ValueError, match="non-positive flux"):
        centroid_from_flux(np.zeros((5, 5)))


def test_recenter_cutout_moves_peak_toward_center():
    cutout = np.zeros((9, 9))
    cutout[2, 6] = 10.0

    recentered = recenter_cutout(cutout)

    x, y = centroid_from_flux(recentered)

    assert np.isclose(x, 4.0, atol=0.5)
    assert np.isclose(y, 4.0, atol=0.5)