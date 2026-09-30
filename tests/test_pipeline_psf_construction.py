import numpy as np

import pytest

import json
from astropy.io import fits

from baymorph.pipeline import (
    centroid_from_flux,
    crop_center,
    extract_centered_cutout,
    load_empirical_psf,
    normalize_psf,
    pipeline_paths,
    recenter_cutout,
    resolve_config,
    save_empirical_psf,
    save_empirical_psf_metadata,
    stack_psf_cutouts,
    subtract_outer_background,
)
from baymorph.pipeline.psf_construction import (
    EMPIRICAL_PSF_CENTROID_TOLERANCE_PIXELS,
    EMPIRICAL_PSF_PROVENANCE_SCHEMA_VERSION,
    build_empirical_psf_provenance,
    empirical_psf_centroid_tolerance_pixels,
    validate_empirical_psf,
)


def _save_valid_empirical_psf_bundle(tmp_path):
    """Create a complete empirical-PSF product with immutable provenance."""
    image_path = tmp_path / "image.fits"
    fits.writeto(
        image_path,
        np.arange(81, dtype=float).reshape(9, 9),
    )

    config = {
        "object": {"name": "example_galaxy"},
        "data": {"image_path": str(image_path)},
        "outputs": {
            "root": str(tmp_path / "outputs"),
            "overwrite": True,
        },
    }
    resolved = resolve_config(config)
    psf = np.zeros((9, 9), dtype=float)
    psf[4, 4] = 1.0

    save_empirical_psf(resolved, psf)

    candidates_path = pipeline_paths(resolved)["psf"]["candidates"]
    candidates_path.write_text(
        '[{"x": 4.0, "y": 4.0, "half_size": 4}]',
        encoding="utf-8",
    )

    validation = validate_empirical_psf(
        psf,
        centroid_tolerance_pixels=(
            empirical_psf_centroid_tolerance_pixels(resolved)
        ),
    )
    save_empirical_psf_metadata(
        resolved,
        {
            "schema_version": EMPIRICAL_PSF_PROVENANCE_SCHEMA_VERSION,
            "type": "empirical",
            "validation": validation,
            "provenance": build_empirical_psf_provenance(
                resolved
            ),
        },
    )

    return resolved, psf, image_path



def test_save_and_load_empirical_psf_roundtrip(tmp_path):
    resolved, psf, _ = _save_valid_empirical_psf_bundle(
        tmp_path
    )
    path = pipeline_paths(resolved)["psf"]["psf_fits"]

    assert path == pipeline_paths(resolved)["psf"]["psf_fits"]
    assert path.exists()

    loaded = load_empirical_psf(resolved)

    assert np.allclose(
        loaded.kernel(),
        psf,
    )


def test_normalize_psf_returns_unit_sum():
    psf = np.ones((3, 3))

    normalized = normalize_psf(psf)

    assert np.isclose(normalized.sum(), 1.0)


def test_normalize_psf_rejects_zero_sum():
    with pytest.raises(ValueError, match="positive and finite"):
        normalize_psf(np.zeros((3, 3)))


def test_empirical_psf_centroid_tolerance_is_resolved_explicitly():
    """The production centering gate must be present in resolved configs."""
    config = {
        "object": {"name": "example_galaxy"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": "outputs"},
    }

    resolved = resolve_config(config)

    assert resolved["psf"]["empirical"]["construction"][
        "centroid_tolerance_pixels"
    ] == EMPIRICAL_PSF_CENTROID_TOLERANCE_PIXELS


@pytest.mark.parametrize(
    ("psf", "message"),
    [
        (np.ones(5), "2D array"),
        (np.ones((4, 4)), "odd-sized kernel"),
        (np.ones((5, 4)), "odd-sized kernel"),
        (
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, np.nan, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
            "finite values",
        ),
        (
            np.array(
                [
                    [0.0, -0.1, 0.0],
                    [0.0, 1.1, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
            "non-negative values",
        ),
        (np.zeros((3, 3)), "positive finite flux"),
        (
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
            "normalized to unit sum",
        ),
    ],
)
def test_validate_empirical_psf_rejects_invalid_raw_kernel(
    psf,
    message,
):
    """Inference must not silently sanitize an invalid stored PSF."""
    with pytest.raises(ValueError, match=message):
        validate_empirical_psf(
            psf,
            centroid_tolerance_pixels=0.25,
        )


def test_validate_empirical_psf_rejects_off_center_kernel():
    """A normalized finite PSF must also satisfy the frozen centroid gate."""
    psf = np.zeros((7, 7), dtype=float)
    psf[3, 4] = 1.0

    with pytest.raises(
        ValueError,
        match="flux centroid is not centered",
    ):
        validate_empirical_psf(
            psf,
            centroid_tolerance_pixels=0.25,
        )


def test_load_empirical_psf_rejects_stale_input_fits(tmp_path):
    """PSF reuse must fail if its source observation changes byte-for-byte."""
    resolved, _, image_path = _save_valid_empirical_psf_bundle(
        tmp_path
    )
    fits.writeto(
        image_path,
        np.ones((9, 9), dtype=float),
        overwrite=True,
    )

    with pytest.raises(
        ValueError,
        match="provenance does not match",
    ):
        load_empirical_psf(resolved)


def test_load_empirical_psf_rejects_stale_candidate_catalog(tmp_path):
    """The exact vetted candidate catalog must remain bound to the PSF."""
    resolved, _, _ = _save_valid_empirical_psf_bundle(
        tmp_path
    )
    candidates_path = pipeline_paths(resolved)["psf"]["candidates"]
    candidates_path.write_text(
        '[{"x": 5.0, "y": 4.0, "half_size": 4}]',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="provenance does not match",
    ):
        load_empirical_psf(resolved)


def test_load_empirical_psf_rejects_stale_psf_fits(tmp_path):
    """The metadata must bind the exact stored PSF FITS product."""
    resolved, _, _ = _save_valid_empirical_psf_bundle(
        tmp_path
    )
    psf_path = pipeline_paths(resolved)["psf"]["psf_fits"]

    with fits.open(psf_path, mode="update") as hdul:
        hdul[0].header["STALE"] = True

    with pytest.raises(
        ValueError,
        match="provenance does not match",
    ):
        load_empirical_psf(resolved)


def test_load_empirical_psf_rejects_changed_centroid_tolerance(tmp_path):
    """A different centering policy must invalidate an existing PSF bundle."""
    resolved, _, _ = _save_valid_empirical_psf_bundle(
        tmp_path
    )
    resolved["psf"]["empirical"]["construction"][
        "centroid_tolerance_pixels"
    ] = 0.5

    with pytest.raises(
        ValueError,
        match="validation metadata does not match",
    ):
        load_empirical_psf(resolved)


def test_load_empirical_psf_requires_metadata(tmp_path):
    """Inference must reject an empirical PSF without its sidecar."""
    resolved, _, _ = _save_valid_empirical_psf_bundle(
        tmp_path
    )
    metadata_path = pipeline_paths(resolved)["psf"]["metadata"]
    metadata_path.unlink()

    with pytest.raises(
        FileNotFoundError,
        match="provenance metadata is required",
    ):
        load_empirical_psf(resolved)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 0),
        ("type", "gaussian"),
    ],
)
def test_load_empirical_psf_rejects_wrong_metadata_contract(
    tmp_path,
    field,
    value,
):
    """Inference must reject an unsupported metadata schema or type."""
    resolved, _, _ = _save_valid_empirical_psf_bundle(
        tmp_path
    )
    metadata_path = pipeline_paths(resolved)["psf"]["metadata"]
    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )
    metadata[field] = value
    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="schema/type is unsupported or missing",
    ):
        load_empirical_psf(resolved)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("validation", "validation metadata is missing"),
        ("provenance", "provenance metadata is missing"),
    ],
)
def test_load_empirical_psf_rejects_incomplete_metadata(
    tmp_path,
    field,
    message,
):
    """Legacy sidecars must fail closed instead of reaching inference."""
    resolved, _, _ = _save_valid_empirical_psf_bundle(
        tmp_path
    )
    metadata_path = pipeline_paths(resolved)["psf"]["metadata"]
    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )
    del metadata[field]
    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_empirical_psf(resolved)


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
    cutouts = []

    for amp in (1.0, 2.0, 3.0):
        cutout = np.ones((5, 5)) * 0.01
        cutout[2, 2] = amp
        cutouts.append(cutout)

    psf = stack_psf_cutouts(cutouts)

    assert psf.shape == (5, 5)
    assert np.isclose(np.sum(psf), 1.0)


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


def test_save_empirical_psf_metadata(tmp_path):
    config = {
        "object": {"name": "example_galaxy"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {
            "root": str(tmp_path / "outputs"),
        },
    }

    resolved = resolve_config(config)

    metadata = {
        "n_candidates_input": 2,
        "n_candidates_used": 2,
        "psf_shape": [9, 9],
        "psf_sum": 1.0,
    }

    path = save_empirical_psf_metadata(
        resolved,
        metadata,
    )

    assert path == pipeline_paths(resolved)["psf"]["metadata"]
    assert path.exists()

    with path.open("r") as f:
        loaded = json.load(f)

    assert loaded == metadata


def test_crop_center_returns_expected_shape_and_center():
    image = np.arange(100).reshape(10, 10)

    cropped = crop_center(image, half_size=2)

    assert cropped.shape == (5, 5)
    assert cropped[2, 2] == image[5, 5]


def test_crop_center_raises_if_too_large():
    image = np.zeros((5, 5))

    with pytest.raises(ValueError, match="outside the image boundaries"):
        crop_center(image, half_size=4)

def test_equivalent_gaussian_fwhm_recovers_sampled_gaussian_width():
    """Moment-based FWHM must recover the nominal Gaussian width."""
    import pytest

    from baymorph.pipeline.psf_construction import estimate_psf_equivalent_gaussian_fwhm
    from baymorph.simulation.psf import GaussianPSF

    for nominal_fwhm in (2.5, 4.0):
        psf = GaussianPSF(fwhm=nominal_fwhm).kernel()

        equivalent_gaussian_fwhm = estimate_psf_equivalent_gaussian_fwhm(psf)

        assert equivalent_gaussian_fwhm == pytest.approx(
            nominal_fwhm,
            abs=0.05,
        )


def test_equivalent_gaussian_fwhm_is_invariant_to_normalization():
    """Multiplicative PSF normalization must not change the width."""
    import pytest

    from baymorph.pipeline.psf_construction import estimate_psf_equivalent_gaussian_fwhm
    from baymorph.simulation.psf import GaussianPSF

    psf = GaussianPSF(fwhm=2.5).kernel()

    reference = estimate_psf_equivalent_gaussian_fwhm(psf)
    rescaled = estimate_psf_equivalent_gaussian_fwhm(100.0 * psf)

    assert rescaled == pytest.approx(
        reference,
        rel=1.0e-12,
        abs=1.0e-12,
    )


def test_equivalent_gaussian_fwhm_rejects_invalid_inputs():
    """Invalid or degenerate PSF arrays must return ``None``."""
    import numpy as np

    from baymorph.pipeline.psf_construction import estimate_psf_equivalent_gaussian_fwhm

    assert estimate_psf_equivalent_gaussian_fwhm(np.ones(10)) is None
    assert estimate_psf_equivalent_gaussian_fwhm(np.zeros((9, 9))) is None
    assert estimate_psf_equivalent_gaussian_fwhm(
        np.full((9, 9), np.nan)
    ) is None



def test_empirical_psf_peak_snr_default_is_resolved():
    """Empirical PSF construction must use an explicit S/N quality threshold."""
    config = {
        "object": {"name": "example_galaxy"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": "outputs"},
    }

    resolved = resolve_config(config)

    assert (
        resolved["psf"]["empirical"]
        ["candidate_selection"]["min_peak_snr"]
        == 30.0
    )


def test_estimate_cutout_peak_snr_uses_robust_border_noise():
    """Peak S/N must be based on a robust local border-noise estimate."""
    from baymorph.pipeline.psf_construction import (
        estimate_cutout_peak_snr,
    )

    rng = np.random.default_rng(42)
    cutout = rng.normal(
        0.0,
        2.0,
        size=(25, 25),
    )
    cutout[12, 12] += 100.0

    peak_snr = estimate_cutout_peak_snr(
        cutout
    )

    assert peak_snr is not None
    assert 30.0 < peak_snr < 70.0


def test_psf_construction_rejects_low_snr_candidates(
    monkeypatch,
):
    """Construction must stack only candidates above the configured S/N."""
    from types import SimpleNamespace

    import baymorph.pipeline.psf_construction as module

    image = np.zeros(
        (31, 31),
        dtype=float,
    )
    image[10, 10] = 100.0
    image[20, 20] = 10.0

    candidates = [
        SimpleNamespace(
            label=1,
            x=10.0,
            y=10.0,
            half_size=3,
        ),
        SimpleNamespace(
            label=2,
            x=20.0,
            y=20.0,
            half_size=3,
        ),
    ]

    observation = SimpleNamespace(
        image=SimpleNamespace(
            data=image
        )
    )

    monkeypatch.setattr(
        module.Observation,
        "from_fits",
        lambda _: observation,
    )
    monkeypatch.setattr(
        module,
        "load_psf_candidates",
        lambda _: candidates,
    )
    monkeypatch.setattr(
        module,
        "estimate_cutout_peak_snr",
        lambda cutout: float(
            np.max(
                cutout
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "observation_metadata_dict",
        lambda _: {},
    )
    monkeypatch.setattr(
        module,
        "save_empirical_psf_diagnostics",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        module,
        "save_empirical_psf",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        module,
        "build_empirical_psf_provenance",
        lambda *_: {
            "schema_version": 1,
            "input_fits": {"sha256": "input"},
            "candidate_catalog": {"sha256": "candidates"},
            "psf_fits": {"sha256": "psf"},
            "empirical_config_sha256": "config",
        },
    )

    captured = {}

    def fake_stack(cutouts):
        captured["n_cutouts"] = len(
            cutouts
        )
        psf = np.zeros(
            (7, 7),
            dtype=float,
        )
        psf[3, 3] = 1.0
        return psf

    def fake_save_metadata(
        config,
        metadata,
    ):
        captured["metadata"] = metadata

    monkeypatch.setattr(
        module,
        "stack_psf_cutouts",
        fake_stack,
    )
    monkeypatch.setattr(
        module,
        "save_empirical_psf_metadata",
        fake_save_metadata,
    )

    config = {
        "data": {
            "image_path": "unused.fits",
        },
        "psf": {
            "empirical": {
                "candidate_selection": {
                    "min_peak_snr": 30.0,
                },
            },
        },
    }

    module.run_psf_construction_from_config(
        config
    )

    assert captured["n_cutouts"] == 1

    metadata = captured[
        "metadata"
    ]

    assert metadata[
        "n_candidates_input"
    ] == 2
    assert metadata[
        "n_candidates_used"
    ] == 1
    assert metadata[
        "n_candidates_rejected_snr"
    ] == 1
    assert metadata[
        "used_candidates"
    ][0]["label"] == 1
    assert metadata[
        "rejected_candidates"
    ][0]["label"] == 2
    assert metadata[
        "rejected_candidates"
    ][0]["reason"] == (
        "peak_snr_below_threshold"
    )
    assert metadata["type"] == "empirical"
    assert metadata["validation"][
        "centroid_offset_pixels"
    ] == pytest.approx(0.0)
    assert metadata["provenance"][
        "empirical_config_sha256"
    ] == "config"


def test_psf_construction_fails_when_all_candidates_fail_snr(
    monkeypatch,
):
    """Construction must fail rather than create a PSF from invalid candidates."""
    from types import SimpleNamespace

    import baymorph.pipeline.psf_construction as module

    observation = SimpleNamespace(
        image=SimpleNamespace(
            data=np.zeros(
                (21, 21),
                dtype=float,
            )
        )
    )

    candidate = SimpleNamespace(
        label=1,
        x=10.0,
        y=10.0,
        half_size=3,
    )

    monkeypatch.setattr(
        module.Observation,
        "from_fits",
        lambda _: observation,
    )
    monkeypatch.setattr(
        module,
        "load_psf_candidates",
        lambda _: [candidate],
    )
    monkeypatch.setattr(
        module,
        "estimate_cutout_peak_snr",
        lambda cutout: 5.0,
    )

    config = {
        "data": {
            "image_path": "unused.fits",
        },
        "psf": {
            "empirical": {
                "candidate_selection": {
                    "min_peak_snr": 30.0,
                },
            },
        },
    }

    with pytest.raises(
        ValueError,
        match="No PSF candidates satisfy min_peak_snr=30",
    ):
        module.run_psf_construction_from_config(
            config
        )


def test_psf_core_fwhm_recovers_sampled_gaussian_width():
    """Sub-pixel core FWHM must recover sampled Gaussian widths."""
    from baymorph.pipeline.psf_construction import (
        estimate_psf_core_fwhm,
    )
    from baymorph.simulation.psf import GaussianPSF

    for nominal_fwhm in (
        2.5,
        4.0,
    ):
        psf = GaussianPSF(
            fwhm=nominal_fwhm
        ).kernel()

        core_fwhm = estimate_psf_core_fwhm(
            psf
        )

        assert core_fwhm == pytest.approx(
            nominal_fwhm,
            abs=0.05,
        )


def test_psf_core_fwhm_is_invariant_to_normalization():
    """Multiplicative normalization must not change the core FWHM."""
    from baymorph.pipeline.psf_construction import (
        estimate_psf_core_fwhm,
    )
    from baymorph.simulation.psf import GaussianPSF

    psf = GaussianPSF(
        fwhm=3.5
    ).kernel()

    reference = estimate_psf_core_fwhm(
        psf
    )

    rescaled = estimate_psf_core_fwhm(
        100.0 * psf
    )

    assert rescaled == pytest.approx(
        reference,
        rel=1.0e-12,
        abs=1.0e-12,
    )


def test_psf_core_fwhm_rejects_invalid_inputs():
    """Invalid or degenerate PSF arrays must not produce a core width."""
    from baymorph.pipeline.psf_construction import (
        estimate_psf_core_fwhm,
    )

    assert estimate_psf_core_fwhm(
        np.ones(10)
    ) is None

    assert estimate_psf_core_fwhm(
        np.zeros((9, 9))
    ) is None

    assert estimate_psf_core_fwhm(
        np.full(
            (9, 9),
            np.nan,
        )
    ) is None


def test_psf_core_fwhm_validates_sampling_parameters():
    """Core-FWHM sampling controls must reject invalid values."""
    from baymorph.pipeline.psf_construction import (
        estimate_psf_core_fwhm,
    )

    psf = np.ones(
        (9, 9)
    )

    with pytest.raises(
        ValueError,
        match="radial_step must be positive",
    ):
        estimate_psf_core_fwhm(
            psf,
            radial_step=0.0,
        )

    with pytest.raises(
        ValueError,
        match="n_angles must be at least 8",
    ):
        estimate_psf_core_fwhm(
            psf,
            n_angles=4,
        )
