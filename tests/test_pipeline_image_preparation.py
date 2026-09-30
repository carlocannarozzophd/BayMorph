import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

import baymorph.pipeline.image_preparation as image_preparation_module

from baymorph.pipeline import (
    PreparedObservation,
    load_image_preparation_products,
    make_target_cutout,
    pipeline_paths,
    prepare_real_fits_image,
    resolve_config,
    resolve_fixed_cutout,
    save_image_preparation_products,
)
from baymorph.pipeline.cutout import CutoutDefinition
from baymorph.pipeline.prepared_identity import build_prepared_input_identity
from baymorph.pipeline.preparation_steps import TargetGeometry


def test_make_target_cutout_returns_expected_shape():
    data = np.zeros((100, 100))

    cutout_definition = resolve_fixed_cutout(
        image_shape=data.shape,
        center_x=50,
        center_y=50,
        size=40,
    )

    cutout = make_target_cutout(
        data,
        cutout_definition,
    )

    assert cutout.shape == (40, 40)


def test_make_target_cutout_shifts_cutout_inside_image():
    data = np.zeros((100, 100))

    cutout_definition = resolve_fixed_cutout(
        image_shape=data.shape,
        center_x=5,
        center_y=5,
        size=40,
    )

    cutout = make_target_cutout(
        data,
        cutout_definition,
    )

    assert cutout.shape == (40, 40)
    assert cutout_definition.xmin == 0
    assert cutout_definition.xmax == 40
    assert cutout_definition.ymin == 0
    assert cutout_definition.ymax == 40


def _preparation_bundle(tmp_path, *, persist=True):
    """Build one source-consistent exact-mask preparation bundle."""
    image_path = tmp_path / "image.fits"
    image = np.arange(100, dtype=float).reshape(10, 10)
    fits.writeto(image_path, image)

    curated_mask = np.zeros((10, 10), dtype=bool)
    curated_mask[0, 0] = True
    curated_mask[2, 3] = True
    curated_mask_path = tmp_path / "curated_mask.fits"
    fits.writeto(curated_mask_path, curated_mask.astype(np.uint8))

    config = {
        "object": {"name": "example_galaxy"},
        "data": {
            "image_path": str(image_path),
            "mask_path": str(curated_mask_path),
            "mask_mode": "exact",
            "target_position": {"x": 5, "y": 5},
            "cutout": {"mode": "fixed", "size": 10},
        },
        "outputs": {"root": str(tmp_path / "outputs"), "overwrite": True},
    }
    resolved = resolve_config(config)
    paths = pipeline_paths(resolved)["image_preparation"]
    paths["dir"].mkdir(parents=True, exist_ok=True)

    products = PreparedObservation(
        cutout=image.copy(),
        prepared_data=image - 1.5,
        noise=np.ones((10, 10)) * 3,
        automatic_mask=np.zeros((10, 10), dtype=bool),
        fitting_mask=curated_mask.copy(),
        external_mask=curated_mask.copy(),
        fitting_mask_mode="exact",
        target_segment=~curated_mask,
        contaminant_mask=np.zeros((10, 10), dtype=bool),
        segmentation=np.ones((10, 10), dtype=int),
        background=1.5,
        target_x=5,
        target_y=6,
        target_label=1,
        original_image=image.copy(),
        cutout_bounds={
            "xmin": 0,
            "xmax": 10,
            "ymin": 0,
            "ymax": 10,
        },
    )

    if persist:
        save_image_preparation_products(resolved, products)

    return resolved, paths, products, image_path


def _preparation_control_paths(paths):
    """Return preparation manifest and completion-marker paths."""
    directory = paths["dir"]
    return (
        directory / image_preparation_module._PREPARATION_MANIFEST_NAME,
        directory / image_preparation_module._PREPARATION_COMPLETE_NAME,
    )


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one test artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resign_preparation_artifacts(paths, changed_paths) -> None:
    """Update records so semantic validators receive tampered data."""
    manifest_path, completion_path = _preparation_control_paths(paths)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for changed_path in changed_paths:
        relative = changed_path.relative_to(paths["dir"]).as_posix()
        manifest["artifacts"][relative] = {
            "size_bytes": changed_path.stat().st_size,
            "sha256": _sha256(changed_path),
        }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["manifest_sha256"] = _sha256(manifest_path)
    completion_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_save_and_load_image_preparation_products_roundtrip(tmp_path):
    resolved, paths, products, _ = _preparation_bundle(tmp_path)

    loaded = load_image_preparation_products(resolved)

    assert np.allclose(loaded.cutout, products.cutout)
    assert np.allclose(loaded.prepared_data, products.prepared_data)
    assert np.allclose(loaded.noise, products.noise)
    assert np.array_equal(loaded.contaminant_mask, products.contaminant_mask)
    assert np.array_equal(loaded.fitting_mask, products.fitting_mask)
    assert loaded.fitting_mask_mode == "exact"
    assert np.array_equal(
        fits.getdata(paths["fitting_mask"]).astype(bool),
        products.fitting_mask,
    )
    assert np.array_equal(loaded.external_mask, products.external_mask)
    assert np.array_equal(loaded.target_segment, products.target_segment)
    assert np.array_equal(loaded.segmentation, products.segmentation)
    assert loaded.background == products.background
    assert loaded.target_x == products.target_x
    assert loaded.target_y == products.target_y
    assert loaded.target_label == products.target_label


def test_save_rejects_source_mutation_during_bundle_write(
    tmp_path,
    monkeypatch,
):
    resolved, paths, products, image_path = _preparation_bundle(
        tmp_path,
        persist=False,
    )
    input_identity = build_prepared_input_identity(resolved)
    original_writeto = image_preparation_module.fits.writeto
    source_mutated = False

    def writeto_then_mutate_source(filename, *args, **kwargs):
        nonlocal source_mutated
        original_writeto(filename, *args, **kwargs)
        if Path(filename).name == "prepared_image.fits" and not source_mutated:
            source_mutated = True
            original_writeto(
                image_path,
                np.full((10, 10), 42.0),
                overwrite=True,
            )

    monkeypatch.setattr(
        image_preparation_module.fits,
        "writeto",
        writeto_then_mutate_source,
    )

    with pytest.raises(RuntimeError, match="immutable inputs"):
        save_image_preparation_products(
            resolved,
            products,
            input_identity=input_identity,
        )

    _, completion_path = _preparation_control_paths(paths)
    assert source_mutated
    assert not completion_path.exists()


def test_load_rejects_missing_fitting_mask_artifact(tmp_path):
    resolved, paths, _, _ = _preparation_bundle(tmp_path)
    assert (paths["dir"] / "fitting_mask.npy").is_file()
    paths["fitting_mask"].unlink()

    with pytest.raises(RuntimeError, match="artifact integrity check failed"):
        load_image_preparation_products(resolved)


def test_load_rejects_resigned_semantically_invalid_fitting_mask(tmp_path):
    resolved, paths, _, _ = _preparation_bundle(tmp_path)
    invalid_mask = np.zeros((10, 10), dtype=bool)
    fits.writeto(
        paths["fitting_mask"],
        invalid_mask.astype(np.uint8),
        overwrite=True,
    )

    changed_paths = [paths["fitting_mask"]]
    duplicate_path = paths["dir"] / "fitting_mask.npy"
    if duplicate_path.exists():
        np.save(duplicate_path, invalid_mask)
        changed_paths.append(duplicate_path)

    _resign_preparation_artifacts(paths, changed_paths)

    with pytest.raises(RuntimeError, match="fitting mask does not match"):
        load_image_preparation_products(resolved)


def test_load_rejects_resigned_tampered_cutout_bounds(tmp_path):
    resolved, paths, _, _ = _preparation_bundle(tmp_path)
    metadata_path = paths["dir"] / "image_preparation_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["cutout_bounds"]["xmax"] = 9
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _resign_preparation_artifacts(paths, [metadata_path])

    with pytest.raises(RuntimeError, match="cutout bounds"):
        load_image_preparation_products(resolved)


def test_load_rejects_tampered_prepared_image_artifact(tmp_path):
    resolved, paths, _, _ = _preparation_bundle(tmp_path)
    fits.writeto(
        paths["prepared_image"],
        np.full((10, 10), 17.0),
        overwrite=True,
    )

    with pytest.raises(RuntimeError, match="artifact integrity check failed"):
        load_image_preparation_products(resolved)


def test_load_rejects_missing_preparation_completion_marker(tmp_path):
    resolved, paths, _, _ = _preparation_bundle(tmp_path)
    _, completion_path = _preparation_control_paths(paths)
    completion_path.unlink()

    with pytest.raises(RuntimeError, match="bundle is incomplete"):
        load_image_preparation_products(resolved)


def test_load_rejects_tampered_preparation_completion_marker(tmp_path):
    resolved, paths, _, _ = _preparation_bundle(tmp_path)
    _, completion_path = _preparation_control_paths(paths)
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["manifest_sha256"] = "0" * 64
    completion_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="completion marker"):
        load_image_preparation_products(resolved)


def test_save_products_rejects_invalid_mask_before_first_write(tmp_path):
    config = resolve_config(
        {
            "object": {"name": "invalid_mask"},
            "data": {"image_path": str(tmp_path / "input.fits")},
            "outputs": {
                "root": str(tmp_path / "outputs"),
                "overwrite": True,
            },
        }
    )

    invalid_fitting_mask = np.zeros((10, 10), dtype=float)
    invalid_fitting_mask[2, 3] = np.nan

    products = PreparedObservation(
        cutout=np.ones((10, 10)),
        prepared_data=np.ones((10, 10)),
        noise=np.ones((10, 10)),
        automatic_mask=np.zeros((10, 10), dtype=bool),
        fitting_mask=invalid_fitting_mask,
        target_segment=np.eye(10, dtype=bool),
        contaminant_mask=np.zeros((10, 10), dtype=bool),
        segmentation=np.zeros((10, 10), dtype=int),
        background=0.0,
        target_x=5,
        target_y=5,
        target_label=1,
    )

    with pytest.raises(ValueError, match="non-finite"):
        save_image_preparation_products(config, products)

    assert not pipeline_paths(config)["image_preparation"][
        "prepared_image"
    ].exists()


def test_prepared_observation_alias():
    """The legacy public name must remain available during v0.4."""



@pytest.mark.parametrize(
    "mask_mode",
    ["union", "exact", "replace"],
)
def test_prepare_real_image_from_config_applies_mask_mode(
    tmp_path,
    mask_mode,
    monkeypatch,
):
    from baymorph.pipeline import prepare_real_image_from_config

    yy, xx = np.indices((80, 80))
    radius_squared = (xx - 40.0) ** 2 + (yy - 40.0) ** 2
    image = 100.0 * np.exp(-0.5 * radius_squared / 4.0**2)

    external_mask = np.zeros_like(image, dtype=np.uint8)
    external_mask[35:38, 30:33] = 1

    image_path = tmp_path / "image.fits"
    external_mask_path = tmp_path / "external_mask.fits"
    fits.writeto(image_path, image)
    fits.writeto(external_mask_path, external_mask)

    config = {
        "object": {"name": "synthetic"},
        "data": {
            "image_path": str(image_path),
            "mask_path": str(external_mask_path),
            "mask_mode": mask_mode,
            "target_position": {"x": 40, "y": 40},
            "cutout": {"mode": "fixed", "size": 70},
        },
        "preparation": {
            "background": {"box_size": 10, "filter_size": 3},
            "detection": {
                "n_sigma": 2.0,
                "n_pixels": 5,
                "n_levels": 32,
            },
        },
        "outputs": {
            "root": str(tmp_path / "outputs"),
            "overwrite": True,
        },
    }

    monkeypatch.setattr(
        image_preparation_module,
        "validate_plateau_cutout_containment",
        lambda **kwargs: None,
    )

    products = prepare_real_image_from_config(config, save=False)

    expected_external_mask = external_mask[
        products.cutout_bounds["ymin"]:products.cutout_bounds["ymax"],
        products.cutout_bounds["xmin"]:products.cutout_bounds["xmax"],
    ].astype(bool)

    assert products.external_mask is not None
    assert np.array_equal(products.external_mask, expected_external_mask)
    assert products.fitting_mask_mode == mask_mode

    if mask_mode == "union":
        assert np.all(products.fitting_mask[products.external_mask])
    else:
        assert np.array_equal(
            products.fitting_mask,
            expected_external_mask,
        )


def test_preparation_does_not_estimate_a_discarded_mesh_background(
    tmp_path,
    monkeypatch,
):
    """Preparation needs source detection only, never a mesh background."""
    from baymorph.pipeline import prepare_real_image_from_config

    yy, xx = np.indices((80, 80))
    radius_squared = (xx - 40.0) ** 2 + (yy - 40.0) ** 2
    image_path = tmp_path / "image.fits"
    fits.writeto(
        image_path,
        100.0 * np.exp(-0.5 * radius_squared / 4.0**2),
    )

    def forbidden_mesh_background(*args, **kwargs):
        raise AssertionError(
            "Image preparation must not estimate a mesh background."
        )

    monkeypatch.setattr(
        image_preparation_module,
        "estimate_background_photutils",
        forbidden_mesh_background,
    )
    monkeypatch.setattr(
        image_preparation_module,
        "validate_plateau_cutout_containment",
        lambda **kwargs: None,
    )

    products = prepare_real_image_from_config(
        {
            "object": {"name": "synthetic"},
            "data": {
                "image_path": str(image_path),
                "target_position": {"x": 40, "y": 40},
                "cutout": {"mode": "fixed", "size": 70},
            },
            "preparation": {
                "detection": {
                    "n_sigma": 2.0,
                    "n_pixels": 5,
                    "n_levels": 32,
                },
            },
            "outputs": {
                "root": str(tmp_path / "outputs"),
                "overwrite": True,
            },
        },
        save=False,
    )

    assert products.fitting_mask.shape == (70, 70)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_prepare_real_fits_image_rejects_non_finite_science_data(
    invalid_value,
):
    image = np.zeros((20, 20), dtype=float)
    image[3, 4] = invalid_value

    with pytest.raises(ValueError, match="non-finite pixels"):
        prepare_real_fits_image(image)


def test_auto_cutout_exact_mask_does_not_relabel_contaminants(
    monkeypatch,
):
    """Exact fitting authority must not redefine target ownership."""
    image = np.zeros(
        (20, 20),
        dtype=float,
    )

    target_segment = np.zeros_like(
        image,
        dtype=bool,
    )
    target_segment[9:12, 9:12] = True

    automatic_mask = np.zeros_like(
        image,
        dtype=bool,
    )
    automatic_mask[2, 2] = True
    automatic_mask[10, 12] = True

    curated_mask = np.zeros_like(
        image,
        dtype=bool,
    )
    curated_mask[2, 2] = True
    curated_mask[10, 10] = True

    segmentation = SimpleNamespace(
        data=np.zeros_like(
            image,
            dtype=int,
        )
    )

    detection = SimpleNamespace(
        target_segment=target_segment,
        contaminant_mask=automatic_mask,
        detection=SimpleNamespace(
            segmentation=segmentation
        ),
    )

    definition = CutoutDefinition(
        xmin=0,
        xmax=20,
        ymin=0,
        ymax=20,
        width=20,
        height=20,
        mode="auto",
    )

    monkeypatch.setattr(
        image_preparation_module,
        "_detect_target_in_image",
        lambda *args, **kwargs: detection,
    )

    preliminary_segments_seen = []

    def fake_resolve_preliminary_auto_cutout(
        **kwargs,
    ):
        preliminary_segments_seen.append(
            np.array(
                kwargs[
                    "target_segment"
                ],
                copy=True,
            )
        )

        return definition

    monkeypatch.setattr(
        image_preparation_module,
        "resolve_preliminary_auto_cutout",
        fake_resolve_preliminary_auto_cutout,
    )

    monkeypatch.setattr(
        image_preparation_module,
        "resolve_plateau_cutout",
        lambda **kwargs: definition,
    )

    masks_seen = []
    segments_seen = []

    def fake_estimate_target_sky(
        cutout,
        segment,
        *,
        mask,
        **kwargs,
    ):
        masks_seen.append(
            np.array(
                mask,
                copy=True,
            )
        )

        segments_seen.append(
            np.array(
                segment,
                copy=True,
            )
        )

        return (
            TargetGeometry(
                x0=10.0,
                y0=10.0,
                q=1.0,
                theta=0.0,
            ),
            SimpleNamespace(),
            SimpleNamespace(
                radius=5.0
            ),
        )

    monkeypatch.setattr(
        image_preparation_module,
        "estimate_target_sky",
        fake_estimate_target_sky,
    )

    image_preparation_module._resolve_auto_cutout(
        image,
        target_x=10.0,
        target_y=10.0,
        cutout_cfg={
            "min_size": 10,
            "sky_margin": 2.0,
        },
        detection_cfg={},
        external_mask=curated_mask,
        fitting_mask_mode="exact",
    )

    assert len(
        masks_seen
    ) == 1

    assert not preliminary_segments_seen[
        0
    ][2, 2]

    assert not preliminary_segments_seen[
        0
    ][10, 12]

    assert not preliminary_segments_seen[
        0
    ][10, 10]

    assert np.array_equal(
        masks_seen[0],
        curated_mask,
    )

    assert masks_seen[
        0
    ][2, 2]

    assert not masks_seen[
        0
    ][10, 12]

    assert not segments_seen[
        0
    ][2, 2]

    assert not segments_seen[
        0
    ][10, 12]

    assert not segments_seen[
        0
    ][10, 10]


def test_exact_geometry_preserves_detected_target_ownership():
    """Unmasked contaminants must not become target support."""
    target = np.zeros(
        (12, 12),
        dtype=bool,
    )
    target[5:7, 5:7] = True

    automatic_contaminants = np.zeros_like(
        target
    )
    automatic_contaminants[6, 7] = True
    automatic_contaminants[1, 1] = True

    curated_mask = np.zeros_like(
        target
    )
    curated_mask[5, 5] = True

    result = (
        image_preparation_module
        ._build_geometry_target_segment(
            target,
            curated_mask,
            contaminant_mask=automatic_contaminants,
            fitting_mask_mode="exact",
        )
    )

    assert not result[5, 5]
    assert result[5, 6]
    assert result[6, 5]
    assert result[6, 6]
    assert not result[6, 7]
    assert not result[1, 1]
    assert np.count_nonzero(
        result
    ) == 3

def test_prepare_real_image_rejects_final_plateau_outside_auto_cutout(
    tmp_path,
    monkeypatch,
):
    image_path = tmp_path / "image.fits"
    fits.writeto(image_path, np.zeros((40, 40), dtype=float))

    resolved = resolve_config(
        {
            "object": {"name": "truncated_auto_cutout"},
            "data": {
                "image_path": str(image_path),
                "target_position": {"x": 20, "y": 20},
                "cutout": {
                    "mode": "auto",
                    "min_size": 20,
                    "max_size": 20,
                    "sky_margin": 2.0,
                },
            },
            "outputs": {
                "root": str(tmp_path / "outputs"),
                "overwrite": True,
            },
        }
    )

    definition = CutoutDefinition(
        xmin=10,
        xmax=30,
        ymin=10,
        ymax=30,
        width=20,
        height=20,
        mode="auto",
    )
    monkeypatch.setattr(
        image_preparation_module,
        "_resolve_auto_cutout",
        lambda *args, **kwargs: SimpleNamespace(
            cutout=definition,
            preliminary_cutout=np.zeros((20, 20), dtype=float),
            preliminary_segmentation=np.ones((20, 20), dtype=int),
        ),
    )

    segmentation = SimpleNamespace(data=np.ones((20, 20), dtype=int))
    detection = SimpleNamespace(
        target_label=1,
        target_segment=np.ones((20, 20), dtype=bool),
        contaminant_mask=np.zeros((20, 20), dtype=bool),
        detection=SimpleNamespace(
            mask=np.zeros((20, 20), dtype=bool),
            segmentation=segmentation,
        ),
    )
    monkeypatch.setattr(
        image_preparation_module,
        "_detect_target_in_image",
        lambda *args, **kwargs: detection,
    )
    monkeypatch.setattr(
        image_preparation_module,
        "estimate_target_sky",
        lambda *args, **kwargs: (
            TargetGeometry(x0=9.5, y0=9.5, q=1.0, theta=0.0),
            SimpleNamespace(),
            SimpleNamespace(radius=9.0),
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="does not contain the final plateau radius",
    ):
        image_preparation_module.prepare_real_image_from_config(
            resolved,
            save=False,
        )


def test_prepare_real_image_rejects_final_plateau_outside_fixed_cutout(
    tmp_path,
    monkeypatch,
):
    image_path = tmp_path / "image.fits"
    fits.writeto(image_path, np.zeros((40, 40), dtype=float))

    resolved = resolve_config(
        {
            "object": {"name": "truncated_fixed_cutout"},
            "data": {
                "image_path": str(image_path),
                "target_position": {"x": 20, "y": 20},
                "cutout": {
                    "mode": "fixed",
                    "size": 20,
                    "sky_margin": 2.0,
                },
            },
            "outputs": {
                "root": str(tmp_path / "outputs"),
                "overwrite": True,
            },
        }
    )

    segmentation = SimpleNamespace(data=np.ones((20, 20), dtype=int))
    detection = SimpleNamespace(
        target_label=1,
        target_segment=np.ones((20, 20), dtype=bool),
        contaminant_mask=np.zeros((20, 20), dtype=bool),
        detection=SimpleNamespace(
            mask=np.zeros((20, 20), dtype=bool),
            segmentation=segmentation,
        ),
    )
    monkeypatch.setattr(
        image_preparation_module,
        "_detect_target_in_image",
        lambda *args, **kwargs: detection,
    )
    monkeypatch.setattr(
        image_preparation_module,
        "estimate_target_sky",
        lambda *args, **kwargs: (
            TargetGeometry(x0=9.5, y0=9.5, q=1.0, theta=0.0),
            SimpleNamespace(),
            SimpleNamespace(radius=9.0),
        ),
    )

    def fail_if_sky_region_is_reached(*args, **kwargs):
        raise AssertionError(
            "Fixed-cutout containment must be checked before sky-region "
            "construction."
        )

    monkeypatch.setattr(
        image_preparation_module,
        "build_sky_region",
        fail_if_sky_region_is_reached,
    )

    with pytest.raises(
        RuntimeError,
        match="does not contain the final plateau radius",
    ):
        image_preparation_module.prepare_real_image_from_config(
            resolved,
            save=False,
        )
