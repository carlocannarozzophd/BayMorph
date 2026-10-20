"""Tests for the working region of a large image (``data.region``)."""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from astropy.io import fits

import baymorph.pipeline.image_preparation as image_preparation
import baymorph.pipeline.opening as opening
import baymorph.pipeline.prepared_identity as prepared_identity
import baymorph.pipeline.psf_construction as psf_construction
from baymorph import ObservationAdvisor
from baymorph.cli import main
from baymorph.core.image import ImageData, WorkingRegion
from baymorph.pipeline import (
    load_image_preparation_products,
    pipeline_paths,
    prepare_real_image_from_config,
    resolve_config,
)
from baymorph.pipeline.config import validate_config
from baymorph.pipeline.config_maker import make_config_from_fits
from baymorph.pipeline.cutout import CutoutDefinition
from baymorph.pipeline.model_screening_advisor import SingleObservationModelAdvisor
from baymorph.pipeline.opening import (
    RegionRequest,
    open_configured_observation,
    open_observation,
    validate_region,
    working_region,
)
from baymorph.pipeline.preparation_steps import TargetGeometry
from baymorph.pipeline.prepared_identity import build_prepared_input_identity
from baymorph.pipeline.psf_candidates import (
    build_psf_candidates_provenance,
    load_psf_candidates,
    run_psf_candidate_selection_from_config,
)
from baymorph.plotting.images import plot_pipeline_fit_maps


@pytest.fixture(autouse=True)
def _fresh_openings():
    """Start and end every test without observations kept for reuse."""
    opening.forget_opened_observations()
    yield
    opening.forget_opened_observations()


def _metadata(width: int, height: int, *, pixel_scale: float | None = 0.1):
    """Return the metadata that the resolution of a region reads."""
    return SimpleNamespace(
        width=width,
        height=height,
        center_x=(width - 1) / 2.0,
        center_y=(height - 1) / 2.0,
        pixel_scale=pixel_scale,
    )


def _wcs_header(scale: float = 0.1) -> fits.Header:
    """Return a tangent-plane header of ``scale`` arcsec per pixel."""
    header = fits.Header()

    for key, value in (
        ("CTYPE1", "RA---TAN"),
        ("CTYPE2", "DEC--TAN"),
        ("CRPIX1", 50.0),
        ("CRPIX2", 40.0),
        ("CRVAL1", 150.0),
        ("CRVAL2", 2.0),
        ("CD1_1", -scale / 3600.0),
        ("CD1_2", 0.0),
        ("CD2_1", 0.0),
        ("CD2_2", scale / 3600.0),
    ):
        header[key] = value

    return header


# ---------------------------------------------------------------- the rules


def test_a_region_is_taken_about_a_position_and_shifted_inside_the_image():
    """The region is centred where it can be, shifted inside, never padded."""
    centred = WorkingRegion.about((300, 500), center_x=250.4, center_y=149.6, side=100)
    edge = WorkingRegion.about((300, 500), center_x=480.2, center_y=20.7, side=100)
    narrow = WorkingRegion.about((3000, 200), center_x=50.0, center_y=2000.0, side=1000)

    assert centred.as_dict() == {"xmin": 200, "xmax": 300, "ymin": 100, "ymax": 200}
    assert edge.as_dict() == {"xmin": 400, "xmax": 500, "ymin": 0, "ymax": 100}
    assert narrow.as_dict() == {"xmin": 0, "xmax": 200, "ymin": 1500, "ymax": 2500}
    assert narrow.shape == (1000, 200)
    assert narrow.image_shape == (3000, 200)
    assert not narrow.is_whole

    with pytest.raises(ValueError, match="outside the image"):
        WorkingRegion.about((300, 500), center_x=600.0, center_y=10.0, side=100)

    with pytest.raises(ValueError, match="does not lie inside"):
        WorkingRegion(0, 600, 0, 10, image_width=500, image_height=300)


def test_a_region_translates_boxes_of_the_whole_image():
    """Boxes of the whole image are found in the array of the region."""
    region = WorkingRegion(100, 300, 50, 150, image_width=500, image_height=300)

    assert region.contains(xmin=100, xmax=300, ymin=50, ymax=150)
    assert not region.contains(xmin=99, xmax=200, ymin=60, ymax=70)
    assert region.local_slices(xmin=120, xmax=130, ymin=60, ymax=65) == (
        slice(10, 15),
        slice(20, 30),
    )

    with pytest.raises(ValueError, match="does not lie in the working region"):
        region.local_slices(xmin=290, xmax=310, ymin=60, ymax=65)


def test_the_default_reads_the_whole_image_up_to_4096_pixels_on_a_side():
    """Only an image with a side beyond 4096 pixels gets a region."""
    request = RegionRequest.of(None, target={"x": 5000.0, "y": 3.0})

    assert request.resolve(_metadata(4096, 4096)) is None

    region = request.resolve(_metadata(8000, 10))

    assert region.as_dict() == {"xmin": 2952, "xmax": 7048, "ymin": 0, "ymax": 10}


def test_whole_and_a_side_in_arcsec_set_the_region():
    """``whole`` reads everything; a side in arcsec sets the square."""
    target = {"x": 250.0, "y": 150.0}

    assert (
        RegionRequest.of("whole", target=target).resolve(_metadata(9000, 9000)) is None
    )

    region = RegionRequest.of({"size": 5}, target=target).resolve(
        _metadata(500, 300, pixel_scale=0.05)
    )
    larger = RegionRequest.of({"size": 60}, target=target).resolve(
        _metadata(500, 300, pixel_scale=0.05)
    )
    declared = RegionRequest.of({"size": 5}, target=target, pixel_scale=0.1).resolve(
        _metadata(500, 300, pixel_scale=0.05)
    )

    assert region.shape == (100, 100)
    assert larger is None
    assert declared.shape == (50, 50)


def test_a_region_needs_a_target_and_a_side_in_arcsec_a_pixel_scale():
    """The region stops when it cannot be placed or measured."""
    with pytest.raises(ValueError, match="no target position is set"):
        RegionRequest.of(None).resolve(_metadata(5000, 5000))

    with pytest.raises(ValueError, match="declare data.pixel_scale"):
        RegionRequest.of({"size": 5}, target="center").resolve(
            _metadata(500, 300, pixel_scale=None)
        )

    center = RegionRequest.of({"size": 5}, target="center").resolve(
        _metadata(500, 300, pixel_scale=0.05)
    )

    assert center.as_dict() == {"xmin": 200, "xmax": 300, "ymin": 100, "ymax": 200}


@pytest.mark.parametrize(
    "value",
    [
        "half",
        4096,
        {"size": 0},
        {"size": -3.0},
        {"size": float("nan")},
        {"size": True},
        {"side": 3},
        {"size": 3, "unit": "arcsec"},
    ],
)
def test_an_invalid_region_is_refused(tmp_path, value):
    """``data.region`` is null, ``whole`` or a positive side in arcsec."""
    with pytest.raises(ValueError, match="data.region must be"):
        validate_region(value)

    with pytest.raises(ValueError, match="data.region must be"):
        validate_config(
            {
                "object": {"name": "x"},
                "data": {"image_path": "x.fits", "region": value},
                "outputs": {"root": str(tmp_path)},
            }
        )


def test_an_unknown_entry_of_the_data_section_is_refused(tmp_path):
    """A misspelt key stops the configuration instead of being ignored."""
    with pytest.raises(ValueError, match=r"section 'data': \['regoin'\]"):
        validate_config(
            {
                "object": {"name": "x"},
                "data": {"image_path": "x.fits", "regoin": "whole"},
                "outputs": {"root": str(tmp_path)},
            }
        )

    assert validate_region(" Whole ") == "whole"
    assert validate_region({"size": 30}) == {"size": 30.0}


# ---------------------------------------------------------------- reading


def _write_kinds(directory: Path) -> None:
    """Write one image as plain, gzip, tile-compressed and scaled FITS files."""
    rng = np.random.default_rng(20261020)
    data = rng.normal(size=(120, 160)).astype(np.float32)
    scaled = np.round(rng.normal(0.0, 300.0, size=(120, 160))).astype(np.int16)
    header = _wcs_header()

    fits.writeto(directory / "plain.fits", data, header=header)

    with open(directory / "plain.fits", "rb") as source:
        with gzip.open(directory / "plain.fits.gz", "wb") as target:
            shutil.copyfileobj(source, target)

    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.CompImageHDU(data, header=header, name="SCI", quantize_level=0),
        ]
    ).writeto(directory / "tiled.fits.fz")

    primary = fits.PrimaryHDU(scaled, header=header)
    primary.header["BSCALE"] = 0.5
    primary.header["BZERO"] = 10.0
    primary.writeto(directory / "scaled.fits")


@pytest.mark.parametrize(
    "kind", ["plain.fits", "plain.fits.gz", "tiled.fits.fz", "scaled.fits"]
)
def test_a_region_holds_the_pixels_of_the_whole_image(tmp_path, kind):
    """The region equals the same pixels of the whole image, read-only."""
    _write_kinds(tmp_path)
    path = tmp_path / kind
    data = {
        "image_path": str(path),
        "target_position": {"x": 150.3, "y": 20.6},
        "region": {"size": 6.0},
    }

    observation = open_configured_observation(data)
    image = observation.image
    whole = open_observation(path).image.data

    assert image.region.as_dict() == {"xmin": 100, "xmax": 160, "ymin": 0, "ymax": 60}
    assert np.array_equal(image.data, whole[0:60, 100:160])
    assert image.full_shape == (120, 160)
    assert (observation.metadata.width, observation.metadata.height) == (160, 120)
    assert observation.metadata.center_x == pytest.approx(79.5)
    assert not image.data.flags.writeable
    assert working_region(data) == image.region

    with pytest.raises(ValueError, match="working region"):
        image.to_fits(tmp_path / "copy.fits")


def test_an_array_must_be_the_region_it_holds():
    """An image cannot claim a region of another shape."""
    region = WorkingRegion(0, 10, 0, 5, image_width=20, image_height=20)

    with pytest.raises(ValueError, match="is not the working region"):
        ImageData(data=np.zeros((5, 9)), region=region)


def test_the_region_is_found_in_the_reading_of_the_pixels(tmp_path, monkeypatch):
    """A large file is opened once, header and region together."""
    monkeypatch.setattr(opening, "REUSE_MIN_BYTES", 0)
    _write_kinds(tmp_path)
    path = tmp_path / "plain.fits"
    opened = []
    fits_open = fits.open

    def counting_open(name, *args, **kwargs):
        if Path(name).resolve() == path.resolve():
            opened.append(name)
        return fits_open(name, *args, **kwargs)

    monkeypatch.setattr(fits, "open", counting_open)
    data = {
        "image_path": str(path),
        "target_position": "center",
        "region": {"size": 6.0},
    }

    first = open_configured_observation(data)
    resolved = {**data, "target_position": {"x": 79.5, "y": 59.5}}

    assert working_region(resolved) == first.image.region
    assert open_configured_observation(resolved) is first
    assert len(opened) == 1


# ---------------------------------------------------------------- preparation


def _galaxy_image(path: Path) -> None:
    """Write a galaxy and a compact neighbour on a noisy sky, 200 x 260."""
    rng = np.random.default_rng(20261020)
    yy, xx = np.mgrid[0:200, 0:260]
    image = 100.0 + rng.normal(0.0, 2.0, size=(200, 260))
    image += 400.0 * np.exp(-0.5 * ((yy - 90.0) ** 2 + (xx - 170.0) ** 2) / 6.0**2)
    image += 300.0 * np.exp(-0.5 * ((yy - 52.0) ** 2 + (xx - 208.0) ** 2) / 1.5**2)

    fits.writeto(path, image)


def _preparation_config(tmp_path: Path, name: str, region, *, size: int = 100) -> dict:
    """Return the configuration of a fixed cutout about the galaxy.

    The object has one name, so the configurations differ only by the region
    and by where their products go.
    """
    return resolve_config(
        {
            "object": {"name": "galaxy"},
            "data": {
                "image_path": str(tmp_path / "image.fits"),
                "pixel_scale": 0.1,
                "region": region,
                "target_position": {"x": 170, "y": 90},
                "cutout": {"mode": "fixed", "size": size},
            },
            "outputs": {"root": str(tmp_path / name), "overwrite": False},
        }
    )


def test_a_cutout_inside_the_region_is_prepared_as_on_the_whole_image(tmp_path):
    """Reading a region changes no product of a fixed cutout inside it."""
    _galaxy_image(tmp_path / "image.fits")
    whole_config = _preparation_config(tmp_path, "whole", "whole")
    region_config = _preparation_config(tmp_path, "region", {"size": 13.0})

    whole = prepare_real_image_from_config(whole_config)
    region = prepare_real_image_from_config(region_config)
    bounds = region.working_region

    assert bounds == {"xmin": 105, "xmax": 235, "ymin": 25, "ymax": 155}
    assert whole.working_region is None
    assert region.cutout_bounds == whole.cutout_bounds

    for name in (
        "cutout",
        "prepared_data",
        "noise",
        "fitting_mask",
        "target_segment",
        "contaminant_mask",
        "segmentation",
    ):
        assert np.array_equal(getattr(region, name), getattr(whole, name)), name

    assert region.background == whole.background
    assert (region.target_x, region.target_y) == (whole.target_x, whole.target_y)
    assert region.geometry == whole.geometry
    assert np.array_equal(
        region.original_image,
        whole.original_image[
            bounds["ymin"] : bounds["ymax"], bounds["xmin"] : bounds["xmax"]
        ],
    )

    whole_identity = build_prepared_input_identity(whole_config)
    region_identity = build_prepared_input_identity(region_config)
    differing = {
        key
        for key in whole_identity["fits"].keys() | region_identity["fits"].keys()
        if whole_identity["fits"].get(key) != region_identity["fits"].get(key)
    }

    assert differing == {"region", "loaded_data_sha256"}
    assert region_identity["fits"]["region"] == bounds
    assert (
        region_identity["preparation_config_sha256"]
        == whole_identity["preparation_config_sha256"]
    )


def test_a_prepared_region_survives_the_round_trip(tmp_path):
    """The saved region is restored and checked when the products are loaded."""
    _galaxy_image(tmp_path / "image.fits")
    config = _preparation_config(tmp_path, "region", {"size": 13.0})
    prepared = prepare_real_image_from_config(config)
    metadata_path = (
        pipeline_paths(config)["image_preparation"]["dir"]
        / "image_preparation_metadata.json"
    )

    loaded = load_image_preparation_products(config)

    assert loaded.working_region == prepared.working_region
    assert json.loads(metadata_path.read_text())["working_region"] == (
        prepared.working_region
    )
    assert np.array_equal(loaded.original_image, prepared.original_image)

    other = _preparation_config(tmp_path, "region", {"size": 14.0})

    with pytest.raises(RuntimeError, match="not match"):
        load_image_preparation_products(other)


def test_a_cutout_beyond_the_region_stops_with_the_side_to_declare(tmp_path):
    """A cutout larger than the region asks for a larger region."""
    _galaxy_image(tmp_path / "image.fits")
    config = _preparation_config(tmp_path, "small", {"size": 8.0}, size=100)

    with pytest.raises(ValueError) as error:
        prepare_real_image_from_config(config, save=False)

    message = str(error.value)

    assert "The cutout, x 120-220 and y 40-140 of the image" in message
    assert "data.region: {size: 11} or more, or data.region: whole" in message


def test_a_target_that_reaches_an_inner_edge_of_the_region_stops():
    """Only an edge of the region inside the image can cut the target."""
    region = WorkingRegion(100, 140, 0, 30, image_width=200, image_height=100)
    inside = np.zeros(region.shape, dtype=bool)
    inside[10:20, 5:30] = True
    on_the_image_edge = np.zeros(region.shape, dtype=bool)
    on_the_image_edge[0:10, 5:30] = True
    on_an_inner_edge = np.zeros(region.shape, dtype=bool)
    on_an_inner_edge[10:20, 25:40] = True

    image_preparation._require_target_inside_region(inside, region, pixel_scale=0.1)
    image_preparation._require_target_inside_region(
        on_the_image_edge, region, pixel_scale=0.1
    )

    with pytest.raises(ValueError, match="reaches the edge of the region"):
        image_preparation._require_target_inside_region(
            on_an_inner_edge, region, pixel_scale=0.1
        )


def _fixed_sky(monkeypatch, *, radius: float) -> None:
    """Make the plateau of the preliminary cutout a given circle."""
    monkeypatch.setattr(
        image_preparation,
        "estimate_target_sky",
        lambda cutout, *args, **kwargs: (
            TargetGeometry(
                x0=(cutout.shape[1] - 1) / 2.0,
                y0=(cutout.shape[0] - 1) / 2.0,
                q=1.0,
                theta=0.0,
            ),
            SimpleNamespace(),
            SimpleNamespace(radius=radius),
        ),
    )


def _auto_cutout(tmp_path: Path, region):
    """Resolve the automatic cutout of the galaxy within a region."""
    _galaxy_image(tmp_path / "image.fits")
    config = _preparation_config(tmp_path, "auto", region)
    observation = open_configured_observation(config["data"])

    return image_preparation._resolve_auto_cutout(
        observation.image.data,
        target_x=170.0,
        target_y=90.0,
        cutout_cfg={"mode": "auto", "min_size": 60, "max_size": 150},
        detection_cfg=config["preparation"]["detection"],
        contaminant_rule="all_segments",
        region=observation.image.region,
        pixel_scale=0.1,
    )


def test_an_automatic_cutout_is_resolved_in_pixels_of_the_whole_image(
    tmp_path, monkeypatch
):
    """The automatic cutouts of a region are placed in the whole image.

    The plateau is a circle of 20 pixels about the centre of the preliminary
    cutout, which is centred on the target, so the final cutout spans 40
    pixels on either side of (170, 90) in the whole image.
    """
    _fixed_sky(monkeypatch, radius=20.0)

    final = _auto_cutout(tmp_path, {"size": 18.0}).cutout

    assert (final.xmin, final.xmax, final.ymin, final.ymax) == (130, 211, 50, 131)


def test_an_automatic_cutout_beyond_the_region_stops(tmp_path, monkeypatch):
    """An automatic cutout that leaves the region asks for a larger region."""
    _galaxy_image(tmp_path / "image.fits")
    config = _preparation_config(tmp_path, "auto", {"size": 13.0})
    config["data"]["cutout"] = {"mode": "auto"}
    definition = CutoutDefinition(
        xmin=100, xmax=200, ymin=40, ymax=140, width=100, height=100, mode="auto"
    )
    monkeypatch.setattr(
        image_preparation,
        "_resolve_auto_cutout",
        lambda *args, **kwargs: SimpleNamespace(cutout=definition),
    )

    with pytest.raises(
        ValueError, match="The automatic cutout, x 100-200 and y 40-140 of the image"
    ):
        prepare_real_image_from_config(config, save=False)


# ---------------------------------------------------------------- PSF


def _star_field(path: Path) -> None:
    """Write 49 isolated Gaussian stars of 1.5 pixels on a flat noisy sky."""
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:300, 0:300]
    image = 50.0 + rng.normal(0.0, 0.5, size=(300, 300))

    for index in range(49):
        row, column = divmod(index, 7)
        x = 30.0 + 40.0 * column + rng.uniform(-0.5, 0.5)
        y = 30.0 + 40.0 * row + rng.uniform(-0.5, 0.5)
        flux = 2000.0 * (1.0 + 0.1 * index)
        image += (
            flux
            / (2.0 * np.pi * 1.5**2)
            * np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / 1.5**2)
        )

    fits.writeto(path, image)


def _psf_config(tmp_path: Path, name: str, image: Path, region, target) -> dict:
    """Return a configuration whose PSF builder fits the star field."""
    return resolve_config(
        {
            "object": {"name": name},
            "data": {
                "image_path": str(image),
                "pixel_scale": 0.1,
                "region": region,
                "target_position": target,
                "cutout": {"mode": "fixed", "size": 50},
            },
            "psf": {
                "empirical": {
                    "candidate_selection": {
                        "min_area": 5,
                        "max_area": 400,
                        "min_peak": 5,
                        "max_peak": 1.0e6,
                        "min_peak_snr": 5.0,
                        "edge_margin": 12,
                        "galaxy_exclusion_box": {
                            "xmin": 145.0,
                            "xmax": 155.0,
                            "ymin": 145.0,
                            "ymax": 155.0,
                        },
                        "min_half_size": 4,
                        "max_half_size": 8,
                    },
                    "construction": {
                        "sky_inner_radius": 14.0,
                        "sky_outer_radius": 22.0,
                        "mask_growth_pixels": 2,
                        "support_half_size": 10,
                    },
                }
            },
            "outputs": {"root": str(tmp_path / name), "overwrite": False},
        }
    )


def test_the_psf_of_a_region_is_the_psf_of_a_file_holding_only_the_region(
    tmp_path, monkeypatch
):
    """Stars are found and built in the region and recorded in the whole image."""
    monkeypatch.setattr(
        psf_construction, "save_empirical_psf_diagnostics", lambda *a, **k: None
    )
    _star_field(tmp_path / "field.fits")
    region_config = _psf_config(
        tmp_path,
        "region",
        tmp_path / "field.fits",
        {"size": 20.0},
        {"x": 150.0, "y": 150.0},
    )
    region = working_region(region_config["data"])
    origin = np.array([region.xmin, region.ymin], dtype=float)
    fits.writeto(
        tmp_path / "crop.fits",
        fits.getdata(tmp_path / "field.fits")[
            region.ymin : region.ymax, region.xmin : region.xmax
        ],
    )
    crop_config = _psf_config(
        tmp_path,
        "crop",
        tmp_path / "crop.fits",
        None,
        {"x": 150.0 - region.xmin, "y": 150.0 - region.ymin},
    )
    box = crop_config["psf"]["empirical"]["candidate_selection"]["galaxy_exclusion_box"]
    crop_config["psf"]["empirical"]["candidate_selection"]["galaxy_exclusion_box"] = {
        "xmin": box["xmin"] - region.xmin,
        "xmax": box["xmax"] - region.xmin,
        "ymin": box["ymin"] - region.ymin,
        "ymax": box["ymax"] - region.ymin,
    }

    assert region.as_dict() == {"xmin": 50, "xmax": 250, "ymin": 50, "ymax": 250}

    in_region = run_psf_candidate_selection_from_config(region_config)
    in_crop = run_psf_candidate_selection_from_config(crop_config)

    assert len(in_region) == len(in_crop) >= 6
    assert np.allclose(
        [(c.x, c.y) for c in in_region],
        [(c.x + origin[0], c.y + origin[1]) for c in in_crop],
        rtol=0.0,
        atol=1.0e-9,
    )
    assert [(c.x, c.y) for c in load_psf_candidates(region_config)] == [
        (c.x, c.y) for c in in_region
    ]
    assert build_psf_candidates_provenance(region_config)["input_fits"]["region"] == (
        region.as_dict()
    )
    assert "region" not in build_psf_candidates_provenance(crop_config)["input_fits"]

    psf_region = psf_construction.run_psf_construction_from_config(region_config)
    psf_crop = psf_construction.run_psf_construction_from_config(crop_config)
    stars = {}

    for name, config in (("region", region_config), ("crop", crop_config)):
        metadata_path = pipeline_paths(config)["psf"]["metadata"]
        stars[name] = json.loads(Path(metadata_path).read_text())["stars"]

    assert np.allclose(psf_region, psf_crop, rtol=1.0e-9, atol=1.0e-12)
    assert len(stars["region"]) == len(stars["crop"])

    for star, crop_star in zip(stars["region"], stars["crop"], strict=True):
        for x, y in (("x", "y"), ("centroid_x", "centroid_y")):
            assert star[x] == pytest.approx(crop_star[x] + origin[0], abs=1.0e-9)
            assert star[y] == pytest.approx(crop_star[y] + origin[1], abs=1.0e-9)


# ---------------------------------------------------------------- interfaces


def _mosaic(path: Path) -> None:
    """Write a 300 x 400 image with a galaxy at (330, 70) and a WCS."""
    rng = np.random.default_rng(11)
    yy, xx = np.mgrid[0:300, 0:400]
    image = 10.0 + rng.normal(0.0, 1.0, size=(300, 400))
    image += 200.0 * np.exp(-0.5 * ((yy - 70.0) ** 2 + (xx - 330.0) ** 2) / 4.0**2)

    fits.writeto(path, image, header=_wcs_header(0.1))


def test_the_config_maker_writes_and_uses_the_region(tmp_path):
    """A declared region is written; the default writes nothing."""
    _mosaic(tmp_path / "mosaic.fits")
    options = {
        "target_position": {"x": 330.0, "y": 70.0},
        "psf_type": "gaussian",
        "gaussian_fwhm_pixels": 2.0,
        "cutout_mode": "fixed",
        "fixed_size": 40,
        "_require_complete_priors": False,
    }

    declared = make_config_from_fits(
        tmp_path / "mosaic.fits", region={"size": 6}, **options
    )
    default = make_config_from_fits(tmp_path / "mosaic.fits", **options)

    assert declared["data"]["region"] == {"size": 6.0}
    assert "region" not in default["data"]
    assert working_region(declared["data"]).as_dict() == {
        "xmin": 300,
        "xmax": 360,
        "ymin": 40,
        "ymax": 100,
    }


def test_the_advisor_inspects_the_region_about_its_target(tmp_path, monkeypatch):
    """The inspection reads the region and reports where it lies."""
    _mosaic(tmp_path / "mosaic.fits")
    advisor = ObservationAdvisor(
        tmp_path / "mosaic.fits",
        target_position={"x": 330.0, "y": 70.0},
        region={"size": 10},
        online=False,
    )

    inspection = advisor.inspect()
    summary = advisor.summary()

    assert inspection.shape == (300, 400)
    assert inspection.working_region == {
        "xmin": 280,
        "xmax": 380,
        "ymin": 20,
        "ymax": 120,
    }
    assert inspection.target_flux_scale is not None
    assert inspection.target_area_pixels > 20
    assert "within a working region of 100 x 100 pixels" in summary
    assert "Noise sigma (diagnostic of the working region)" in summary

    def refuse_pixels(*args, **kwargs):
        raise AssertionError("setting the target opened the pixels")

    monkeypatch.setattr("baymorph.pipeline.advisor.open_observation", refuse_pixels)

    assert advisor.set_target_position(399.0, 0.0) == (399.0, 0.0)

    with pytest.raises(ValueError, match="inside the FITS image"):
        advisor.set_target_position(400.0, 0.0)


def test_model_screening_reads_the_region_about_its_target(tmp_path):
    """The screening converts its target into the region it reads."""
    _mosaic(tmp_path / "mosaic.fits")
    advisor = SingleObservationModelAdvisor(
        tmp_path / "mosaic.fits",
        target_position={"x": 330.0, "y": 70.0},
        region={"size": 10},
        online=False,
    )
    whole = SingleObservationModelAdvisor(
        tmp_path / "mosaic.fits", target_position={"x": 330.0, "y": 70.0}, online=False
    )
    classifier = SimpleNamespace(screen=lambda features: features)

    in_region = advisor.screen_models(classifier, psf_fwhm_pixels=2.0)
    on_whole = whole.screen_models(classifier, psf_fwhm_pixels=2.0)

    assert in_region.shape == on_whole.shape == (300, 400)
    assert (in_region.target_x, in_region.target_y) == (330.0, 70.0)
    assert np.all(np.isfinite(in_region.features.values))


def test_the_config_command_declares_the_region(tmp_path, monkeypatch, capsys):
    """The command line writes data.region and says what will be read."""
    _mosaic(tmp_path / "mosaic.fits")
    base = [
        "baymorph",
        "config",
        str(tmp_path / "mosaic.fits"),
        "--target-x",
        "330",
        "--target-y",
        "70",
        "--psf",
        "gaussian",
        "--gaussian-fwhm-pixels",
        "2",
        "--uniform-prior",
        "single_sersic.flux",
        "1",
        "1000",
    ]

    monkeypatch.setattr(
        "sys.argv", [*base, "--region-size", "10", "--output", str(tmp_path / "a.yaml")]
    )
    assert main() == 0
    assert yaml.safe_load((tmp_path / "a.yaml").read_text())["data"]["region"] == {
        "size": 10.0
    }
    assert "within a working region of 100 x 100 pixels" in capsys.readouterr().out

    monkeypatch.setattr(
        "sys.argv", [*base, "--whole-image", "--output", str(tmp_path / "b.yaml")]
    )
    assert main() == 0
    assert yaml.safe_load((tmp_path / "b.yaml").read_text())["data"]["region"] == (
        "whole"
    )
    assert "is read whole" in capsys.readouterr().out

    monkeypatch.setattr("sys.argv", [*base, "--whole-image", "--region-size", "10"])

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 2


def test_the_original_image_of_a_region_is_drawn_in_whole_image_pixels():
    """The first panel spans the region in pixels of the whole image."""
    region = {"xmin": 100, "xmax": 160, "ymin": 20, "ymax": 60}
    cutout = np.ones((10, 10))

    fig, axes = plot_pipeline_fit_maps(
        original_cutout=cutout,
        prepared_cutout=cutout,
        model=cutout,
        norm_residual=np.zeros((10, 10)),
        original_image=np.ones((40, 60)),
        cutout_bounds={"xmin": 120, "xmax": 130, "ymin": 30, "ymax": 40},
        working_region=region,
    )

    assert tuple(np.asarray(axes)[0, 0].images[0].get_extent()) == (
        99.5,
        159.5,
        19.5,
        59.5,
    )

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_a_supplied_observation_must_come_from_the_configured_region(tmp_path):
    """The identity refuses an observation read within another region."""
    _galaxy_image(tmp_path / "image.fits")
    config = _preparation_config(tmp_path, "region", {"size": 13.0})
    other = open_configured_observation({**config["data"], "region": {"size": 14.0}})

    with pytest.raises(ValueError, match="another working region"):
        prepared_identity.build_prepared_input_identity(config, observation=other)
