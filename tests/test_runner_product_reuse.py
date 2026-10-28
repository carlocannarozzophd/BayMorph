"""Tests for reuse of accepted preparation and PSF products by the runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy.io import fits

from baymorph import BayMorphRunner
from baymorph.pipeline import (
    pipeline_paths,
    resolve_config,
    save_empirical_psf,
    save_empirical_psf_metadata,
)
from baymorph.pipeline.psf_candidates import (
    PSFCandidate,
    save_psf_candidates,
)
from baymorph.pipeline.psf_construction import (
    EMPIRICAL_PSF_PRODUCT_SCHEMA_VERSION,
    build_empirical_psf_provenance,
    empirical_psf_centroid_tolerance_pixels,
    validate_empirical_psf,
)
from baymorph.rendering.effective_psf import SUBPIXEL_FACTOR


PACKAGED_SIMULATED_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "baymorph"
    / "resources"
    / "configs"
    / "simulated_galaxy.yaml"
)


def _simulated_runner(
    tmp_path: Path,
    *,
    overwrite: bool,
) -> BayMorphRunner:
    """Return a runner for the packaged simulated galaxy in a temporary tree."""
    config = yaml.safe_load(
        PACKAGED_SIMULATED_CONFIG.read_text(
            encoding="utf-8",
        )
    )

    config["outputs"] = {
        "root": str(tmp_path / "runs"),
        "overwrite": overwrite,
    }
    config["data"]["image_path"] = str(
        tmp_path
        / "runs"
        / config["object"]["name"]
        / "input"
        / f"{config['object']['name']}.fits"
    )

    return BayMorphRunner(
        config,
        verbose=False,
    )


def _preparation_state(
    runner: BayMorphRunner,
) -> dict[str, int]:
    """Return modification times of the persisted preparation products."""
    directory = pipeline_paths(runner.config)["image_preparation"]["dir"]

    return {
        path.name: path.stat().st_mtime_ns
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def _empirical_psf_runner(
    tmp_path: Path,
    *,
    point_source_screen: str = "none",
) -> tuple[BayMorphRunner, np.ndarray]:
    """Return a runner whose accepted empirical PSF already exists on disk.

    The candidates record no provenance, as those of BayMorph 0.5.1 and
    earlier, which selected the stars by shape only.
    """
    image_path = tmp_path / "image.fits"
    fits.writeto(
        image_path,
        np.arange(81, dtype=float).reshape(9, 9),
    )

    config = {
        "object": {"name": "example_galaxy"},
        "data": {
            "image_path": str(image_path),
            "target_position": {"x": 4.0, "y": 4.0},
            "cutout": {"mode": "fixed", "size": 9},
        },
        "outputs": {
            "root": str(tmp_path / "runs"),
            "overwrite": True,
        },
        "psf": {
            "type": "empirical",
            "empirical": {
                "candidate_selection": {"point_source_screen": point_source_screen}
            },
        },
        "fit": {
            "models": ["single_sersic"],
            "priors": {
                "single_sersic": {
                    "flux": {
                        "type": "uniform",
                        "lower": 1.0,
                        "upper": 10.0,
                    },
                },
            },
        },
    }

    resolved = resolve_config(config)

    psf = np.zeros((9, 9), dtype=float)
    psf[4, 4] = 1.0

    save_empirical_psf(
        resolved,
        psf,
        subpixel_kernels=np.broadcast_to(
            psf,
            (SUBPIXEL_FACTOR, SUBPIXEL_FACTOR, *psf.shape),
        ).copy(),
    )
    save_psf_candidates(
        resolved,
        [
            PSFCandidate(
                x=4.0,
                y=4.0,
                half_size=4,
            )
        ],
    )
    save_empirical_psf_metadata(
        resolved,
        {
            "schema_version": EMPIRICAL_PSF_PRODUCT_SCHEMA_VERSION,
            "type": "empirical",
            "validation": validate_empirical_psf(
                psf,
                centroid_tolerance_pixels=(
                    empirical_psf_centroid_tolerance_pixels(resolved)
                ),
            ),
            "provenance": build_empirical_psf_provenance(resolved),
        },
    )

    config["outputs"]["overwrite"] = False

    return (
        BayMorphRunner(
            config,
            verbose=False,
        ),
        psf,
    )


def test_prepare_image_reuses_accepted_products(
    tmp_path: Path,
) -> None:
    """A completed preparation is loaded again, not rebuilt."""
    runner = _simulated_runner(
        tmp_path,
        overwrite=False,
    )
    runner.simulate()

    built = runner.prepare_image()
    state = _preparation_state(runner)

    reused = runner.prepare_image()

    assert _preparation_state(runner) == state
    assert np.array_equal(
        reused.prepared_data,
        built.prepared_data,
    )
    assert np.array_equal(
        reused.noise,
        built.noise,
    )
    assert np.array_equal(
        reused.fitting_mask,
        built.fitting_mask,
    )


def test_prepare_image_rebuilds_when_overwrite_is_enabled(
    tmp_path: Path,
) -> None:
    """Explicit overwrite keeps rebuilding the preparation products."""
    runner = _simulated_runner(
        tmp_path,
        overwrite=True,
    )
    runner.simulate()

    runner.prepare_image()
    state = _preparation_state(runner)

    runner.prepare_image()

    assert _preparation_state(runner) != state


def test_prepare_image_rejects_products_from_different_inputs(
    tmp_path: Path,
) -> None:
    """Reuse never bypasses the immutable-input identity check."""
    runner = _simulated_runner(
        tmp_path,
        overwrite=False,
    )
    runner.simulate()
    runner.prepare_image()

    image_path = Path(runner.config["data"]["image_path"])
    fits.writeto(
        image_path,
        fits.getdata(image_path).astype(float) + 1.0,
        overwrite=True,
    )

    with pytest.raises(
        RuntimeError,
        match="do not match the current immutable inputs",
    ):
        runner.prepare_image()


def test_psf_stages_reuse_accepted_products(
    tmp_path: Path,
) -> None:
    """Accepted PSF candidates and kernels are loaded, not rewritten."""
    runner, psf = _empirical_psf_runner(tmp_path)

    paths = pipeline_paths(runner.config)["psf"]
    candidates_state = paths["candidates"].stat().st_mtime_ns
    psf_state = paths["psf_fits"].stat().st_mtime_ns

    candidates = runner.select_psf_candidates()
    reused_psf = runner.build_psf()

    assert paths["candidates"].stat().st_mtime_ns == candidates_state
    assert paths["psf_fits"].stat().st_mtime_ns == psf_state
    assert [(candidate.x, candidate.y) for candidate in candidates] == [
        (4.0, 4.0)
    ]
    assert np.array_equal(
        np.asarray(reused_psf),
        psf,
    )


def test_psf_products_are_not_overwritten_in_place(
    tmp_path: Path,
) -> None:
    """Saving over accepted PSF products fails closed without overwrite."""
    runner, psf = _empirical_psf_runner(tmp_path)

    with pytest.raises(
        FileExistsError,
        match="Empirical PSF output already exists",
    ):
        save_empirical_psf(runner.config, psf)

    with pytest.raises(
        FileExistsError,
        match="PSF candidate output already exists",
    ):
        save_psf_candidates(
            runner.config,
            [
                PSFCandidate(
                    x=1.0,
                    y=1.0,
                    half_size=2,
                )
            ],
        )


def _screened_candidate_runner(
    tmp_path: Path, *, record: bool, point_source_screen: str | None = None
) -> BayMorphRunner:
    """Return a runner whose screened candidates already exist on disk.

    The candidates and their provenance are written as the selection writes
    them under ``point_source_screen`` (the default rule when None), with
    the record of their screen when ``record`` is true. The runner reads
    them under the default rule.
    """
    from baymorph.pipeline.psf_candidate_screen import (
        PSF_CANDIDATE_SCREEN_SCHEMA,
        PSF_CANDIDATE_SCREEN_SCHEMA_VERSION,
        SIZE_SEQUENCE_V1_SHA256,
    )
    from baymorph.pipeline.psf_candidates import (
        save_psf_candidate_screen,
        save_psf_candidates_provenance,
    )

    image_path = tmp_path / "image.fits"
    fits.writeto(image_path, np.zeros((9, 9)))
    config = {
        "object": {"name": "example_galaxy"},
        "data": {"image_path": str(image_path)},
        "outputs": {"root": str(tmp_path / "runs"), "overwrite": False},
    }
    written = json.loads(json.dumps(config))

    if point_source_screen is not None:
        written["psf"] = {
            "empirical": {
                "candidate_selection": {"point_source_screen": point_source_screen}
            }
        }

    resolved = resolve_config(written)
    save_psf_candidates(resolved, [PSFCandidate(x=4.0, y=4.0, half_size=4)])

    if record:
        save_psf_candidate_screen(
            resolved,
            {
                "schema": PSF_CANDIDATE_SCREEN_SCHEMA,
                "schema_version": PSF_CANDIDATE_SCREEN_SCHEMA_VERSION,
                "rule": "size_sequence_v1",
                "constants_sha256": SIZE_SEQUENCE_V1_SHA256,
                "outcome": "found",
                "n_pool": 1,
                "n_members": 1,
                "n_kept": 1,
                "sequence": {"size_pix": 1.3},
                "counts": {"member": 1},
                "candidates": [
                    {"label": None, "x": 4.0, "y": 4.0, "kept": True},
                    {"label": 7, "x": 2.0, "y": 6.0, "kept": False},
                ],
            },
        )

    save_psf_candidates_provenance(resolved)

    return BayMorphRunner(config, verbose=False)


def test_screened_candidates_are_reused_with_the_record_of_their_screen(
    tmp_path: Path,
) -> None:
    runner = _screened_candidate_runner(tmp_path, record=True)

    candidates = runner.select_psf_candidates()

    assert [(candidate.x, candidate.y) for candidate in candidates] == [(4.0, 4.0)]


def test_screened_candidates_without_their_record_are_refused(
    tmp_path: Path,
) -> None:
    runner = _screened_candidate_runner(tmp_path, record=False)

    with pytest.raises(ValueError, match="psf_candidate_screen.json is missing"):
        runner.select_psf_candidates()


def test_candidates_selected_before_the_screen_are_refused_by_default(
    tmp_path: Path,
) -> None:
    """Candidates selected by shape only are not reused under the screen.

    Only the screen differs, so the refusal names the declaration that
    reuses them; it never asks for an overwrite of accepted products.
    """
    from baymorph.pipeline.psf_candidate_screen import PointSourceScreenChangeError

    runner = _screened_candidate_runner(
        tmp_path, record=False, point_source_screen="none"
    )

    with pytest.raises(PointSourceScreenChangeError) as error:
        runner.select_psf_candidates()

    message = str(error.value)

    assert "selected without the point-source screen" in message
    assert "as BayMorph does by default" in message
    assert "point_source_screen: none to reuse them unchanged" in message
    assert "overwrite" not in message


def test_candidates_of_earlier_releases_are_reused_only_without_the_screen(
    tmp_path: Path,
) -> None:
    """A 0.5.1 set, without candidate provenance, binds only the none rule."""
    from baymorph.pipeline.psf_candidate_screen import PointSourceScreenChangeError

    (tmp_path / "neutral").mkdir()
    (tmp_path / "screened").mkdir()
    runner, _ = _empirical_psf_runner(tmp_path / "neutral")

    assert [(c.x, c.y) for c in runner.select_psf_candidates()] == [(4.0, 4.0)]

    legacy, _ = _empirical_psf_runner(tmp_path / "screened")
    config = legacy.config
    config["psf"]["empirical"]["candidate_selection"].pop("point_source_screen")
    screened = BayMorphRunner(config, verbose=False)

    # The accepted PSF binds its stars; only the screen differs, so the
    # refusal names the declaration that reuses them, not an overwrite.
    with pytest.raises(PointSourceScreenChangeError) as error:
        screened.select_psf_candidates()

    assert "empirical PSF and its stars were selected without" in str(error.value)
    assert "overwrite" not in str(error.value)

    with pytest.raises(PointSourceScreenChangeError):
        screened.build_psf()


def test_a_screen_record_of_other_candidates_is_refused(tmp_path: Path) -> None:
    """The record must list the saved candidates as the stars it kept."""
    runner = _screened_candidate_runner(tmp_path, record=True)
    save_psf_candidates(
        {**runner.config, "outputs": {**runner.config["outputs"], "overwrite": True}},
        [PSFCandidate(x=5.0, y=4.0, half_size=4)],
    )

    with pytest.raises(ValueError, match="does not list the saved PSF candidates"):
        runner.select_psf_candidates()
