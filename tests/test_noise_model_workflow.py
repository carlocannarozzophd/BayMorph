"""End-to-end fit of a simulated image with the noise model of AD-047.

The run goes through the production path of ``baymorph run``: simulation,
preparation, the sky correlation, its nearest-neighbour factorization and a
Nautilus fit of the single-observation path. It takes about a minute and is
excluded from ``scripts/test_fast.sh``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import yaml

from baymorph import BayMorphRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SIZE = 101


def _config() -> dict:
    """Return the shipped simulated example, reduced for a fast fit."""
    config = copy.deepcopy(
        yaml.safe_load(
            (PROJECT_ROOT / "configs" / "simulated_galaxy.yaml").read_text(
                encoding="utf-8"
            )
        )
    )
    centre = SIZE // 2
    config["object"]["name"] = "workflow"
    config["data"]["image_path"] = "runs/workflow/input/workflow.fits"
    config["data"]["cutout"]["size"] = SIZE
    config["data"]["target_position"] = {"x": centre, "y": centre}
    config["simulation"]["shape"] = [SIZE, SIZE]
    config["simulation"]["truth"]["x0"] = float(centre)
    config["simulation"]["truth"]["y0"] = float(centre)
    config["outputs"]["root"] = "runs"
    config["fit"]["sampler"] = {
        "name": "nautilus",
        "init": {"n_live": 30, "seed": 3},
        "run": {"n_eff": 50, "f_live": 0.2},
        "execution": {"processes": 1, "progress": False},
    }

    return config


def test_simulated_image_is_fitted_with_the_noise_model_of_ad047(
    tmp_path,
    monkeypatch,
):
    config = _config()
    monkeypatch.chdir(tmp_path)
    Path("workflow.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    runner = BayMorphRunner("workflow.yaml", verbose=False)
    runner.simulate()
    runner.prepare_image()
    runner.fit()

    fit_directory = Path("runs") / "workflow" / "fits" / "single_sersic"
    summary = json.loads((fit_directory / "fit_summary.json").read_text())
    likelihood = summary["likelihood"]

    assert (fit_directory / "COMPLETE.json").is_file()
    assert likelihood["type"] == "correlated_gaussian"
    assert likelihood["marginal_model"] == "prepared_sky_scale"
    assert likelihood["effective_gain"]["source"] == "absent"
    assert likelihood["correlation_estimator"] == "masked_sky_autocovariance_v1"
    assert likelihood["calibration_schema_version"] == 3
    assert likelihood["vecchia_factorization_schema_version"] == 2
    assert likelihood["vecchia_max_neighbors"] == 24
    assert likelihood["normalization_complete"] is True
    assert likelihood["fit_statistic"] == "nearest_neighbor_innovation_quadratic"

    # The simulated noise is white with the scale of the read noise, which
    # the noise model reproduces.
    assert abs(likelihood["effective_gain"]["sky_scale"] - 2.0) < 0.1
    assert 0.9 < summary["reduced_chi2"] < 1.1

    with np.load(fit_directory / "posterior.npz", allow_pickle=False) as posterior:
        names = list(posterior["parameter_names"])
        samples = np.asarray(posterior["samples"])
        weights = np.asarray(posterior["weights"])

    flux = samples[:, names.index("flux")]
    mean = float(np.sum(weights * flux) / np.sum(weights))

    assert abs(mean / config["simulation"]["truth"]["flux"] - 1.0) < 0.05
