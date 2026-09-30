"""High-level pipeline workflow entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from baymorph.pipeline.image_preparation import prepare_real_image_from_config
from baymorph.pipeline.psf_candidates import run_psf_candidate_selection_from_config
from baymorph.pipeline.psf_construction import run_psf_construction_from_config
from baymorph.pipeline.real_fit import run_real_fit_from_config


def run_image_preparation(
    config: str | Path | dict[str, Any],
):
    """Run the image-preparation stage.

    Parameters
    ----------
    config : dict, str, or pathlib.Path
        Configuration dictionary or path to a YAML configuration file.

    Returns
    -------
    PreparedObservation
        Prepared image products.
    """
    return prepare_real_image_from_config(config)


def run_psf_candidate_selection(
    config: str | Path | dict[str, Any],
):
    """Run empirical PSF candidate selection.

    Parameters
    ----------
    config : dict, str, or pathlib.Path
        Configuration dictionary or path to a YAML configuration file.

    Returns
    -------
    list of PSFCandidate
        Selected empirical PSF candidates.
    """
    return run_psf_candidate_selection_from_config(config)


def run_psf_construction(
    config: str | Path | dict[str, Any],
):
    """Run empirical PSF construction.

    Parameters
    ----------
    config : dict, str, or pathlib.Path
        Configuration dictionary or path to a YAML configuration file.

    Returns
    -------
    ndarray
        Constructed empirical PSF image.
    """
    return run_psf_construction_from_config(config)


def run_real_fit(
    config: str | Path | dict[str, Any],
):
    """Run Bayesian model fitting.

    Parameters
    ----------
    config : dict, str, or pathlib.Path
        Configuration dictionary or path to a YAML configuration file.

    Returns
    -------
    RealFitProducts or dict
        Fitting products for one or more fitted models.
    """
    return run_real_fit_from_config(config)
